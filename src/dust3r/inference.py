import tqdm
import torch
import torch.nn.functional as F
from dust3r.utils.device import to_cpu, collate_with_cat
from dust3r.utils.misc import invalid_to_nans
from dust3r.utils.geometry import depthmap_to_pts3d, geotrf
from dust3r.model import ARCroco3DStereo
from dust3r.smpl_model import SMPLModel
from accelerate import Accelerator
import re


def custom_sort_key(key):
    text = key.split("/")
    if len(text) > 1:
        text, num = text[0], text[-1]
        return (text, int(num))
    else:
        return (key, -1)


def merge_chunk_dict(old_dict, curr_dict, add_number):
    new_dict = {}
    for key, value in curr_dict.items():

        match = re.search(r"(\d+)$", key)
        if match:

            num_part = int(match.group()) + add_number

            new_key = re.sub(r"(\d+)$", str(num_part), key, 1)
            new_dict[new_key] = value
        else:
            new_dict[key] = value
    new_dict = old_dict | new_dict
    return {k: new_dict[k] for k in sorted(new_dict.keys(), key=custom_sort_key)}


def _interleave_imgs(img1, img2):
    res = {}
    for key, value1 in img1.items():
        value2 = img2[key]
        if isinstance(value1, torch.Tensor):
            value = torch.stack((value1, value2), dim=1).flatten(0, 1)
        else:
            value = [x for pair in zip(value1, value2) for x in pair]
        res[key] = value
    return res


def make_batch_symmetric(batch):
    view1, view2 = batch
    view1, view2 = (_interleave_imgs(view1, view2), _interleave_imgs(view2, view1))
    return view1, view2


def _unwrap_model(model):
    return getattr(model, "module", model)


def _as_shot_label(view, ref_tensor):
    shot_label = view.get("shot_label", None)
    if shot_label is None:
        return None
    if torch.is_tensor(shot_label):
        shot_label = shot_label.to(device=ref_tensor.device)
    else:
        shot_label = torch.as_tensor(shot_label, device=ref_tensor.device)
    shot_label = shot_label.to(dtype=ref_tensor.dtype).reshape(-1)
    if shot_label.numel() == 1 and ref_tensor.numel() > 1:
        shot_label = shot_label.expand(ref_tensor.numel())
    return shot_label


def _compute_shot_bce_loss(batch, preds, model):
    base_model = _unwrap_model(model)
    weight = float(getattr(base_model, "shot_loss_weight", 0.0))
    if weight <= 0:
        return None, {}

    logits, labels = [], []
    for view, pred in zip(batch, preds):
        shot_logit = pred.get("shot_logit", None)
        if shot_logit is None:
            continue
        shot_logit = shot_logit.reshape(-1).float()
        shot_label = _as_shot_label(view, shot_logit)
        if shot_label is None:
            continue
        logits.append(shot_logit)
        labels.append(shot_label.float())

    if not logits:
        return None, {}

    shot_logits = torch.cat(logits, dim=0)
    shot_labels = torch.cat(labels, dim=0)
    shot_bce = F.binary_cross_entropy_with_logits(shot_logits, shot_labels)

    with torch.no_grad():
        shot_prob = torch.sigmoid(shot_logits)
        shot_pred = shot_prob >= 0.5
        shot_acc = (shot_pred == (shot_labels >= 0.5)).float().mean()
        pos_mask = shot_labels >= 0.5
        neg_mask = ~pos_mask
        shot_prob_pos = shot_prob[pos_mask].mean() if pos_mask.any() else shot_prob.new_tensor(0.0)
        shot_prob_neg = shot_prob[neg_mask].mean() if neg_mask.any() else shot_prob.new_tensor(0.0)

    details = {
        "shot_bce": float(shot_bce.detach()),
        "shot_bce_weighted": float((shot_bce * weight).detach()),
        "shot_acc": float(shot_acc.detach()),
        "shot_prob_pos": float(shot_prob_pos.detach()),
        "shot_prob_neg": float(shot_prob_neg.detach()),
    }
    return shot_bce * weight, details


# **========== Layer 2 原始代码备份：无连续帧 q_t 能量正则 ==========**
# def _compute_shot_q0_loss(batch, preds, model):
#     return None, {}
# **========== 结束 ==========**


def _compute_shot_q0_loss(batch, preds, model):
    base_model = _unwrap_model(model)
    weight = float(getattr(base_model, "shot_q0_loss_weight", 0.0))
    if weight <= 0:
        return None, {}

    energies, labels = [], []
    for view, pred in zip(batch, preds):
        shot_energy = pred.get("shot_q_energy", None)
        if shot_energy is None:
            continue
        shot_energy = shot_energy.reshape(-1).float()
        shot_label = _as_shot_label(view, shot_energy)
        if shot_label is None:
            continue
        energies.append(shot_energy)
        labels.append(shot_label.float())

    if not energies:
        return None, {}

    shot_energy = torch.cat(energies, dim=0)
    shot_labels = torch.cat(labels, dim=0)
    cont_mask = (shot_labels < 0.5).float()
    denom = cont_mask.sum().clamp_min(1.0)
    q0_loss = (shot_energy * cont_mask).sum() / denom

    with torch.no_grad():
        jump_mask = shot_labels >= 0.5
        cont_energy = q0_loss.detach()
        jump_energy = shot_energy[jump_mask].mean() if jump_mask.any() else shot_energy.new_tensor(0.0)

    details = {
        "shot_q0_loss": float(q0_loss.detach()),
        "shot_q0_loss_weighted": float((q0_loss * weight).detach()),
        "shot_q_energy_cont": float(cont_energy.detach()),
        "shot_q_energy_jump": float(jump_energy.detach()),
    }
    return q0_loss * weight, details


def _add_aux_loss(loss, aux_loss, aux_details):
    if loss is None or aux_loss is None:
        return loss
    main_loss, loss_details = loss
    main_loss = main_loss + aux_loss
    loss_details.update(aux_details)
    return main_loss, loss_details


def loss_of_one_batch(
    batch,
    model,
    criterion,
    accelerator: Accelerator,
    symmetrize_batch=False,
    use_amp=False,
    ret=None,
    img_mask=None,
    inference=False,
    smpl_model: SMPLModel = None
):
    if len(batch) > 2:
        assert (
            symmetrize_batch is False
        ), "cannot symmetrize batch with more than 2 views"
    if symmetrize_batch:
        batch = make_batch_symmetric(batch)

    with torch.cuda.amp.autocast(enabled=not inference):
        if inference:
            output, state_args = model(batch, ret_state=True, inference=True)
            preds, batch = output.ress, output.views
            result = dict(views=batch, pred=preds)
            return result[ret] if ret else result, state_args
        else:
            smpl_model.update_smpl_gt(batch)
            output = model(batch)
            preds, batch = output.ress, output.views

        # **========== Layer 1 原始代码备份：训练 loss 仅来自主 criterion ==========**
        # with torch.cuda.amp.autocast(enabled=False):
        #     loss = criterion(batch, preds) if criterion is not None else None
        # **========== 结束 ==========**
        with torch.cuda.amp.autocast(enabled=False):
            loss = criterion(batch, preds) if criterion is not None else None
            if loss is not None:
                shot_loss, shot_details = _compute_shot_bce_loss(batch, preds, model)
                loss = _add_aux_loss(loss, shot_loss, shot_details)
                q0_loss, q0_details = _compute_shot_q0_loss(batch, preds, model)
                loss = _add_aux_loss(loss, q0_loss, q0_details)

    result = dict(views=batch, pred=preds, loss=loss)
    return result[ret] if ret else result

@torch.no_grad()
def inference(groups, model, device, verbose=True):
    ignore_keys = set(
        ["depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"]
    )
    for view in groups:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            if isinstance(view[name], tuple) or isinstance(view[name], list):
                view[name] = [x.to(device, non_blocking=True) for x in view[name]]
            else:
                view[name] = view[name].to(device, non_blocking=True)

    if verbose:
        print(f">> Inference with model on {len(groups)} image/raymaps")

    res, state_args = loss_of_one_batch(groups, model, None, None, inference=True)
    result = to_cpu(res)
    return result, state_args


@torch.no_grad()
def inference_step(view, state_args, model, device, verbose=True):
    ignore_keys = set(
        ["depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"]
    )
    for name in view.keys():  # pseudo_focal
        if name in ignore_keys:
            continue
        if isinstance(view[name], tuple) or isinstance(view[name], list):
            view[name] = [x.to(device, non_blocking=True) for x in view[name]]
        else:
            view[name] = view[name].to(device, non_blocking=True)

    with torch.cuda.amp.autocast(enabled=False):
        state_feat, state_pos, init_state_feat, mem, init_mem = state_args
        pred, _ = model.inference_step(
            view, state_feat, state_pos, init_state_feat, mem, init_mem
        )

    res = dict(pred=pred)
    result = to_cpu(res)
    return result


@torch.no_grad()
def inference_recurrent(groups, model, device, verbose=True):
    ignore_keys = set(
        ["depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"]
    )
    for view in groups:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            if isinstance(view[name], tuple) or isinstance(view[name], list):
                view[name] = [x.to(device, non_blocking=True) for x in view[name]]
            else:
                view[name] = view[name].to(device, non_blocking=True)

    if verbose:
        print(f">> Inference with model on {len(groups)} image/raymaps")

    with torch.cuda.amp.autocast(enabled=False):
        preds, batch, state_args = model.forward_recurrent(
            groups, device, ret_state=True
        )
        res = dict(views=batch, pred=preds)
    result = to_cpu(res)
    return result, state_args

@torch.no_grad()
def inference_recurrent_lighter(groups, model, device, verbose=True, is_naive=False, use_ttt3r=False):
    if verbose:
        print(f">> Inference with model on {len(groups)} image/raymaps")

    with torch.cuda.amp.autocast(enabled=False):
        if is_naive:
            preds, batch, state_args = model.forward_recurrent_lighter_naive(
            groups, device, ret_state=True, use_ttt3r=use_ttt3r
        )
        else:
            preds, batch, state_args = model.forward_recurrent_lighter(
                groups, device, ret_state=True, use_ttt3r=use_ttt3r
            )
        res = dict(views=batch, pred=preds)
    return res, state_args

def check_if_same_size(pairs):
    shapes1 = [img1["img"].shape[-2:] for img1, img2 in pairs]
    shapes2 = [img2["img"].shape[-2:] for img1, img2 in pairs]
    return all(shapes1[0] == s for s in shapes1) and all(
        shapes2[0] == s for s in shapes2
    )


def get_pred_pts3d(gt, pred, use_pose=False, inplace=False):
    if "depth" in pred and "pseudo_focal" in pred:
        try:
            pp = gt["camera_intrinsics"][..., :2, 2]
        except KeyError:
            pp = None
        pts3d = depthmap_to_pts3d(**pred, pp=pp)

    elif "pts3d" in pred:

        pts3d = pred["pts3d"]

    elif "pts3d_in_other_view" in pred:

        assert use_pose is True
        return (
            pred["pts3d_in_other_view"]
            if inplace
            else pred["pts3d_in_other_view"].clone()
        )

    if use_pose:
        camera_pose = pred.get("camera_pose")
        assert camera_pose is not None
        pts3d = geotrf(camera_pose, pts3d)

    return pts3d


def find_opt_scaling(
    gt_pts1,
    gt_pts2,
    pr_pts1,
    pr_pts2=None,
    fit_mode="weiszfeld_stop_grad",
    valid1=None,
    valid2=None,
):
    assert gt_pts1.ndim == pr_pts1.ndim == 4
    assert gt_pts1.shape == pr_pts1.shape
    if gt_pts2 is not None:
        assert gt_pts2.ndim == pr_pts2.ndim == 4
        assert gt_pts2.shape == pr_pts2.shape

    nan_gt_pts1 = invalid_to_nans(gt_pts1, valid1).flatten(1, 2)
    nan_gt_pts2 = (
        invalid_to_nans(gt_pts2, valid2).flatten(1, 2) if gt_pts2 is not None else None
    )

    pr_pts1 = invalid_to_nans(pr_pts1, valid1).flatten(1, 2)
    pr_pts2 = (
        invalid_to_nans(pr_pts2, valid2).flatten(1, 2) if pr_pts2 is not None else None
    )

    all_gt = (
        torch.cat((nan_gt_pts1, nan_gt_pts2), dim=1)
        if gt_pts2 is not None
        else nan_gt_pts1
    )
    all_pr = torch.cat((pr_pts1, pr_pts2), dim=1) if pr_pts2 is not None else pr_pts1

    dot_gt_pr = (all_pr * all_gt).sum(dim=-1)
    dot_gt_gt = all_gt.square().sum(dim=-1)

    if fit_mode.startswith("avg"):

        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
    elif fit_mode.startswith("median"):
        scaling = (dot_gt_pr / dot_gt_gt).nanmedian(dim=1).values
    elif fit_mode.startswith("weiszfeld"):

        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)

        for iter in range(10):

            dis = (all_pr - scaling.view(-1, 1, 1) * all_gt).norm(dim=-1)

            w = dis.clip_(min=1e-8).reciprocal()

            scaling = (w * dot_gt_pr).nanmean(dim=1) / (w * dot_gt_gt).nanmean(dim=1)
    else:
        raise ValueError(f"bad {fit_mode=}")

    if fit_mode.endswith("stop_grad"):
        scaling = scaling.detach()

    scaling = scaling.clip(min=1e-3)

    return scaling
