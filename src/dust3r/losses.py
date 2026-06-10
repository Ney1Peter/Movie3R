from copy import copy, deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dust3r.utils.geometry import (
    inv,
    geotrf,
    normalize_pointcloud_group,
    get_group_pointcloud_center_scale,
    to_euclidean_dist,
)
import numpy as np
from dust3r.utils.camera import (
    pose_encoding_to_camera,
    camera_to_pose_encoding,
    relative_pose_absT_quatR,
)
from dust3r.utils.image import unpad_image
from dust3r.utils import SMPL_Layer

import roma
from tqdm import tqdm

def Sum(*losses_and_masks):
    loss, mask = losses_and_masks[0]
    if loss.ndim > 0:
        # we are actually returning the loss for every pixels
        return losses_and_masks
    else:
        # we are returning the global loss
        for loss2, mask2 in losses_and_masks[1:]:
            loss = loss + loss2
        return loss


def stack_view(ls, k=None):
    if isinstance(ls[0], dict):
        v = torch.stack([g[k] for g in ls], dim=0)
    else:
        v = torch.stack(ls, dim=0)
    return v.view(-1, *v.shape[2:])


def _neg_loss(pred, gt):
  '''
  Code modified from: https://github.com/xingyizhou/CenterNet/blob/4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c/src/lib/models/losses.py#L42
    Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    
  '''
  assert pred.shape == gt.shape

  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  eps = 1e-7

  pos_loss = torch.log(pred + eps) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred + eps) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss


class BaseCriterion(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction


class LLoss(BaseCriterion):
    """L-norm loss"""

    def forward(self, a, b):
        # assert (
        #     a.shape == b.shape and a.ndim >= 2 and 1 <= a.shape[-1] <= 3
        # ), f"Bad shape = {a.shape}"
        dist = self.distance(a, b)
        if self.reduction == "none":
            return dist
        if self.reduction == "sum":
            return dist.sum()
        if self.reduction == "mean":
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f"bad {self.reduction=} mode")

    def distance(self, a, b):
        raise NotImplementedError()


class L21Loss(LLoss):
    """Euclidean distance between 3d points"""

    def distance(self, a, b):
        return torch.norm(a - b, dim=-1)  # normalized L2 distance


L21 = L21Loss()


class MSELoss(LLoss):
    def distance(self, a, b):
        return (a - b) ** 2


MSE = MSELoss()


class L1Loss(LLoss):
    def distance(self, a, b):
        return (a - b).abs()

L1 = L1Loss()


class Criterion(nn.Module):
    def __init__(self, criterion=None):
        super().__init__()
        assert isinstance(
            criterion, BaseCriterion
        ), f"{criterion} is not a proper criterion!"
        self.criterion = copy(criterion)

    def get_name(self):
        return f"{type(self).__name__}({self.criterion})"

    def with_reduction(self, mode="none"):
        res = loss = deepcopy(self)
        while loss is not None:
            assert isinstance(loss, Criterion)
            loss.criterion.reduction = mode  # make it return the loss for each sample
            loss = loss._loss2  # we assume loss is a Multiloss
        return res


class MultiLoss(nn.Module):
    """Easily combinable losses (also keep track of individual loss values):
        loss = MyLoss1() + 0.1*MyLoss2()
    Usage:
        Inherit from this class and override get_name() and compute_loss()
    """

    def __init__(self):
        super().__init__()
        self._alpha = 1
        self._loss2 = None

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

    def __mul__(self, alpha):
        assert isinstance(alpha, (int, float))
        res = copy(self)
        res._alpha = alpha
        return res

    __rmul__ = __mul__  # same

    def __add__(self, loss2):
        assert isinstance(loss2, MultiLoss)
        res = cur = copy(self)
        # find the end of the chain
        while cur._loss2 is not None:
            cur = cur._loss2
        cur._loss2 = loss2
        return res

    def __repr__(self):
        name = self.get_name()
        if self._alpha != 1:
            name = f"{self._alpha:g}*{name}"
        if self._loss2:
            name = f"{name} + {self._loss2}"
        return name

    def forward(self, *args, **kwargs):
        loss = self.compute_loss(*args, **kwargs)
        if isinstance(loss, tuple):
            loss, details = loss
        elif loss.ndim == 0:
            details = {self.get_name(): float(loss)}
        else:
            details = {}
        loss = loss * self._alpha

        if self._loss2:
            loss2, details2 = self._loss2(*args, **kwargs)
            loss = loss + loss2
            details |= details2

        return loss, details


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images"""

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x**2) - mu_x**2
        sigma_y = self.sig_y_pool(y**2) - mu_y**2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x**2 + mu_y**2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class RGBLoss(Criterion, MultiLoss):
    def __init__(self, criterion):
        super().__init__(criterion)
        self.ssim = SSIM()

    def img_loss(self, a, b):
        return self.criterion(a, b)

    def compute_loss(self, gts, preds, **kw):
        gt_rgbs = [gt["img"].permute(0, 2, 3, 1) for gt in gts]
        pred_rgbs = [pred["rgb"] for pred in preds]
        ls = [
            self.img_loss(pred_rgb, gt_rgb)
            for pred_rgb, gt_rgb in zip(pred_rgbs, gt_rgbs)
        ]
        details = {}
        self_name = type(self).__name__
        for i, l in enumerate(ls):
            details[self_name + f"_rgb/{i+1}"] = float(l)
            details[f"pred_rgb_{i+1}"] = pred_rgbs[i]
        rgb_loss = sum(ls) / len(ls)
        return rgb_loss, details


class SMPLLoss(Criterion, MultiLoss):
    def __init__(self, criterion):
        super().__init__(criterion)
        scale = 0.01
        self.alpha_msk = 100.0 * scale
        self.alpha_bce = 10.0 * scale
        self.alpha_rotmat = 100.0 * scale
        self.alpha_shape = 10.0 * scale
        self.alpha_transl = 100.0 * scale
        self.alpha_j3d = 100.0 * scale
        self.alpha_v3d = 100.0 * scale
        self.alpha_j2d = 1.0 * scale
        self.alpha_v2d = 1.0 * scale

        # SMPL layer
        person_center = 'head'
        dict_smpl_layer = {
            'neutral': {
                10: SMPL_Layer(type='smplx', gender='neutral', num_betas=10, kid=False, person_center=person_center),
                11: SMPL_Layer(type='smplx', gender='neutral', num_betas=11, kid=False, person_center=person_center),
                }
            }
        _moduleDict = []
        for k, _smpl_layer in dict_smpl_layer.items():
            for x, y in _smpl_layer.items():
                _moduleDict.append([f"{k}_{x}", deepcopy(y)])
        self.smpl_layer = nn.ModuleDict(_moduleDict)

    def get_name(self):
        return "SMPLLoss"

    def mask_loss(self, gts, preds, masks, ret_pred=False):
        gt_msks = [gt["msk_mhmr"].unsqueeze(-1) for gt in gts]
        pred_msks = [pred["msk"] for pred in preds]
        ls = [
            F.binary_cross_entropy(p[m], g[m])
            for p, g, m in zip(pred_msks, gt_msks, masks)
        ]
        details = {}
        self_name = self.get_name()
        for i, l in enumerate(ls):
            details[self_name + f"_msk/{i+1}"] = float(l)
            if ret_pred:
                details[f"pred_msk_{i+1}"] = pred_msks[i]
        bce = sum(ls) / len(ls)
        return bce, details
        
    def bce(self, gts, preds, masks, ret_pred=False):
        gt_scores = [(gt["smpl_scores"] >= 1).to(int).unsqueeze(-1) for gt in gts]
        pred_scores = [pred["smpl_scores"] for pred in preds]
        ls = [
            _neg_loss(p[m], g[m])
            for p, g, m in zip(pred_scores, gt_scores, masks)
        ]
        details = {}
        self_name = self.get_name()
        for i, l in enumerate(ls):
            details[self_name + f"_scores/{i+1}"] = float(l)
            if ret_pred:
                details[f"pred_smpl_scores_{i+1}"] = pred_scores[i]
        bce = sum(ls) / len(ls)
        return bce, details
    
    def smpl_param_loss(self, gts, preds, masks, k, ret_pred=False):
        if isinstance(gts[0], dict):
            gts = [gt[k] for gt in gts]
        if isinstance(preds[0], dict):
            preds = [pred[k] for pred in preds]
        
        ls = [
            self.criterion(p[m], g[m])
            for p, g, m in zip(preds, gts, masks)
        ]
        details = {}
        self_name = self.get_name()
        k_name = k.split('smpl_')[-1] if 'smpl_' in k else k
        for i, l in enumerate(ls):
            details[self_name + f"_{k_name}/{i+1}"] = float(l)
            if ret_pred:
                details[f"pred_{k}_{i+1}"] = preds[i]
        loss = sum(ls) / len(ls)

        return loss, details

    def point3d_loss(self, gts, preds, gt_t_p, pr_t_ps, masks, k, ret_pred=False):
        if isinstance(gts[0], dict):
            gts = [gt[k] for gt in gts]
        if isinstance(preds[0], dict):
            preds = [pred[k] for pred in preds]
        
        if gt_t_p is None or pr_t_ps is None:
            ls = [
                self.criterion(p[m], g[m])
                for p, g, m in zip(preds, gts, masks)
            ]
        else:
            ls = [
                self.criterion(p[m]-pr_p[m], g[m]-gt_p[m])
                for p, pr_p, g, gt_p, m in zip(preds, pr_t_ps, gts, gt_t_p, masks)
            ]

        details = {}
        self_name = self.get_name()
        k_name = k.split('smpl_')[-1] if 'smpl_' in k else k
        if gt_t_p is None or pr_t_ps is None:
            k_name = "c" + k_name
        for i, l in enumerate(ls):
            details[self_name + f"_{k_name}/{i+1}"] = float(l)
            if ret_pred:
                details[f"pred_{k}_{i+1}"] = preds[i]
        loss = sum(ls) / len(ls)

        return loss, details

    def point2d_loss(self, gts, preds, masks, k, shape=None, ret_pred=False):
        if isinstance(gts[0], dict):
            shape = gts[0]['true_shape'][0]
            gts = [gt[k] for gt in gts]
        if isinstance(preds[0], dict):
            preds = [pred[k] for pred in preds]
        
        valid_mask = [
            ((gt[..., 0] > 0) & (gt[..., 0] < shape[1]) & (gt[..., 1] > 0) & (gt[..., 1] < shape[0]
            )) for gt in gts]

        ls = [
            self.criterion(p[m1.unsqueeze(-1) & m2], g[m1.unsqueeze(-1) & m2])
            for p, g, m1, m2 in zip(preds, gts, masks, valid_mask)
        ]
        details = {}
        self_name = self.get_name()
        k_name = k.split('smpl_')[-1] if 'smpl_' in k else k
        for i, l in enumerate(ls):
            details[self_name + f"_{k_name}/{i+1}"] = float(l)
            if ret_pred:
                details[f"pred_{k}_{i+1}"] = preds[i]
        loss = sum(ls) / len(ls)

        return loss, details

    def compute_loss(self, gts, preds, **kw):
        img_mask_list = [gt["img_mask"] for gt in gts]
        smpl_mask_list = [gt["smpl_mask"] for gt in gts]
        masks_list = [a.unsqueeze(1) & b for a, b in zip(img_mask_list, smpl_mask_list)]

        # Detection loss
        score_loss, score_details = self.bce(gts, preds, img_mask_list, ret_pred=True)

        has_msk = "msk" in preds[0]
        if has_msk:
            msk_loss, msk_details = self.mask_loss(gts, preds, img_mask_list, ret_pred=True)

        K = stack_view(gts, 'camera_intrinsics')
        img_mask = stack_view(img_mask_list,'img_mask').unsqueeze(1)
        smpl_mask = stack_view(smpl_mask_list, 'smpl_mask') * img_mask
        idx_h = torch.where(smpl_mask)
        if int(smpl_mask.sum()) == 0:
            total_loss = self.alpha_bce * score_loss
            details = {
                **score_details,
            }
            if has_msk:
                total_loss += self.alpha_msk * msk_loss
                details.update(msk_details)
            return total_loss, details
    
        # Prediction
        pred_rotmat = stack_view(preds, 'smpl_rotmat')
        pred_rotvec = roma.rotmat_to_rotvec(pred_rotmat[smpl_mask])
        pred_shape = stack_view(preds, 'smpl_shape')
        pred_transl = [pred.pop("smpl_transl") for pred in preds]
        pred_transl = stack_view(pred_transl, 'smpl_transl')
        pred_expression = stack_view(preds, 'smpl_expression')
        
        smpl_out = self.smpl_layer[f"neutral_{pred_shape.shape[-1]}"](
            pred_rotvec, 
            pred_shape[smpl_mask], 
            pred_transl[smpl_mask], 
            None, None, 
            K=K[idx_h[0]], 
            expression=pred_expression[smpl_mask])
        
        pred_smpl = {}
        batch_size = img_mask_list[0].shape[0]
        num_view = len(gts)
        max_humans = smpl_mask.shape[1]

        for k, v in smpl_out.items():
            full_out = torch.zeros(
                num_view * batch_size, max_humans, *v.shape[1:], 
                device=v.device, dtype=v.dtype,
            )
            full_out[smpl_mask] = v
            pred_smpl[k] = full_out.chunk(num_view, dim=0)

        # SMPL-X params
        rotmat_loss, rotmat_details = self.smpl_param_loss(gts, preds, masks_list, "smpl_rotmat")
        transl_loss, transl_details = self.smpl_param_loss(gts, pred_smpl['smpl_transl'], masks_list, "smpl_transl")
        shape_dim = min([gts[0]['smpl_shape'].shape[-1], preds[0]['smpl_shape'].shape[-1]])
        gt_shape = [gt['smpl_shape'][...,:shape_dim] for gt in gts]
        pred_shape = [pred['smpl_shape'][...,:shape_dim] for pred in preds]
        shape_loss, shape_details = self.smpl_param_loss(gt_shape, pred_shape, masks_list, "smpl_shape")

        # 3D points
        gt_transl_pelvis= [gt['smpl_transl_pelvis'][..., None, :] for gt in gts]
        pred_transl_pelvis = pred_smpl['smpl_transl_pelvis']
        j3d_loss, j3d_details = self.point3d_loss(
                                    gts, 
                                    pred_smpl['smpl_j3d'], 
                                    gt_transl_pelvis, 
                                    pred_transl_pelvis, 
                                    masks_list, "smpl_j3d")
        v3d_loss, v3d_details = self.point3d_loss(
                                    gts, 
                                    pred_smpl['smpl_v3d'], 
                                    gt_transl_pelvis, 
                                    pred_transl_pelvis, 
                                    masks_list, "smpl_v3d",
                                    ret_pred=True)
        cj3d_loss, cj3d_details = self.point3d_loss(
                                    gts, 
                                    pred_smpl['smpl_j3d'], 
                                    None, 
                                    None, 
                                    masks_list, "smpl_j3d")
        cv3d_loss, cv3d_details = self.point3d_loss(
                                    gts, 
                                    pred_smpl['smpl_v3d'], 
                                    None, 
                                    None, 
                                    masks_list, "smpl_v3d")
        
        # total loss
        total_loss = self.alpha_bce * score_loss +\
                     self.alpha_rotmat * rotmat_loss +\
                     self.alpha_shape * shape_loss +\
                     self.alpha_transl * transl_loss +\
                     self.alpha_j3d * j3d_loss +\
                     self.alpha_v3d * v3d_loss +\
                     self.alpha_j3d * cj3d_loss +\
                     self.alpha_v3d * cv3d_loss

        details = {
            **score_details,
            **rotmat_details,
            **transl_details,
            **shape_details,
            **j3d_details,
            **v3d_details,
            **cj3d_details,
            **cv3d_details,
        }

        if has_msk:
            total_loss += self.alpha_msk * msk_loss
            details.update(msk_details)

        if cv3d_loss < 1.0:
            # 2D reprojection
            j2d_loss, j2d_details = self.point2d_loss(
                                        gts, 
                                        pred_smpl['smpl_j2d'], 
                                        masks_list, "smpl_j2d")
            v2d_loss, v2d_details = self.point2d_loss(
                                        gts, 
                                        pred_smpl['smpl_v2d'], 
                                        masks_list, "smpl_v2d")
            total_loss += self.alpha_j2d * j2d_loss +\
                          self.alpha_v2d * v2d_loss
            details.update({
                **j2d_details,
                **v2d_details
            })

        return total_loss, details

class NaiveSMPLLoss(SMPLLoss):
    def __init__(self, criterion):
        super().__init__(criterion)
        
    def compute_loss(self, gts, preds, **kw):
        img_mask_list = [gt["img_mask"] for gt in gts]
        smpl_mask_list = [gt["smpl_mask"] for gt in gts]
        masks_list = [a.unsqueeze(1) & b for a, b in zip(img_mask_list, smpl_mask_list)]

        # Detection loss
        score_loss, score_details = self.bce(gts, preds, img_mask_list, ret_pred=True)

        # only for inference SMPL model
        K = stack_view(gts, 'camera_intrinsics')
        img_mask = stack_view(img_mask_list,'img_mask').unsqueeze(1)
        smpl_mask = stack_view(smpl_mask_list, 'smpl_mask') * img_mask
        idx_h = torch.where(smpl_mask)
        if int(smpl_mask.sum()) == 0:
            total_loss = self.alpha_bce * score_loss
            details = {
                **score_details,
            }
            return total_loss, details
    
        # Prediction
        pred_rotmat = stack_view(preds, 'smpl_rotmat')
        pred_rotvec = roma.rotmat_to_rotvec(pred_rotmat[smpl_mask])
        pred_shape = stack_view(preds, 'smpl_shape')
        pred_transl = [pred.pop("smpl_transl") for pred in preds]
        pred_transl = stack_view(pred_transl, 'smpl_transl')
        pred_expression = stack_view(preds, 'smpl_expression')
        
        # Neutral for MHMR
        K_mhmr = stack_view(gts, 'K_mhmr')
        mhmr_img_res = gts[0]["img_mhmr"].shape[-1]
        # fine head uv
        pred_loc = stack_view(preds, 'smpl_loc')
        # Distance 
        dist = pred_transl[smpl_mask][:, 0].unsqueeze(-1)
        dist = to_euclidean_dist(mhmr_img_res, dist, K_mhmr[idx_h[0]])  # use K GT
        smpl_out = self.smpl_layer[f"neutral_{pred_shape.shape[-1]}"](
            pred_rotvec, 
            pred_shape[smpl_mask], 
            None, 
            pred_loc[smpl_mask], 
            dist, 
            K=K_mhmr[idx_h[0]],
            expression=pred_expression[smpl_mask],
            K_to_proj=K[idx_h[0]], # if use K of CUT3R for projection
            )
        
        pred_smpl = {}
        batch_size = img_mask_list[0].shape[0]
        num_view = len(gts)
        max_humans = smpl_mask.shape[1]

        for k, v in smpl_out.items():
            full_out = torch.zeros(
                num_view * batch_size, max_humans, *v.shape[1:], 
                device=v.device, dtype=v.dtype,
            )
            full_out[smpl_mask] = v
            pred_smpl[k] = full_out.chunk(num_view, dim=0)

        # SMPL-X params
        rotmat_loss, rotmat_details = self.smpl_param_loss(gts, preds, masks_list, "smpl_rotmat")
        transl_loss, transl_details = self.smpl_param_loss(gts, pred_smpl['smpl_transl'], masks_list, "smpl_transl")
        shape_dim = min([gts[0]['smpl_shape'].shape[-1], preds[0]['smpl_shape'].shape[-1]])
        gt_shape = [gt['smpl_shape'][...,:shape_dim] for gt in gts]
        pred_shape = [pred['smpl_shape'][...,:shape_dim] for pred in preds]
        shape_loss, shape_details = self.smpl_param_loss(gt_shape, pred_shape, masks_list, "smpl_shape")

        # 3D points
        gt_transl_pelvis= [gt['smpl_transl_pelvis'][..., None, :] for gt in gts]
        pred_transl_pelvis = pred_smpl['smpl_transl_pelvis']
        j3d_loss, j3d_details = self.point3d_loss(
                                    gts, 
                                    pred_smpl['smpl_j3d'], 
                                    gt_transl_pelvis, 
                                    pred_transl_pelvis, 
                                    masks_list, "smpl_j3d")
        v3d_loss, v3d_details = self.point3d_loss(
                                    gts, 
                                    pred_smpl['smpl_v3d'], 
                                    gt_transl_pelvis, 
                                    pred_transl_pelvis, 
                                    masks_list, "smpl_v3d",
                                    ret_pred=True)
        cj3d_loss, cj3d_details = self.point3d_loss(
                                    gts, 
                                    pred_smpl['smpl_j3d'], 
                                    None, 
                                    None, 
                                    masks_list, "smpl_j3d")
        cv3d_loss, cv3d_details = self.point3d_loss(
                                    gts, 
                                    pred_smpl['smpl_v3d'], 
                                    None, 
                                    None, 
                                    masks_list, "smpl_v3d")
        
        # total loss
        total_loss = self.alpha_bce * score_loss +\
                     self.alpha_rotmat * rotmat_loss +\
                     self.alpha_shape * shape_loss +\
                     self.alpha_transl * transl_loss +\
                     self.alpha_j3d * j3d_loss +\
                     self.alpha_v3d * v3d_loss +\
                     self.alpha_j3d * cj3d_loss +\
                     self.alpha_v3d * cv3d_loss

        details = {
            **score_details,
            **rotmat_details,
            **transl_details,
            **shape_details,
            **j3d_details,
            **v3d_details,
            **cj3d_details,
            **cv3d_details,
        }

        if cv3d_loss < 1.0:
            # 2D reprojection
            j2d_loss, j2d_details = self.point2d_loss(
                                        gts, 
                                        pred_smpl['smpl_j2d'], 
                                        masks_list, "smpl_j2d")
            v2d_loss, v2d_details = self.point2d_loss(
                                        gts, 
                                        pred_smpl['smpl_v2d'], 
                                        masks_list, "smpl_v2d")
            total_loss += self.alpha_j2d * j2d_loss +\
                          self.alpha_v2d * v2d_loss
            details.update({
                **j2d_details,
                **v2d_details
            })

        return total_loss, details

class DepthScaleShiftInvLoss(BaseCriterion):
    """scale and shift invariant loss"""

    def __init__(self, reduction="none"):
        super().__init__(reduction)

    def forward(self, pred, gt, mask):
        assert pred.shape == gt.shape and pred.ndim == 3, f"Bad shape = {pred.shape}"
        dist = self.distance(pred, gt, mask)
        # assert dist.ndim == a.ndim - 1  # one dimension less
        if self.reduction == "none":
            return dist
        if self.reduction == "sum":
            return dist.sum()
        if self.reduction == "mean":
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f"bad {self.reduction=} mode")

    def normalize(self, x, mask):
        x_valid = x[mask]
        splits = mask.sum(dim=(1, 2)).tolist()
        x_valid_list = torch.split(x_valid, splits)
        shift = [x.mean() for x in x_valid_list]
        x_valid_centered = [x - m for x, m in zip(x_valid_list, shift)]
        scale = [x.abs().mean() for x in x_valid_centered]
        scale = torch.stack(scale)
        shift = torch.stack(shift)
        x = (x - shift.view(-1, 1, 1)) / scale.view(-1, 1, 1).clamp(min=1e-6)
        return x

    def distance(self, pred, gt, mask):
        pred = self.normalize(pred, mask)
        gt = self.normalize(gt, mask)
        return torch.abs((pred - gt)[mask])


class ScaleInvLoss(BaseCriterion):
    """scale invariant loss"""

    def __init__(self, reduction="none"):
        super().__init__(reduction)

    def forward(self, pred, gt, mask):
        assert pred.shape == gt.shape and pred.ndim == 4, f"Bad shape = {pred.shape}"
        dist = self.distance(pred, gt, mask)
        # assert dist.ndim == a.ndim - 1  # one dimension less
        if self.reduction == "none":
            return dist
        if self.reduction == "sum":
            return dist.sum()
        if self.reduction == "mean":
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f"bad {self.reduction=} mode")

    def distance(self, pred, gt, mask):
        pred_norm_factor = (torch.norm(pred, dim=-1) * mask).sum(dim=(1, 2)) / mask.sum(
            dim=(1, 2)
        ).clamp(min=1e-6)
        gt_norm_factor = (torch.norm(gt, dim=-1) * mask).sum(dim=(1, 2)) / mask.sum(
            dim=(1, 2)
        ).clamp(min=1e-6)
        pred = pred / pred_norm_factor.view(-1, 1, 1, 1).clamp(min=1e-6)
        gt = gt / gt_norm_factor.view(-1, 1, 1, 1).clamp(min=1e-6)
        return torch.norm(pred - gt, dim=-1)[mask]


class Regr3DPose(Criterion, MultiLoss):
    """Ensure that all 3D points are correct.
    Asymmetric loss: view1 is supposed to be the anchor.

    P1 = RT1 @ D1
    P2 = RT2 @ D2
    loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
    loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
          = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """

    def __init__(
        self,
        criterion,
        norm_mode="?avg_dis",
        gt_scale=False,
        sky_loss_value=2,
        max_metric_scale=False,
    ):
        super().__init__(criterion)
        if norm_mode.startswith("?"):
            # do no norm pts from metric scale datasets
            self.norm_all = False
            self.norm_mode = norm_mode[1:]
        else:
            self.norm_all = True
            self.norm_mode = norm_mode
        self.gt_scale = gt_scale

        self.sky_loss_value = sky_loss_value
        self.max_metric_scale = max_metric_scale

    def get_norm_factor_point_cloud(
        self, pts_self, pts_cross, valids, conf_self, conf_cross, norm_self_only=False
    ):
        if norm_self_only:
            norm_factor = normalize_pointcloud_group(
                pts_self, self.norm_mode, valids, conf_self, ret_factor_only=True
            )
        else:
            pts = [torch.cat([x, y], dim=2) for x, y in zip(pts_self, pts_cross)]
            valids = [torch.cat([x, x], dim=2) for x in valids]
            confs = [torch.cat([x, y], dim=2) for x, y in zip(conf_self, conf_cross)]
            norm_factor = normalize_pointcloud_group(
                pts, self.norm_mode, valids, confs, ret_factor_only=True
            )
        return norm_factor

    def get_norm_factor_poses(self, gt_trans, pr_trans, not_metric_mask):

        if self.norm_mode and not self.gt_scale:
            gt_trans = [x[:, None, None, :].clone() for x in gt_trans]
            valids = [torch.ones_like(x[..., 0], dtype=torch.bool) for x in gt_trans]
            norm_factor_gt = (
                normalize_pointcloud_group(
                    gt_trans,
                    self.norm_mode,
                    valids,
                    ret_factor_only=True,
                )
                .squeeze(-1)
                .squeeze(-1)
            )
        else:
            norm_factor_gt = torch.ones(
                len(gt_trans), dtype=gt_trans[0].dtype, device=gt_trans[0].device
            )

        norm_factor_pr = norm_factor_gt.clone()
        if self.norm_mode and not_metric_mask.sum() > 0 and not self.gt_scale:
            pr_trans_not_metric = [
                x[not_metric_mask][:, None, None, :].clone() for x in pr_trans
            ]
            valids = [
                torch.ones_like(x[..., 0], dtype=torch.bool)
                for x in pr_trans_not_metric
            ]
            norm_factor_pr_not_metric = (
                normalize_pointcloud_group(
                    pr_trans_not_metric,
                    self.norm_mode,
                    valids,
                    ret_factor_only=True,
                )
                .squeeze(-1)
                .squeeze(-1)
            )
            norm_factor_pr[not_metric_mask] = norm_factor_pr_not_metric
        return norm_factor_gt, norm_factor_pr

    def get_all_pts3d(
        self,
        gts,
        preds,
        dist_clip=None,
        norm_self_only=False,
        norm_pose_separately=False,
        eps=1e-3,
        camera1=None,
    ):
        # everything is normalized w.r.t. camera of view1
        in_camera1 = inv(gts[0]["camera_pose"]) if camera1 is None else inv(camera1)
        gt_pts_self = [geotrf(inv(gt["camera_pose"]), gt["pts3d"]) for gt in gts]
        gt_pts_cross = [geotrf(in_camera1, gt["pts3d"]) for gt in gts]
        valids = [gt["valid_mask"].clone() for gt in gts]
        camera_only = gts[0]["camera_only"]

        if dist_clip is not None:
            # points that are too far-away == invalid
            dis = [gt_pt.norm(dim=-1) for gt_pt in gt_pts_cross]
            valids = [valid & (dis <= dist_clip) for valid, dis in zip(valids, dis)]

        pr_pts_self = [pred["pts3d_in_self_view"] for pred in preds]
        pr_pts_cross = [pred["pts3d_in_other_view"] for pred in preds]
        conf_self = [torch.log(pred["conf_self"]).detach().clip(eps) for pred in preds]
        conf_cross = [torch.log(pred["conf"]).detach().clip(eps) for pred in preds]

        if not self.norm_all:
            if self.max_metric_scale:
                B = valids[0].shape[0]
                dist = [
                    torch.where(valid, torch.linalg.norm(gt_pt_cross, dim=-1), 0).view(
                        B, -1
                    )
                    for valid, gt_pt_cross in zip(valids, gt_pts_cross)
                ]
                for d in dist:
                    gts[0]["is_metric"] = gts[0]["is_metric_scale"] & (
                        d.max(dim=-1).values < self.max_metric_scale
                    )
            not_metric_mask = ~gts[0]["is_metric"]
        else:
            not_metric_mask = torch.ones_like(gts[0]["is_metric"])

        # normalize 3d points
        # compute the scale using only the self view point maps
        if self.norm_mode and not self.gt_scale:
            norm_factor_gt = self.get_norm_factor_point_cloud(
                gt_pts_self,
                gt_pts_cross,
                valids,
                conf_self,
                conf_cross,
                norm_self_only=norm_self_only,
            )
        else:
            norm_factor_gt = torch.ones_like(
                preds[0]["pts3d_in_other_view"][:, :1, :1, :1]
            )

        norm_factor_pr = norm_factor_gt.clone()
        if self.norm_mode and not_metric_mask.sum() > 0 and not self.gt_scale:
            norm_factor_pr_not_metric = self.get_norm_factor_point_cloud(
                [pr_pt_self[not_metric_mask] for pr_pt_self in pr_pts_self],
                [pr_pt_cross[not_metric_mask] for pr_pt_cross in pr_pts_cross],
                [valid[not_metric_mask] for valid in valids],
                [conf[not_metric_mask] for conf in conf_self],
                [conf[not_metric_mask] for conf in conf_cross],
                norm_self_only=norm_self_only,
            )
            norm_factor_pr[not_metric_mask] = norm_factor_pr_not_metric

        norm_factor_gt = norm_factor_gt.clip(eps)
        norm_factor_pr = norm_factor_pr.clip(eps)

        gt_pts_self = [pts / norm_factor_gt for pts in gt_pts_self]
        gt_pts_cross = [pts / norm_factor_gt for pts in gt_pts_cross]
        pr_pts_self = [pts / norm_factor_pr for pts in pr_pts_self]
        pr_pts_cross = [pts / norm_factor_pr for pts in pr_pts_cross]

        # [(Bx3, BX4), (BX3, BX4), ...], 3 for translation, 4 for quaternion
        gt_poses = [
            camera_to_pose_encoding(in_camera1 @ gt["camera_pose"]).clone()
            for gt in gts
        ]
        pr_poses = [pred["camera_pose"].clone() for pred in preds]
        pose_norm_factor_gt = norm_factor_gt.clone().squeeze(2, 3)
        pose_norm_factor_pr = norm_factor_pr.clone().squeeze(2, 3)

        if norm_pose_separately:
            gt_trans = [gt[:, :3] for gt in gt_poses]
            pr_trans = [pr[:, :3] for pr in pr_poses]
            pose_norm_factor_gt, pose_norm_factor_pr = self.get_norm_factor_poses(
                gt_trans, pr_trans, not_metric_mask
            )
        elif any(camera_only):
            gt_trans = [gt[:, :3] for gt in gt_poses]
            pr_trans = [pr[:, :3] for pr in pr_poses]
            pose_only_norm_factor_gt, pose_only_norm_factor_pr = (
                self.get_norm_factor_poses(gt_trans, pr_trans, not_metric_mask)
            )
            pose_norm_factor_gt = torch.where(
                camera_only[:, None], pose_only_norm_factor_gt, pose_norm_factor_gt
            )
            pose_norm_factor_pr = torch.where(
                camera_only[:, None], pose_only_norm_factor_pr, pose_norm_factor_pr
            )

        gt_poses = [
            (gt[:, :3] / pose_norm_factor_gt.clip(eps), gt[:, 3:]) for gt in gt_poses
        ]
        pr_poses = [
            (pr[:, :3] / pose_norm_factor_pr.clip(eps), pr[:, 3:]) for pr in pr_poses
        ]
        # **========== 原始代码备份：batch size 1 时 squeeze 会产生 0-d pose_masks ==========**
        # pose_masks = (pose_norm_factor_gt.squeeze() > eps) & (
        #     pose_norm_factor_pr.squeeze() > eps
        # )
        # **========== 结束 ==========**
        pose_masks = (pose_norm_factor_gt.reshape(-1) > eps) & (
            pose_norm_factor_pr.reshape(-1) > eps
        )

        if any(camera_only):
            # this is equal to a loss for camera intrinsics
            gt_pts_self = [
                torch.where(
                    camera_only[:, None, None, None],
                    (gt / gt[..., -1:].clip(1e-6)).clip(-2, 2),
                    gt,
                )
                for gt in gt_pts_self
            ]
            pr_pts_self = [
                torch.where(
                    camera_only[:, None, None, None],
                    (pr / pr[..., -1:].clip(1e-6)).clip(-2, 2),
                    pr,
                )
                for pr in pr_pts_self
            ]
            # # do not add cross view loss when there is only camera supervision

        skys = [gt["sky_mask"] & ~valid for gt, valid in zip(gts, valids)]
        return (
            gt_pts_self,
            gt_pts_cross,
            pr_pts_self,
            pr_pts_cross,
            gt_poses,
            pr_poses,
            valids,
            skys,
            pose_masks,
            {},
        )

    def get_all_pts3d_with_scale_loss(
        self,
        gts,
        preds,
        dist_clip=None,
        norm_self_only=False,
        norm_pose_separately=False,
        eps=1e-3,
    ):
        # everything is normalized w.r.t. camera of view1
        in_camera1 = inv(gts[0]["camera_pose"])
        gt_pts_self = [geotrf(inv(gt["camera_pose"]), gt["pts3d"]) for gt in gts]
        gt_pts_cross = [geotrf(in_camera1, gt["pts3d"]) for gt in gts]
        valids = [gt["valid_mask"].clone() for gt in gts]
        camera_only = gts[0]["camera_only"]

        if dist_clip is not None:
            # points that are too far-away == invalid
            dis = [gt_pt.norm(dim=-1) for gt_pt in gt_pts_cross]
            valids = [valid & (dis <= dist_clip) for valid, dis in zip(valids, dis)]

        pr_pts_self = [pred["pts3d_in_self_view"] for pred in preds]
        pr_pts_cross = [pred["pts3d_in_other_view"] for pred in preds]
        conf_self = [torch.log(pred["conf_self"]).detach().clip(eps) for pred in preds]
        conf_cross = [torch.log(pred["conf"]).detach().clip(eps) for pred in preds]

        if not self.norm_all:
            if self.max_metric_scale:
                B = valids[0].shape[0]
                dist = [
                    torch.where(valid, torch.linalg.norm(gt_pt_cross, dim=-1), 0).view(
                        B, -1
                    )
                    for valid, gt_pt_cross in zip(valids, gt_pts_cross)
                ]
                for d in dist:
                    gts[0]["is_metric"] = gts[0]["is_metric_scale"] & (
                        d.max(dim=-1).values < self.max_metric_scale
                    )
            not_metric_mask = ~gts[0]["is_metric"]
        else:
            not_metric_mask = torch.ones_like(gts[0]["is_metric"])

        # normalize 3d points
        # compute the scale using only the self view point maps
        if self.norm_mode and not self.gt_scale:
            norm_factor_gt = self.get_norm_factor_point_cloud(
                gt_pts_self[:1],
                gt_pts_cross[:1],
                valids[:1],
                conf_self[:1],
                conf_cross[:1],
                norm_self_only=norm_self_only,
            )
        else:
            norm_factor_gt = torch.ones_like(
                preds[0]["pts3d_in_other_view"][:, :1, :1, :1]
            )

        if self.norm_mode:
            norm_factor_pr = self.get_norm_factor_point_cloud(
                pr_pts_self[:1],
                pr_pts_cross[:1],
                valids[:1],
                conf_self[:1],
                conf_cross[:1],
                norm_self_only=norm_self_only,
            )
        else:
            raise NotImplementedError
        # only add loss to metric scale norm factor
        if (~not_metric_mask).sum() > 0:
            pts_scale_loss = torch.abs(
                norm_factor_pr[~not_metric_mask] - norm_factor_gt[~not_metric_mask]
            ).mean()
        else:
            pts_scale_loss = 0.0

        norm_factor_gt = norm_factor_gt.clip(eps)
        norm_factor_pr = norm_factor_pr.clip(eps)

        gt_pts_self = [pts / norm_factor_gt for pts in gt_pts_self]
        gt_pts_cross = [pts / norm_factor_gt for pts in gt_pts_cross]
        pr_pts_self = [pts / norm_factor_pr for pts in pr_pts_self]
        pr_pts_cross = [pts / norm_factor_pr for pts in pr_pts_cross]

        # [(Bx3, BX4), (BX3, BX4), ...], 3 for translation, 4 for quaternion
        gt_poses = [
            camera_to_pose_encoding(in_camera1 @ gt["camera_pose"]).clone()
            for gt in gts
        ]
        pr_poses = [pred["camera_pose"].clone() for pred in preds]
        pose_norm_factor_gt = norm_factor_gt.clone().squeeze(2, 3)
        pose_norm_factor_pr = norm_factor_pr.clone().squeeze(2, 3)

        if norm_pose_separately:
            gt_trans = [gt[:, :3] for gt in gt_poses][:1]
            pr_trans = [pr[:, :3] for pr in pr_poses][:1]
            pose_norm_factor_gt, pose_norm_factor_pr = self.get_norm_factor_poses(
                gt_trans, pr_trans, torch.ones_like(not_metric_mask)
            )
        elif any(camera_only):
            gt_trans = [gt[:, :3] for gt in gt_poses][:1]
            pr_trans = [pr[:, :3] for pr in pr_poses][:1]
            pose_only_norm_factor_gt, pose_only_norm_factor_pr = (
                self.get_norm_factor_poses(
                    gt_trans, pr_trans, torch.ones_like(not_metric_mask)
                )
            )
            pose_norm_factor_gt = torch.where(
                camera_only[:, None], pose_only_norm_factor_gt, pose_norm_factor_gt
            )
            pose_norm_factor_pr = torch.where(
                camera_only[:, None], pose_only_norm_factor_pr, pose_norm_factor_pr
            )
        # only add loss to metric scale norm factor
        if (~not_metric_mask).sum() > 0:
            pose_scale_loss = torch.abs(
                pose_norm_factor_pr[~not_metric_mask]
                - pose_norm_factor_gt[~not_metric_mask]
            ).mean()
        else:
            pose_scale_loss = 0.0
        gt_poses = [
            (gt[:, :3] / pose_norm_factor_gt.clip(eps), gt[:, 3:]) for gt in gt_poses
        ]
        pr_poses = [
            (pr[:, :3] / pose_norm_factor_pr.clip(eps), pr[:, 3:]) for pr in pr_poses
        ]

        # **========== 原始代码备份：batch size 1 时 squeeze 会产生 0-d pose_masks ==========**
        # pose_masks = (pose_norm_factor_gt.squeeze() > eps) & (
        #     pose_norm_factor_pr.squeeze() > eps
        # )
        # **========== 结束 ==========**
        pose_masks = (pose_norm_factor_gt.reshape(-1) > eps) & (
            pose_norm_factor_pr.reshape(-1) > eps
        )

        if any(camera_only):
            # this is equal to a loss for camera intrinsics
            gt_pts_self = [
                torch.where(
                    camera_only[:, None, None, None],
                    (gt / gt[..., -1:].clip(1e-6)).clip(-2, 2),
                    gt,
                )
                for gt in gt_pts_self
            ]
            pr_pts_self = [
                torch.where(
                    camera_only[:, None, None, None],
                    (pr / pr[..., -1:].clip(1e-6)).clip(-2, 2),
                    pr,
                )
                for pr in pr_pts_self
            ]
            # # do not add cross view loss when there is only camera supervision

        skys = [gt["sky_mask"] & ~valid for gt, valid in zip(gts, valids)]
        return (
            gt_pts_self,
            gt_pts_cross,
            pr_pts_self,
            pr_pts_cross,
            gt_poses,
            pr_poses,
            valids,
            skys,
            pose_masks,
            {"scale_loss": pose_scale_loss + pts_scale_loss},
        )

    def compute_relative_pose_loss(
        self, gt_trans, gt_quats, pr_trans, pr_quats, masks=None
    ):
        if masks is None:
            masks = torch.ones(len(gt_trans), dtype=torch.bool, device=gt_trans.device)
        gt_trans_matrix1 = gt_trans[:, :, None, :].repeat(1, 1, gt_trans.shape[1], 1)[
            masks
        ]
        gt_trans_matrix2 = gt_trans[:, None, :, :].repeat(1, gt_trans.shape[1], 1, 1)[
            masks
        ]
        gt_quats_matrix1 = gt_quats[:, :, None, :].repeat(1, 1, gt_quats.shape[1], 1)[
            masks
        ]
        gt_quats_matrix2 = gt_quats[:, None, :, :].repeat(1, gt_quats.shape[1], 1, 1)[
            masks
        ]
        pr_trans_matrix1 = pr_trans[:, :, None, :].repeat(1, 1, pr_trans.shape[1], 1)[
            masks
        ]
        pr_trans_matrix2 = pr_trans[:, None, :, :].repeat(1, pr_trans.shape[1], 1, 1)[
            masks
        ]
        pr_quats_matrix1 = pr_quats[:, :, None, :].repeat(1, 1, pr_quats.shape[1], 1)[
            masks
        ]
        pr_quats_matrix2 = pr_quats[:, None, :, :].repeat(1, pr_quats.shape[1], 1, 1)[
            masks
        ]

        gt_rel_trans, gt_rel_quats = relative_pose_absT_quatR(
            gt_trans_matrix1, gt_quats_matrix1, gt_trans_matrix2, gt_quats_matrix2
        )
        pr_rel_trans, pr_rel_quats = relative_pose_absT_quatR(
            pr_trans_matrix1, pr_quats_matrix1, pr_trans_matrix2, pr_quats_matrix2
        )
        rel_trans_err = torch.norm(gt_rel_trans - pr_rel_trans, dim=-1)
        rel_quats_err = torch.norm(gt_rel_quats - pr_rel_quats, dim=-1)
        return rel_trans_err.mean() + rel_quats_err.mean()

    def compute_pose_loss(self, gt_poses, pred_poses, masks=None):
        """
        gt_pose: list of (Bx3, Bx4)
        pred_pose: list of (Bx3, Bx4)
        masks: None, or B
        """
        gt_trans = torch.stack([gt[0] for gt in gt_poses], dim=1)  # BxNx3
        gt_quats = torch.stack([gt[1] for gt in gt_poses], dim=1)  # BXNX4
        pred_trans = torch.stack([pr[0] for pr in pred_poses], dim=1)  # BxNx3
        pred_quats = torch.stack([pr[1] for pr in pred_poses], dim=1)  # BxNx4
        if masks == None:
            pose_loss = (
                torch.norm(pred_trans - gt_trans, dim=-1).mean()
                + torch.norm(pred_quats - gt_quats, dim=-1).mean()
            )
        else:
            if not any(masks):
                return torch.tensor(0.0)
            pose_loss = (
                torch.norm(pred_trans - gt_trans, dim=-1)[masks].mean()
                + torch.norm(pred_quats - gt_quats, dim=-1)[masks].mean()
            )

        return pose_loss

    def compute_loss(self, gts, preds, **kw):
        (
            gt_pts_self,
            gt_pts_cross,
            pred_pts_self,
            pred_pts_cross,
            gt_poses,
            pr_poses,
            masks,
            skys,
            pose_masks,
            monitoring,
        ) = self.get_all_pts3d(gts, preds, **kw)

        if self.sky_loss_value > 0:
            assert (
                self.criterion.reduction == "none"
            ), "sky_loss_value should be 0 if no conf loss"
            masks = [mask | sky for mask, sky in zip(masks, skys)]

        # self view loss and details
        if "Quantile" in self.criterion.__class__.__name__:
            # masks are overwritten taking into account self view losses
            ls_self, masks = self.criterion(
                pred_pts_self, gt_pts_self, masks, gts[0]["quantile"]
            )
        else:
            ls_self = [
                self.criterion(pred_pt[mask], gt_pt[mask])
                for pred_pt, gt_pt, mask in zip(pred_pts_self, gt_pts_self, masks)
            ]

        if self.sky_loss_value > 0:
            assert (
                self.criterion.reduction == "none"
            ), "sky_loss_value should be 0 if no conf loss"
            for i, l in enumerate(ls_self):
                ls_self[i] = torch.where(skys[i][masks[i]], self.sky_loss_value, l)

        self_name = type(self).__name__

        details = {}
        for i in range(len(ls_self)):
            details[self_name + f"_self_pts3d/{i+1}"] = float(ls_self[i].mean())
            details[f"gt_img{i+1}"] = gts[i]["img"].permute(0, 2, 3, 1).detach()
            details[f"self_conf_{i+1}"] = preds[i]["conf_self"].detach()
            details[f"valid_mask_{i+1}"] = masks[i].detach()

            if "img_mask" in gts[i] and "ray_mask" in gts[i]:
                details[f"img_mask_{i+1}"] = gts[i]["img_mask"].detach()
                details[f"ray_mask_{i+1}"] = gts[i]["ray_mask"].detach()

            if "desc" in preds[i]:
                details[f"desc_{i+1}"] = preds[i]["desc"].detach()

        # cross view loss and details
        camera_only = gts[0]["camera_only"]
        pred_pts_cross = [pred_pts[~camera_only] for pred_pts in pred_pts_cross]
        gt_pts_cross = [gt_pts[~camera_only] for gt_pts in gt_pts_cross]
        masks_cross = [mask[~camera_only] for mask in masks]
        skys_cross = [sky[~camera_only] for sky in skys]

        if "Quantile" in self.criterion.__class__.__name__:
            # quantile masks have already been determined by self view losses, here pass in None as quantile
            ls_cross, _ = self.criterion(
                pred_pts_cross, gt_pts_cross, masks_cross, None
            )
        else:
            ls_cross = [
                self.criterion(pred_pt[mask], gt_pt[mask])
                for pred_pt, gt_pt, mask in zip(
                    pred_pts_cross, gt_pts_cross, masks_cross
                )
            ]

        if self.sky_loss_value > 0:
            assert (
                self.criterion.reduction == "none"
            ), "sky_loss_value should be 0 if no conf loss"
            for i, l in enumerate(ls_cross):
                ls_cross[i] = torch.where(
                    skys_cross[i][masks_cross[i]], self.sky_loss_value, l
                )

        for i in range(len(ls_cross)):
            details[self_name + f"_pts3d/{i+1}"] = float(
                ls_cross[i].mean() if ls_cross[i].numel() > 0 else 0
            )
            details[f"conf_{i+1}"] = preds[i]["conf"].detach()

        ls = ls_self + ls_cross
        masks = masks + masks_cross
        details["is_self"] = [True] * len(ls_self) + [False] * len(ls_cross)
        details["img_ids"] = (
            np.arange(len(ls_self)).tolist() + np.arange(len(ls_cross)).tolist()
        )
        details["pose_loss"] = self.compute_pose_loss(gt_poses, pr_poses, pose_masks)

        return Sum(*list(zip(ls, masks))), (details | monitoring)


class Regr3DPoseBatchList(Regr3DPose):
    """Ensure that all 3D points are correct.
    Asymmetric loss: view1 is supposed to be the anchor.

    P1 = RT1 @ D1
    P2 = RT2 @ D2
    loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
    loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
          = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """

    def __init__(
        self,
        criterion,
        norm_mode="?avg_dis",
        gt_scale=False,
        sky_loss_value=2,
        max_metric_scale=False,
        shot_boundary_abs_weight=0.0,
        shot_jump_rel_weight=0.0,
        shot_anchor_weight=0.0,
    ):
        super().__init__(
            criterion, norm_mode, gt_scale, sky_loss_value, max_metric_scale
        )
        self.depth_only_criterion = DepthScaleShiftInvLoss()
        self.single_view_criterion = ScaleInvLoss()
        self.shot_boundary_abs_weight = float(shot_boundary_abs_weight)
        self.shot_jump_rel_weight = float(shot_jump_rel_weight)
        self.shot_anchor_weight = float(shot_anchor_weight)

    def _pose_abs_components(self, gt_poses, pr_poses, view_idx, mask):
        gt_trans = gt_poses[view_idx][0][mask]
        gt_quat = gt_poses[view_idx][1][mask]
        pr_trans = pr_poses[view_idx][0][mask]
        pr_quat = pr_poses[view_idx][1][mask]
        t_err = torch.norm(pr_trans - gt_trans, dim=-1)
        q_err = torch.norm(pr_quat - gt_quat, dim=-1)
        return t_err.mean(), q_err.mean()

    def _pose_rel_components(self, gt_poses, pr_poses, src_idx, dst_idx, mask):
        gt_rel_trans, gt_rel_quat = relative_pose_absT_quatR(
            gt_poses[src_idx][0][mask],
            gt_poses[src_idx][1][mask],
            gt_poses[dst_idx][0][mask],
            gt_poses[dst_idx][1][mask],
        )
        pr_rel_trans, pr_rel_quat = relative_pose_absT_quatR(
            pr_poses[src_idx][0][mask],
            pr_poses[src_idx][1][mask],
            pr_poses[dst_idx][0][mask],
            pr_poses[dst_idx][1][mask],
        )
        t_err = torch.norm(pr_rel_trans - gt_rel_trans, dim=-1)
        q_err = torch.norm(pr_rel_quat - gt_rel_quat, dim=-1)
        return t_err.mean(), q_err.mean()

    def _add_v5_shot_pose_losses(self, details, gt_poses, pr_poses, pose_masks, is_video):
        if len(gt_poses) < 4 or len(pr_poses) < 4:
            return details
        is_aabb_mask = (~is_video) & pose_masks.bool()
        if not is_aabb_mask.any():
            return details

        zero = gt_poses[0][0].new_tensor(0.0)
        extra_pose_loss = zero

        a2_t_err, a2_q_err = self._pose_abs_components(gt_poses, pr_poses, 1, is_aabb_mask)
        b1_t_err, b1_q_err = self._pose_abs_components(gt_poses, pr_poses, 2, is_aabb_mask)
        boundary_abs_loss = a2_t_err + a2_q_err + b1_t_err + b1_q_err
        view2_pose_loss = b1_t_err + b1_q_err
        details.update({
            "pose_loss_view2_AABB": float(view2_pose_loss.detach()),
            "shot_boundary_abs_loss": float(boundary_abs_loss.detach()),
            "shot_boundary_abs_loss_weighted": float((boundary_abs_loss * self.shot_boundary_abs_weight).detach()),
            "shot_boundary_abs_t_err": float(((a2_t_err + b1_t_err) * 0.5).detach()),
            "shot_boundary_abs_q_err": float(((a2_q_err + b1_q_err) * 0.5).detach()),
        })
        extra_pose_loss = extra_pose_loss + self.shot_boundary_abs_weight * boundary_abs_loss

        jump_t_err, jump_q_err = self._pose_rel_components(gt_poses, pr_poses, 1, 2, is_aabb_mask)
        jump_rel_loss = jump_t_err + jump_q_err
        details.update({
            "shot_jump_rel_loss": float(jump_rel_loss.detach()),
            "shot_jump_rel_loss_weighted": float((jump_rel_loss * self.shot_jump_rel_weight).detach()),
            "shot_jump_t_err": float(jump_t_err.detach()),
            "shot_jump_q_err": float(jump_q_err.detach()),
        })
        extra_pose_loss = extra_pose_loss + self.shot_jump_rel_weight * jump_rel_loss

        anchor2_t_err, anchor2_q_err = self._pose_rel_components(gt_poses, pr_poses, 1, 2, is_aabb_mask)
        anchor3_t_err, anchor3_q_err = self._pose_rel_components(gt_poses, pr_poses, 1, 3, is_aabb_mask)
        anchor_view2_loss = anchor2_t_err + anchor2_q_err
        anchor_view3_loss = anchor3_t_err + anchor3_q_err
        anchor_loss = 0.5 * (anchor_view2_loss + anchor_view3_loss)
        details.update({
            "shot_anchor_loss": float(anchor_loss.detach()),
            "shot_anchor_loss_weighted": float((anchor_loss * self.shot_anchor_weight).detach()),
            "shot_anchor_t_err": float((0.5 * (anchor2_t_err + anchor3_t_err)).detach()),
            "shot_anchor_q_err": float((0.5 * (anchor2_q_err + anchor3_q_err)).detach()),
            "shot_anchor_view2_t_err": float(anchor2_t_err.detach()),
            "shot_anchor_view3_t_err": float(anchor3_t_err.detach()),
        })
        extra_pose_loss = extra_pose_loss + self.shot_anchor_weight * anchor_loss

        details["pose_loss"] = details["pose_loss"] + extra_pose_loss
        return details

    def reorg(self, ls_b, masks_b):
        ids_split = [mask.sum(dim=(1, 2)) for mask in masks_b]
        ls = [[] for _ in range(len(masks_b[0]))]
        for i in range(len(ls_b)):
            ls_splitted_i = torch.split(ls_b[i], ids_split[i].tolist())
            for j in range(len(masks_b[0])):
                ls[j].append(ls_splitted_i[j])
        ls = [torch.cat(l) for l in ls]
        return ls

    def compute_loss(self, gts, preds, **kw):
        (
            gt_pts_self,
            gt_pts_cross,
            pred_pts_self,
            pred_pts_cross,
            gt_poses,
            pr_poses,
            masks,
            skys,
            pose_masks,
            monitoring,
        ) = self.get_all_pts3d(gts, preds, **kw)

        if self.sky_loss_value > 0:
            assert (
                self.criterion.reduction == "none"
            ), "sky_loss_value should be 0 if no conf loss"
            masks = [mask | sky for mask, sky in zip(masks, skys)]

        camera_only = gts[0]["camera_only"]
        depth_only = gts[0]["depth_only"]
        single_view = gts[0]["single_view"]
        is_metric = gts[0]["is_metric"]

        # self view loss and details
        if "Quantile" in self.criterion.__class__.__name__:
            raise NotImplementedError
        else:
            # list [(B, h, w, 3)] x num_views -> list [num_views, h, w, 3] x B
            gt_pts_self_b = torch.unbind(torch.stack(gt_pts_self, dim=1), dim=0)
            pred_pts_self_b = torch.unbind(torch.stack(pred_pts_self, dim=1), dim=0)
            masks_b = torch.unbind(torch.stack(masks, dim=1), dim=0)
            ls_self_b = []
            for i in range(len(gt_pts_self_b)):
                if depth_only[
                    i
                ]:  # if only have relative depth, no intrinsics or anything
                    ls_self_b.append(
                        self.depth_only_criterion(
                            pred_pts_self_b[i][..., -1],
                            gt_pts_self_b[i][..., -1],
                            masks_b[i],
                        )
                    )
                elif (
                    single_view[i] and not is_metric[i]
                ):  # if single view, with intrinsics and not metric
                    ls_self_b.append(
                        self.single_view_criterion(
                            pred_pts_self_b[i], gt_pts_self_b[i], masks_b[i]
                        )
                    )
                else:  # if multiple view, or metric single view
                    ls_self_b.append(
                        self.criterion(
                            pred_pts_self_b[i][masks_b[i]], gt_pts_self_b[i][masks_b[i]]
                        )
                    )
            ls_self = self.reorg(ls_self_b, masks_b)

        if self.sky_loss_value > 0:
            assert (
                self.criterion.reduction == "none"
            ), "sky_loss_value should be 0 if no conf loss"
            for i, l in enumerate(ls_self):
                ls_self[i] = torch.where(skys[i][masks[i]], self.sky_loss_value, l)

        self_name = type(self).__name__

        details = {}
        for i in range(len(ls_self)):
            details[self_name + f"_self_pts3d/{i+1}"] = float(ls_self[i].mean())
            details[f"self_conf_{i+1}"] = preds[i]["conf_self"].detach()
            details[f"gt_img{i+1}"] = gts[i]["img"].permute(0, 2, 3, 1).detach()
            details[f"valid_mask_{i+1}"] = masks[i].detach()

            if "img_mask" in gts[i] and "ray_mask" in gts[i]:
                details[f"img_mask_{i+1}"] = gts[i]["img_mask"].detach()
                details[f"ray_mask_{i+1}"] = gts[i]["ray_mask"].detach()

            if "desc" in preds[i]:
                details[f"desc_{i+1}"] = preds[i]["desc"].detach()

        if "Quantile" in self.criterion.__class__.__name__:
            # quantile masks have already been determined by self view losses, here pass in None as quantile
            raise NotImplementedError
        else:
            gt_pts_cross_b = torch.unbind(
                torch.stack(gt_pts_cross, dim=1)[~camera_only], dim=0
            )
            pred_pts_cross_b = torch.unbind(
                torch.stack(pred_pts_cross, dim=1)[~camera_only], dim=0
            )
            masks_cross_b = torch.unbind(torch.stack(masks, dim=1)[~camera_only], dim=0)
            ls_cross_b = []
            for i in range(len(gt_pts_cross_b)):
                if depth_only[~camera_only][i]:
                    ls_cross_b.append(
                        self.depth_only_criterion(
                            pred_pts_cross_b[i][..., -1],
                            gt_pts_cross_b[i][..., -1],
                            masks_cross_b[i],
                        )
                    )
                elif single_view[~camera_only][i] and not is_metric[~camera_only][i]:
                    ls_cross_b.append(
                        self.single_view_criterion(
                            pred_pts_cross_b[i], gt_pts_cross_b[i], masks_cross_b[i]
                        )
                    )
                else:
                    ls_cross_b.append(
                        self.criterion(
                            pred_pts_cross_b[i][masks_cross_b[i]],
                            gt_pts_cross_b[i][masks_cross_b[i]],
                        )
                    )
            ls_cross = self.reorg(ls_cross_b, masks_cross_b)

        if self.sky_loss_value > 0:
            assert (
                self.criterion.reduction == "none"
            ), "sky_loss_value should be 0 if no conf loss"
            masks_cross = [mask[~camera_only] for mask in masks]
            skys_cross = [sky[~camera_only] for sky in skys]
            for i, l in enumerate(ls_cross):
                ls_cross[i] = torch.where(
                    skys_cross[i][masks_cross[i]], self.sky_loss_value, l
                )

        for i in range(len(ls_cross)):
            details[self_name + f"_pts3d/{i+1}"] = float(
                ls_cross[i].mean() if ls_cross[i].numel() > 0 else 0
            )
            details[f"conf_{i+1}"] = preds[i]["conf"].detach()

        ls = ls_self + ls_cross
        masks = masks + masks_cross
        details["is_self"] = [True] * len(ls_self) + [False] * len(ls_cross)
        details["img_ids"] = (
            np.arange(len(ls_self)).tolist() + np.arange(len(ls_cross)).tolist()
        )
        pose_masks = pose_masks * gts[i]["img_mask"]
        details["pose_loss"] = self.compute_pose_loss(gt_poses, pr_poses, pose_masks)

        # **========== V4 原始代码备份：AABB 只额外监督 view2/B1 absolute pose ==========**
        # # ===== AABB view2 pose loss =====
        # # AABB: view0,view1 from camA, view2,view3 from camB
        # # 对 AABB 数据的 view2（第一个 B 帧）单独计算 pose L2 loss
        # # gts[0]["is_video"] = True for Video, False for AABB
        # is_video = gts[0]["is_video"]
        # if not is_video.all():
        #     is_aabb_mask = ~is_video
        #     gt_trans_view2 = gt_poses[2][0][is_aabb_mask]
        #     gt_quat_view2 = gt_poses[2][1][is_aabb_mask]
        #     pr_trans_view2 = pr_poses[2][0][is_aabb_mask]
        #     pr_quat_view2 = pr_poses[2][1][is_aabb_mask]
        #     view2_pose_loss = (
        #         torch.norm(pr_trans_view2 - gt_trans_view2, dim=-1).mean()
        #         + torch.norm(pr_quat_view2 - gt_quat_view2, dim=-1).mean()
        #     )
        #     details["pose_loss_view2_AABB"] = float(view2_pose_loss)
        #     details["pose_loss"] = details["pose_loss"] + view2_pose_loss
        # **========== 结束 ==========**
        # **========== V4 原始代码备份：实际运行的 view2/B1 absolute pose 加权 ==========**
        # # ===== AABB view2 pose loss =====
        # # AABB: view0,view1 from camA, view2,view3 from camB
        # # 对 AABB 数据的 view2（第一个 B 帧）单独计算 pose L2 loss
        # # gts[0]["is_video"] = True for Video, False for AABB
        # is_video = gts[0]["is_video"]
        # if not is_video.all():
        #     is_aabb_mask = ~is_video
        #     gt_trans_view2 = gt_poses[2][0][is_aabb_mask]
        #     gt_quat_view2 = gt_poses[2][1][is_aabb_mask]
        #     pr_trans_view2 = pr_poses[2][0][is_aabb_mask]
        #     pr_quat_view2 = pr_poses[2][1][is_aabb_mask]
        #     view2_pose_loss = (
        #         torch.norm(pr_trans_view2 - gt_trans_view2, dim=-1).mean()
        #         + torch.norm(pr_quat_view2 - gt_quat_view2, dim=-1).mean()
        #     )
        #     details["pose_loss_view2_AABB"] = float(view2_pose_loss)
        #     details["pose_loss"] = details["pose_loss"] + view2_pose_loss
        # **========== 结束 ==========**
        is_video = gts[0]["is_video"]
        details = self._add_v5_shot_pose_losses(
            details, gt_poses, pr_poses, pose_masks, is_video
        )

        return Sum(*list(zip(ls, masks))), (details | monitoring)


class V7PosePseudoLoss(MultiLoss):
    """Supervise V7 implicit pose adapter from teacher pseudo labels.

    Expected per-view target keys are intentionally explicit to avoid mixing this
    with ordinary Human3R GT fields:
    v7_delta_t, v7_delta_rotvec, v7_alpha, v7_r_human, v7_r_scene, v7_valid.
    Missing views are skipped so the same criterion can run on mixed batches.
    """

    def __init__(
        self,
        delta_t_weight=1.0,
        delta_r_weight=4.0,
        alpha_weight=0.2,
        reliability_weight=0.1,
        prior_weight=0.0,
        loss_type="smooth_l1",
    ):
        super().__init__()
        self.delta_t_weight = float(delta_t_weight)
        self.delta_r_weight = float(delta_r_weight)
        self.alpha_weight = float(alpha_weight)
        self.reliability_weight = float(reliability_weight)
        self.prior_weight = float(prior_weight)
        self.loss_type = str(loss_type)

    def get_name(self):
        return "V7PosePseudoLoss"

    def _first_tensor(self, preds):
        for pred in preds:
            for value in pred.values():
                if torch.is_tensor(value):
                    return value
        return None

    def _get_target(self, gt, names, ref, trailing_dim=None):
        for name in names:
            if name not in gt:
                continue
            value = gt[name]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value, device=ref.device)
            value = value.to(device=ref.device, dtype=ref.dtype)
            if trailing_dim is not None:
                if value.ndim == 1 and value.shape[0] == trailing_dim:
                    value = value.unsqueeze(0)
                value = value.reshape(ref.shape[0], trailing_dim)
            else:
                value = value.reshape(-1)
                if value.numel() == 1 and ref.shape[0] > 1:
                    value = value.expand(ref.shape[0])
            return value
        return None

    def _component_loss(self, pred, target):
        if self.loss_type == "l1":
            return (pred - target).abs().mean(dim=-1)
        if self.loss_type == "mse":
            return (pred - target).pow(2).mean(dim=-1)
        return F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)

    def _masked_mean(self, value, mask):
        denom = mask.sum().clamp_min(1.0)
        return (value * mask).sum() / denom

    def compute_loss(self, gts, preds, **kw):
        zero_ref = self._first_tensor(preds)
        if zero_ref is None:
            return torch.zeros(())
        zero = zero_ref.float().new_zeros(())

        total = zero
        details = {}
        count = 0.0
        t_errs, r_errs, alpha_errs = [], [], []
        r_human_errs, r_scene_errs = [], []
        prior_terms = []

        for gt, pred in zip(gts, preds):
            pred_t = pred.get("v7_pose_delta_t", None)
            pred_r = pred.get("v7_pose_delta_rotvec", None)
            if pred_t is None or pred_r is None:
                continue
            pred_t = pred_t.float()
            pred_r = pred_r.float()
            target_t = self._get_target(
                gt,
                ["v7_delta_t", "v7_pose_delta_t_target", "v7_pose_target_delta_t", "pseudo_delta_t"],
                pred_t,
                trailing_dim=3,
            )
            target_r = self._get_target(
                gt,
                ["v7_delta_rotvec", "v7_pose_delta_rotvec_target", "v7_pose_target_delta_rotvec", "pseudo_delta_rotvec"],
                pred_r,
                trailing_dim=3,
            )
            if target_t is None or target_r is None:
                continue
            valid = self._get_target(
                gt,
                ["v7_valid", "v7_pose_label_mask", "v7_label_mask", "pseudo_valid"],
                pred_t,
            )
            if valid is None:
                valid = torch.ones(pred_t.shape[0], device=pred_t.device, dtype=pred_t.dtype)
            valid = valid.float().clamp(0.0, 1.0)
            if valid.sum() <= 0:
                continue

            t_loss = self._component_loss(pred_t, target_t)
            r_loss = self._component_loss(pred_r, target_r)
            total = total + self.delta_t_weight * self._masked_mean(t_loss, valid)
            total = total + self.delta_r_weight * self._masked_mean(r_loss, valid)
            count += float(valid.detach().sum())
            t_errs.append((torch.linalg.norm(pred_t.detach() - target_t.detach(), dim=-1) * valid).sum() / valid.sum().clamp_min(1.0))
            r_errs.append((torch.linalg.norm(pred_r.detach() - target_r.detach(), dim=-1) * valid).sum() / valid.sum().clamp_min(1.0))

            pred_alpha = pred.get("v7_pose_alpha", None)
            if pred_alpha is not None and self.alpha_weight > 0:
                pred_alpha = pred_alpha.float().reshape(-1).clamp(1e-4, 1.0 - 1e-4)
                target_alpha = self._get_target(
                    gt,
                    ["v7_alpha", "v7_pose_alpha_target", "pseudo_alpha"],
                    pred_alpha,
                )
                if target_alpha is None:
                    target_alpha = (target_t.norm(dim=-1) + target_r.norm(dim=-1) > 1e-5).to(pred_alpha.dtype)
                target_alpha = target_alpha.float().clamp(0.0, 1.0)
                alpha_loss = F.binary_cross_entropy(pred_alpha, target_alpha, reduction="none")
                total = total + self.alpha_weight * self._masked_mean(alpha_loss, valid)
                alpha_errs.append(((pred_alpha.detach() - target_alpha.detach()).abs() * valid).sum() / valid.sum().clamp_min(1.0))

            for pred_key, target_names, store in [
                ("v7_pose_r_human", ["v7_r_human", "v7_pose_r_human_target", "pseudo_r_human"], r_human_errs),
                ("v7_pose_r_scene", ["v7_r_scene", "v7_pose_r_scene_target", "pseudo_r_scene"], r_scene_errs),
            ]:
                pred_rel = pred.get(pred_key, None)
                if pred_rel is None or self.reliability_weight <= 0:
                    continue
                pred_rel = pred_rel.float().reshape(-1)
                target_rel = self._get_target(gt, target_names, pred_rel)
                if target_rel is None:
                    continue
                rel_loss = (pred_rel - target_rel.float().clamp(0.0, 1.0)).pow(2)
                total = total + self.reliability_weight * self._masked_mean(rel_loss, valid)
                store.append(((pred_rel.detach() - target_rel.detach()).abs() * valid).sum() / valid.sum().clamp_min(1.0))

            if self.prior_weight > 0:
                prior = pred_t.pow(2).mean(dim=-1) + pred_r.pow(2).mean(dim=-1)
                prior_terms.append(self._masked_mean(prior, valid))

        if prior_terms:
            total = total + self.prior_weight * torch.stack(prior_terms).mean()

        details["v7_pose_label_count"] = count
        if t_errs:
            details["v7_pose_delta_t_err"] = float(torch.stack(t_errs).mean())
        if r_errs:
            details["v7_pose_delta_r_err_deg"] = float(torch.rad2deg(torch.stack(r_errs).mean()))
        if alpha_errs:
            details["v7_pose_alpha_err"] = float(torch.stack(alpha_errs).mean())
        if r_human_errs:
            details["v7_pose_r_human_err"] = float(torch.stack(r_human_errs).mean())
        if r_scene_errs:
            details["v7_pose_r_scene_err"] = float(torch.stack(r_scene_errs).mean())
        details["v7_pose_pseudo_loss"] = float(total.detach())
        return total, details


class V81PosePromptLoss(MultiLoss):
    """Supervise the V8.1 decoder-in pose prompt from GT camera poses.

    The model branch predicts a corrected pose by replacing the decoder pose
    token before the original pose head. This criterion therefore supervises
    pred["camera_pose"] directly, and uses V8 auxiliary outputs only as light
    regularizers/debug metrics.
    """

    def __init__(
        self,
        translation_weight=1.0,
        rotation_weight=1.0,
        latent_weight=0.01,
        gate_weight=0.0,
        loss_type="smooth_l1",
        pose_key="camera_pose",
    ):
        super().__init__()
        self.translation_weight = float(translation_weight)
        self.rotation_weight = float(rotation_weight)
        self.latent_weight = float(latent_weight)
        self.gate_weight = float(gate_weight)
        self.loss_type = str(loss_type)
        self.pose_key = str(pose_key)

    def get_name(self):
        return "V81PosePromptLoss"

    def _first_tensor(self, gts, preds):
        for seq in (preds, gts):
            for item in seq:
                for value in item.values():
                    if torch.is_tensor(value):
                        return value
        return None

    def _component_loss(self, pred, target):
        if self.loss_type == "l1":
            return (pred - target).abs().mean(dim=-1)
        if self.loss_type == "mse":
            return (pred - target).pow(2).mean(dim=-1)
        return F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)

    def _quat_angle(self, pred_quat, gt_quat, eps=1e-7):
        pred_quat = F.normalize(pred_quat, dim=-1, eps=eps)
        gt_quat = F.normalize(gt_quat, dim=-1, eps=eps)
        dot = (pred_quat * gt_quat).sum(dim=-1).abs().clamp(max=1.0 - eps)
        return 2.0 * torch.acos(dot)

    def _gt_pose_encodings(self, gts):
        pose_key = self.pose_key
        if pose_key not in gts[0]:
            raise KeyError(f"V81PosePromptLoss pose_key={pose_key!r} not found in GT view")
        in_camera0 = inv(gts[0][pose_key])
        return [
            camera_to_pose_encoding(in_camera0 @ gt[pose_key]).clone()
            for gt in gts
        ]

    def _pose_errors(self, pred_pose, gt_pose):
        pred_t, pred_q = pred_pose[:, :3], pred_pose[:, 3:]
        gt_t, gt_q = gt_pose[:, :3], gt_pose[:, 3:]
        trans_err = torch.linalg.norm(pred_t - gt_t, dim=-1)
        rot_err = self._quat_angle(pred_q, gt_q)
        return trans_err, rot_err

    def _pose_loss(self, pred_pose, gt_pose):
        pred_t, pred_q = pred_pose[:, :3], pred_pose[:, 3:]
        gt_t, gt_q = gt_pose[:, :3], gt_pose[:, 3:]
        t_loss = self._component_loss(pred_t, gt_t)
        r_loss = self._quat_angle(pred_q, gt_q)
        return self.translation_weight * t_loss + self.rotation_weight * r_loss

    def compute_loss(self, gts, preds, **kw):
        ref = self._first_tensor(gts, preds)
        if ref is None:
            return torch.zeros(()), {}
        zero = ref.float().new_zeros(())
        gt_poses = self._gt_pose_encodings(gts)

        total = zero
        details = {}
        pose_losses = []
        trans_errs = []
        rot_errs = []
        raw_trans_errs = []
        raw_rot_errs = []
        gate_values = []
        delta_norms = []
        latent_terms = []
        gate_terms = []

        for view_idx, (gt_pose, gt, pred) in enumerate(zip(gt_poses, gts, preds)):
            pred_pose = pred.get("camera_pose", None)
            if pred_pose is None:
                continue
            pred_pose = pred_pose.float()
            gt_pose = gt_pose.to(device=pred_pose.device, dtype=pred_pose.dtype)
            pose_loss = self._pose_loss(pred_pose, gt_pose)
            pose_losses.append(pose_loss.mean())

            trans_err, rot_err = self._pose_errors(pred_pose.detach(), gt_pose.detach())
            trans_errs.append(trans_err.mean())
            rot_errs.append(rot_err.mean())
            details[f"v8_pose_prompt_trans_err/{view_idx}"] = float(trans_err.mean())
            details[f"v8_pose_prompt_rot_err_deg/{view_idx}"] = float(torch.rad2deg(rot_err.mean()))

            raw_pose = pred.get("v8_raw_camera_pose", None)
            if raw_pose is not None:
                raw_pose = raw_pose.to(device=pred_pose.device, dtype=pred_pose.dtype)
                raw_t_err, raw_r_err = self._pose_errors(raw_pose.detach(), gt_pose.detach())
                raw_trans_errs.append(raw_t_err.mean())
                raw_rot_errs.append(raw_r_err.mean())
                details[f"v8_raw_trans_err/{view_idx}"] = float(raw_t_err.mean())
                details[f"v8_raw_rot_err_deg/{view_idx}"] = float(torch.rad2deg(raw_r_err.mean()))

            delta_raw = pred.get("v8_pose_prompt_delta_raw", None)
            if delta_raw is not None:
                delta_term = delta_raw.float().pow(2).mean()
                latent_terms.append(delta_term)
                delta_norm = delta_raw.detach().float().norm(dim=-1).mean()
                delta_norms.append(delta_norm)
                details[f"v8_delta_norm/{view_idx}"] = float(delta_norm)

            gate = pred.get("v8_pose_prompt_gate", None)
            if gate is not None:
                gate = gate.float()
                gate_values.append(gate.detach().mean())
                details[f"v8_gate/{view_idx}"] = float(gate.detach().mean())
                if self.gate_weight > 0 and "shot_label" in gt:
                    target = gt["shot_label"]
                    if not torch.is_tensor(target):
                        target = torch.as_tensor(target, device=gate.device)
                    target = target.to(device=gate.device, dtype=gate.dtype).reshape(gate.shape[0], 1, 1)
                    gate_terms.append(F.binary_cross_entropy(gate.clamp(1e-4, 1.0 - 1e-4), target))

        if pose_losses:
            pose_loss = torch.stack(pose_losses).mean()
            total = total + pose_loss
            details["v8_pose_prompt_pose_loss"] = float(pose_loss.detach())
        if latent_terms and self.latent_weight > 0:
            latent_loss = torch.stack(latent_terms).mean()
            total = total + self.latent_weight * latent_loss
            details["v8_pose_prompt_latent_loss"] = float(latent_loss.detach())
            details["v8_pose_prompt_latent_loss_weighted"] = float((self.latent_weight * latent_loss).detach())
        if gate_terms and self.gate_weight > 0:
            gate_loss = torch.stack(gate_terms).mean()
            total = total + self.gate_weight * gate_loss
            details["v8_pose_prompt_gate_loss"] = float(gate_loss.detach())
            details["v8_pose_prompt_gate_loss_weighted"] = float((self.gate_weight * gate_loss).detach())

        if trans_errs:
            details["v8_pose_prompt_trans_err"] = float(torch.stack(trans_errs).mean())
            details["v8_pose_prompt_rot_err_deg"] = float(torch.rad2deg(torch.stack(rot_errs).mean()))
        if raw_trans_errs:
            details["v8_raw_trans_err"] = float(torch.stack(raw_trans_errs).mean())
            details["v8_raw_rot_err_deg"] = float(torch.rad2deg(torch.stack(raw_rot_errs).mean()))
        if gate_values:
            details["v8_pose_prompt_gate_mean"] = float(torch.stack(gate_values).mean())
        if delta_norms:
            details["v8_pose_prompt_delta_norm"] = float(torch.stack(delta_norms).mean())
        details["v8_pose_prompt_loss"] = float(total.detach())
        return total, details


class V82PoseRelationLoss(V81PosePromptLoss):
    """V8.2 loss for the UniCon-style pose-relation prompt.

    Compared with V81PosePromptLoss, this criterion adds relation-specific
    supervision:
      - drift/gate target from raw camera-pose error,
      - improvement margin against the raw pose,
      - residual-size regularization.
    """

    def __init__(
        self,
        translation_weight=1.0,
        rotation_weight=1.0,
        residual_weight=1.0e-4,
        drift_weight=0.2,
        improvement_weight=0.1,
        loss_type="smooth_l1",
        pose_key="camera_pose",
        drift_trans_scale=0.5,
        drift_rot_scale_deg=45.0,
        improvement_margin=0.0,
        pose_noop_before_view=-1,
        pose_noop_weight=0.0,
        human_trans_weight=0.0,
        human_trans_delta_weight=0.0,
        human_trans_supervise_from_view=0,
        human_trans_noop_before_view=-1,
        human_trans_noop_weight=0.0,
        pose_lora_norm_weight=0.0,
        human_lora_norm_weight=0.0,
    ):
        super().__init__(
            translation_weight=translation_weight,
            rotation_weight=rotation_weight,
            latent_weight=residual_weight,
            gate_weight=0.0,
            loss_type=loss_type,
            pose_key=pose_key,
        )
        self.residual_weight = float(residual_weight)
        self.drift_weight = float(drift_weight)
        self.improvement_weight = float(improvement_weight)
        self.drift_trans_scale = float(drift_trans_scale)
        self.drift_rot_scale_deg = float(drift_rot_scale_deg)
        self.improvement_margin = float(improvement_margin)
        self.pose_noop_before_view = int(pose_noop_before_view)
        self.pose_noop_weight = float(pose_noop_weight)
        self.human_trans_weight = float(human_trans_weight)
        self.human_trans_delta_weight = float(human_trans_delta_weight)
        self.human_trans_supervise_from_view = int(human_trans_supervise_from_view)
        self.human_trans_noop_before_view = int(human_trans_noop_before_view)
        self.human_trans_noop_weight = float(human_trans_noop_weight)
        self.pose_lora_norm_weight = float(pose_lora_norm_weight)
        self.human_lora_norm_weight = float(human_lora_norm_weight)

    def get_name(self):
        return "V82PoseRelationLoss"

    def _normalized_pose_error(self, trans_err, rot_err):
        trans_scale = max(self.drift_trans_scale, 1e-6)
        rot_scale = max(math.radians(self.drift_rot_scale_deg), 1e-6)
        return trans_err / trans_scale + rot_err / rot_scale

    def _human_trans_mask(self, gt, pred_transl):
        mask = gt.get("smpl_mask", None)
        if mask is None:
            return torch.ones(
                pred_transl.shape[:2],
                device=pred_transl.device,
                dtype=torch.bool,
            )
        mask = mask.to(device=pred_transl.device, dtype=torch.bool)
        img_mask = gt.get("img_mask", None)
        if img_mask is not None:
            img_mask = img_mask.to(device=pred_transl.device, dtype=torch.bool)
            mask = mask & img_mask[:, None]
        return mask

    def compute_loss(self, gts, preds, **kw):
        ref = self._first_tensor(gts, preds)
        if ref is None:
            return torch.zeros(()), {}
        zero = ref.float().new_zeros(())
        gt_poses = self._gt_pose_encodings(gts)

        total = zero
        details = {}
        pose_losses = []
        trans_errs = []
        rot_errs = []
        raw_trans_errs = []
        raw_rot_errs = []
        gate_values = []
        delta_norms = []
        residual_terms = []
        pose_noop_terms = []
        drift_terms = []
        drift_targets = []
        improvement_terms = []
        improvements = []
        human_trans_terms = []
        human_trans_delta_terms = []
        human_trans_noop_terms = []
        human_trans_errs = []
        raw_human_trans_errs = []
        pose_lora_norm_terms = []
        human_lora_norm_terms = []

        for view_idx, (gt_pose, gt, pred) in enumerate(zip(gt_poses, gts, preds)):
            pred_pose = pred.get("camera_pose", None)
            if pred_pose is None:
                continue
            pred_pose = pred_pose.float()
            gt_pose = gt_pose.to(device=pred_pose.device, dtype=pred_pose.dtype)

            pose_loss = self._pose_loss(pred_pose, gt_pose)
            pose_losses.append(pose_loss.mean())

            trans_err_for_loss, rot_err_for_loss = self._pose_errors(pred_pose, gt_pose)
            corrected_norm_err_for_loss = self._normalized_pose_error(
                trans_err_for_loss,
                rot_err_for_loss,
            )
            trans_err, rot_err = self._pose_errors(pred_pose.detach(), gt_pose.detach())
            trans_errs.append(trans_err.mean())
            rot_errs.append(rot_err.mean())
            details[f"v82_trans_err/{view_idx}"] = float(trans_err.mean())
            details[f"v82_rot_err_deg/{view_idx}"] = float(torch.rad2deg(rot_err.mean()))

            corrected_norm_err = self._normalized_pose_error(trans_err, rot_err)
            raw_norm_err = None
            raw_pose = pred.get("v8_raw_camera_pose", None)
            if raw_pose is not None:
                raw_pose = raw_pose.to(device=pred_pose.device, dtype=pred_pose.dtype)
                raw_t_err, raw_r_err = self._pose_errors(raw_pose.detach(), gt_pose.detach())
                raw_trans_errs.append(raw_t_err.mean())
                raw_rot_errs.append(raw_r_err.mean())
                raw_norm_err = self._normalized_pose_error(raw_t_err, raw_r_err)
                details[f"v82_raw_trans_err/{view_idx}"] = float(raw_t_err.mean())
                details[f"v82_raw_rot_err_deg/{view_idx}"] = float(torch.rad2deg(raw_r_err.mean()))

                if self.improvement_weight > 0:
                    margin_term = F.relu(
                        corrected_norm_err_for_loss - raw_norm_err.detach() + self.improvement_margin
                    )
                    improvement_terms.append(margin_term.mean())
                    improvements.append((raw_norm_err - corrected_norm_err).mean())

                if self.drift_weight > 0:
                    drift_target = raw_norm_err.clamp(0.0, 1.0).detach().reshape(-1, 1, 1)
                    drift_targets.append(drift_target.mean())
                    drift_logit = pred.get("v8_pose_prompt_drift_logit", None)
                    gate = pred.get("v8_pose_prompt_gate", None)
                    if drift_logit is not None:
                        drift_logit = drift_logit.float().reshape_as(drift_target)
                        drift_terms.append(F.binary_cross_entropy_with_logits(drift_logit, drift_target))
                    elif gate is not None:
                        gate_for_loss = gate.float().reshape_as(drift_target).clamp(1e-4, 1.0 - 1e-4)
                        drift_terms.append(F.binary_cross_entropy(gate_for_loss, drift_target))

            delta_raw = pred.get("v8_pose_prompt_delta_raw", None)
            if delta_raw is not None:
                delta_raw = delta_raw.float()
                residual_terms.append(delta_raw.pow(2).mean())
                delta_norm = delta_raw.detach().norm(dim=-1).mean()
                delta_norms.append(delta_norm)
                details[f"v82_delta_norm/{view_idx}"] = float(delta_norm)

            delta_applied = pred.get("v8_pose_prompt_delta_applied", None)
            if delta_applied is not None:
                delta_applied = delta_applied.float()
                details[f"v82_delta_applied_norm/{view_idx}"] = float(
                    delta_applied.detach().norm(dim=-1).mean()
                )
                if (
                    self.pose_noop_weight > 0
                    and self.pose_noop_before_view >= 0
                    and view_idx < self.pose_noop_before_view
                ):
                    pose_noop_loss = F.smooth_l1_loss(delta_applied, torch.zeros_like(delta_applied))
                    pose_noop_terms.append(pose_noop_loss)
                    details[f"v82_pose_noop_loss/{view_idx}"] = float(pose_noop_loss.detach())

            gate = pred.get("v8_pose_prompt_gate", None)
            if gate is not None:
                gate = gate.float()
                gate_values.append(gate.detach().mean())
                details[f"v82_gate/{view_idx}"] = float(gate.detach().mean())

            pose_lora_l2 = pred.get("v8_pose_head_lora_l2", None)
            if pose_lora_l2 is not None:
                pose_lora_l2 = pose_lora_l2.float()
                pose_lora_norm_terms.append(pose_lora_l2)
                details[f"v82_pose_head_lora_l2/{view_idx}"] = float(pose_lora_l2.detach())

            human_lora_l2 = pred.get("v8_human_head_lora_l2", None)
            if human_lora_l2 is not None:
                human_lora_l2 = human_lora_l2.float()
                human_lora_norm_terms.append(human_lora_l2)
                details[f"v82_human_head_lora_l2/{view_idx}"] = float(human_lora_l2.detach())

            if (
                (self.human_trans_weight > 0 or self.human_trans_noop_weight > 0)
                and "smpl_transl" in gt
                and "smpl_transl" in pred
            ):
                pred_human_t = pred["smpl_transl"].float()
                gt_human_t = gt["smpl_transl"].to(device=pred_human_t.device, dtype=pred_human_t.dtype)
                num_humans = min(pred_human_t.shape[1], gt_human_t.shape[1])
                pred_human_t = pred_human_t[:, :num_humans]
                gt_human_t = gt_human_t[:, :num_humans]
                mask = self._human_trans_mask(gt, pred_human_t)[:, :num_humans]
                if mask.any():
                    human_err = torch.norm((pred_human_t - gt_human_t).detach(), dim=-1)[mask].mean()
                    human_trans_errs.append(human_err)
                    details[f"v82_human_trans_err/{view_idx}"] = float(human_err)

                    raw_human_t = pred.get("v8_human_trans_corr_smpl_transl_raw", None)
                    if raw_human_t is not None:
                        raw_human_t = raw_human_t.to(device=pred_human_t.device, dtype=pred_human_t.dtype)
                        raw_human_t = raw_human_t[:, :num_humans]
                        raw_human_err = torch.norm((raw_human_t - gt_human_t).detach(), dim=-1)[mask].mean()
                        raw_human_trans_errs.append(raw_human_err)
                        details[f"v82_raw_human_trans_err/{view_idx}"] = float(raw_human_err)

                    if self.human_trans_weight > 0 and view_idx >= self.human_trans_supervise_from_view:
                        human_loss = F.smooth_l1_loss(pred_human_t[mask], gt_human_t[mask])
                        human_trans_terms.append(human_loss)

                    if (
                        self.human_trans_noop_weight > 0
                        and self.human_trans_noop_before_view >= 0
                        and view_idx < self.human_trans_noop_before_view
                    ):
                        applied_delta = pred.get("v8_human_trans_corr_delta_applied", None)
                        if applied_delta is not None:
                            applied_delta = applied_delta.float()[:, :num_humans]
                            noop_loss = F.smooth_l1_loss(applied_delta[mask], torch.zeros_like(applied_delta[mask]))
                            human_trans_noop_terms.append(noop_loss)
                            details[f"v82_human_trans_noop_loss/{view_idx}"] = float(noop_loss.detach())
                            details[f"v82_human_trans_applied_delta_norm/{view_idx}"] = float(
                                applied_delta.detach().norm(dim=-1)[mask].mean()
                            )

                human_delta = pred.get("v8_human_trans_corr_delta_raw", None)
                if human_delta is not None and self.human_trans_delta_weight > 0:
                    human_delta = human_delta.float()
                    human_trans_delta_terms.append(human_delta.pow(2).mean())
                    details[f"v82_human_trans_delta_norm/{view_idx}"] = float(
                        human_delta.detach().norm(dim=-1).mean()
                    )

        if pose_losses:
            pose_loss = torch.stack(pose_losses).mean()
            total = total + pose_loss
            details["v82_pose_loss"] = float(pose_loss.detach())
        if residual_terms and self.residual_weight > 0:
            residual_loss = torch.stack(residual_terms).mean()
            total = total + self.residual_weight * residual_loss
            details["v82_residual_small_loss"] = float(residual_loss.detach())
            details["v82_residual_small_loss_weighted"] = float((self.residual_weight * residual_loss).detach())
        if pose_noop_terms and self.pose_noop_weight > 0:
            pose_noop_loss = torch.stack(pose_noop_terms).mean()
            total = total + self.pose_noop_weight * pose_noop_loss
            details["v82_pose_noop_loss"] = float(pose_noop_loss.detach())
            details["v82_pose_noop_loss_weighted"] = float((self.pose_noop_weight * pose_noop_loss).detach())
        if drift_terms and self.drift_weight > 0:
            drift_loss = torch.stack(drift_terms).mean()
            total = total + self.drift_weight * drift_loss
            details["v82_drift_loss"] = float(drift_loss.detach())
            details["v82_drift_loss_weighted"] = float((self.drift_weight * drift_loss).detach())
        if improvement_terms and self.improvement_weight > 0:
            improvement_loss = torch.stack(improvement_terms).mean()
            total = total + self.improvement_weight * improvement_loss
            details["v82_improvement_margin_loss"] = float(improvement_loss.detach())
            details["v82_improvement_margin_loss_weighted"] = float((self.improvement_weight * improvement_loss).detach())
        if human_trans_terms and self.human_trans_weight > 0:
            human_trans_loss = torch.stack(human_trans_terms).mean()
            total = total + self.human_trans_weight * human_trans_loss
            details["v82_human_trans_loss"] = float(human_trans_loss.detach())
            details["v82_human_trans_loss_weighted"] = float((self.human_trans_weight * human_trans_loss).detach())
        if human_trans_delta_terms and self.human_trans_delta_weight > 0:
            human_delta_loss = torch.stack(human_trans_delta_terms).mean()
            total = total + self.human_trans_delta_weight * human_delta_loss
            details["v82_human_trans_delta_small_loss"] = float(human_delta_loss.detach())
            details["v82_human_trans_delta_small_loss_weighted"] = float(
                (self.human_trans_delta_weight * human_delta_loss).detach()
            )
        if human_trans_noop_terms and self.human_trans_noop_weight > 0:
            human_noop_loss = torch.stack(human_trans_noop_terms).mean()
            total = total + self.human_trans_noop_weight * human_noop_loss
            details["v82_human_trans_noop_loss"] = float(human_noop_loss.detach())
            details["v82_human_trans_noop_loss_weighted"] = float(
                (self.human_trans_noop_weight * human_noop_loss).detach()
            )
        if pose_lora_norm_terms:
            pose_lora_norm = torch.stack(pose_lora_norm_terms).mean()
            details["v82_pose_head_lora_l2"] = float(pose_lora_norm.detach())
            if self.pose_lora_norm_weight > 0:
                total = total + self.pose_lora_norm_weight * pose_lora_norm
                details["v82_pose_head_lora_l2_weighted"] = float(
                    (self.pose_lora_norm_weight * pose_lora_norm).detach()
                )
        if human_lora_norm_terms:
            human_lora_norm = torch.stack(human_lora_norm_terms).mean()
            details["v82_human_head_lora_l2"] = float(human_lora_norm.detach())
            if self.human_lora_norm_weight > 0:
                total = total + self.human_lora_norm_weight * human_lora_norm
                details["v82_human_head_lora_l2_weighted"] = float(
                    (self.human_lora_norm_weight * human_lora_norm).detach()
                )

        if trans_errs:
            details["v82_trans_err"] = float(torch.stack(trans_errs).mean())
            details["v82_rot_err_deg"] = float(torch.rad2deg(torch.stack(rot_errs).mean()))
        if raw_trans_errs:
            details["v82_raw_trans_err"] = float(torch.stack(raw_trans_errs).mean())
            details["v82_raw_rot_err_deg"] = float(torch.rad2deg(torch.stack(raw_rot_errs).mean()))
        if gate_values:
            details["v82_gate_mean"] = float(torch.stack(gate_values).mean())
        if delta_norms:
            details["v82_delta_norm"] = float(torch.stack(delta_norms).mean())
        if drift_targets:
            details["v82_drift_target_mean"] = float(torch.stack(drift_targets).mean())
        if improvements:
            details["v82_norm_error_improvement"] = float(torch.stack(improvements).mean())
        if human_trans_errs:
            details["v82_human_trans_err"] = float(torch.stack(human_trans_errs).mean())
        if raw_human_trans_errs:
            details["v82_raw_human_trans_err"] = float(torch.stack(raw_human_trans_errs).mean())

        # Keep the old V8 names in logs so existing parsers can still find the
        # headline metrics while V8.2-specific keys carry the new losses.
        if "v82_pose_loss" in details:
            details["v8_pose_prompt_pose_loss"] = details["v82_pose_loss"]
        if "v82_trans_err" in details:
            details["v8_pose_prompt_trans_err"] = details["v82_trans_err"]
        if "v82_rot_err_deg" in details:
            details["v8_pose_prompt_rot_err_deg"] = details["v82_rot_err_deg"]
        if "v82_raw_trans_err" in details:
            details["v8_raw_trans_err"] = details["v82_raw_trans_err"]
        if "v82_raw_rot_err_deg" in details:
            details["v8_raw_rot_err_deg"] = details["v82_raw_rot_err_deg"]
        if "v82_gate_mean" in details:
            details["v8_pose_prompt_gate_mean"] = details["v82_gate_mean"]
        if "v82_delta_norm" in details:
            details["v8_pose_prompt_delta_norm"] = details["v82_delta_norm"]
        if "v82_human_trans_err" in details:
            details["v8_human_trans_corr_err"] = details["v82_human_trans_err"]
        if "v82_raw_human_trans_err" in details:
            details["v8_raw_human_trans_err"] = details["v82_raw_human_trans_err"]
        details["v82_pose_relation_loss"] = float(total.detach())
        details["v8_pose_prompt_loss"] = float(total.detach())
        return total, details


class ConfLoss(MultiLoss):
    """Weighted regression by learned confidence.
        Assuming the input pixel_loss is a pixel-level regression loss.

    Principle:
        high-confidence means high conf = 0.1 ==> conf_loss = x / 10 + alpha*log(10)
        low  confidence means low  conf = 10  ==> conf_loss = x * 10 - alpha*log(10)

        alpha: hyperparameter
    """

    def __init__(self, pixel_loss, alpha=1):
        super().__init__()
        assert alpha > 0
        self.alpha = alpha
        self.pixel_loss = pixel_loss.with_reduction("none")

    def get_name(self):
        return f"ConfLoss({self.pixel_loss})"

    def get_conf_log(self, x):
        return x, torch.log(x)

    def compute_loss(self, gts, preds, **kw):
        # compute per-pixel loss
        losses_and_masks, details = self.pixel_loss(gts, preds, **kw)
        if "is_self" in details and "img_ids" in details:
            is_self = details["is_self"]
            img_ids = details["img_ids"]
        else:
            is_self = [False] * len(losses_and_masks)
            img_ids = list(range(len(losses_and_masks)))

        # weight by confidence
        conf_losses = []

        for i in range(len(losses_and_masks)):
            pred = preds[img_ids[i]]
            conf_key = "conf_self" if is_self[i] else "conf"
            if not is_self[i]:
                camera_only = gts[0]["camera_only"]
                conf, log_conf = self.get_conf_log(
                    pred[conf_key][~camera_only][losses_and_masks[i][1]]
                )
            else:
                conf, log_conf = self.get_conf_log(
                    pred[conf_key][losses_and_masks[i][1]]
                )

            conf_loss = losses_and_masks[i][0] * conf - self.alpha * log_conf
            conf_loss = conf_loss.mean() if conf_loss.numel() > 0 else 0
            conf_losses.append(conf_loss)

            if is_self[i]:
                details[self.get_name() + f"_conf_loss_self/{img_ids[i]+1}"] = float(
                    conf_loss
                )
            else:
                details[self.get_name() + f"_conf_loss/{img_ids[i]+1}"] = float(
                    conf_loss
                )

        details.pop("is_self", None)
        details.pop("img_ids", None)

        final_loss = sum(conf_losses) / len(conf_losses) * 2.0
        if "pose_loss" in details:
            final_loss = (
                final_loss + details["pose_loss"]
            )  # , details
        if "scale_loss" in details:
            final_loss = final_loss + details["scale_loss"]
        return final_loss, details


class Regr3DPose_ScaleInv(Regr3DPose):
    """Same than Regr3D but invariant to depth shift.
    if gt_scale == True: enforce the prediction to take the same scale than GT
    """

    def get_all_pts3d(self, gts, preds):
        # compute depth-normalized points
        (
            gt_pts_self,
            gt_pts_cross,
            pr_pts_self,
            pr_pts_cross,
            gt_poses,
            pr_poses,
            masks,
            skys,
            pose_masks,
            monitoring,
        ) = super().get_all_pts3d(gts, preds)

        # measure scene scale
        _, gt_scale_self = get_group_pointcloud_center_scale(gt_pts_self, masks)
        _, pred_scale_self = get_group_pointcloud_center_scale(pr_pts_self, masks)

        _, gt_scale_cross = get_group_pointcloud_center_scale(gt_pts_cross, masks)
        _, pred_scale_cross = get_group_pointcloud_center_scale(pr_pts_cross, masks)

        # prevent predictions to be in a ridiculous range
        pred_scale_self = pred_scale_self.clip(min=1e-3, max=1e3)
        pred_scale_cross = pred_scale_cross.clip(min=1e-3, max=1e3)

        # subtract the median depth
        if self.gt_scale:
            pr_pts_self = [
                pr_pt_self * gt_scale_self / pred_scale_self
                for pr_pt_self in pr_pts_self
            ]
            pr_pts_cross = [
                pr_pt_cross * gt_scale_cross / pred_scale_cross
                for pr_pt_cross in pr_pts_cross
            ]
        else:
            gt_pts_self = [gt_pt_self / gt_scale_self for gt_pt_self in gt_pts_self]
            gt_pts_cross = [
                gt_pt_cross / gt_scale_cross for gt_pt_cross in gt_pts_cross
            ]
            pr_pts_self = [pr_pt_self / pred_scale_self for pr_pt_self in pr_pts_self]
            pr_pts_cross = [
                pr_pt_cross / pred_scale_cross for pr_pt_cross in pr_pts_cross
            ]

        return (
            gt_pts_self,
            gt_pts_cross,
            pr_pts_self,
            pr_pts_cross,
            gt_poses,
            pr_poses,
            masks,
            skys,
            pose_masks,
            monitoring,
        )
