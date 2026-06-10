# --------------------------------------------------------
# training code for Human3R
# --------------------------------------------------------
# References:
# CUT3R: https://github.com/CUT3R/CUT3R
# DUSt3R: https://github.com/naver/dust3r
# --------------------------------------------------------
#
# 阅读建议：这个文件是训练编排层，不直接实现模型结构。
# 主要职责是把 Hydra 配置、DataLoader、模型、loss、optimizer、Accelerate/DDP、
# checkpoint、TensorBoard 串起来。真正的模型 forward 在 dust3r/model.py，
# 真正的 forward+loss 调用入口在 dust3r/inference.py::loss_of_one_batch。
import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
import math
from collections import defaultdict
from pathlib import Path
from typing import Sized

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dust3r.utils.device import todevice

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

from dust3r.model import (
    PreTrainedModel,
    ARCroco3DStereo,
    ARCroco3DStereoConfig,
    inf,
    strip_module,
    strip_module_mhmr,
)  # noqa: F401, needed when loading the model
from dust3r.smpl_model import SMPLModel
from dust3r.datasets import get_data_loader
from dust3r.losses import *  # noqa: F401, needed when loading the model
from dust3r.inference import loss_of_one_batch  # noqa
from dust3r.viz import colorize
from dust3r.utils.render import get_render_results, get_render_smpl
import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
# change to gradient accu 
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa

import hydra
from omegaconf import OmegaConf
import logging
import pathlib
from tqdm import tqdm
import random
import builtins
import shutil

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.logging import get_logger
from datetime import timedelta
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

printer = get_logger(__name__, log_level="DEBUG")


def setup_for_distributed(accelerator: Accelerator):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (accelerator.num_processes > 8)
        if accelerator.is_main_process or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def save_current_code(outdir):
    # 训练开始时保存当前代码快照到 output_dir/code/时间戳。
    # 这样后续回看 checkpoint 时，可以知道当时使用的是哪一版源码。
    now = datetime.datetime.now()  # current date and time
    date_time = now.strftime("%m_%d-%H:%M:%S")
    src_dir = "."
    dst_dir = os.path.join(outdir, "code", "{}".format(date_time))
    shutil.copytree(
        src_dir,
        dst_dir,
        ignore=shutil.ignore_patterns(
            ".vscode*",
            "assets*",
            "example*",
            "checkpoints*",
            "OLD*",
            "logs*",
            "out*",
            "runs*",
            "*.png",
            "*.mp4",
            "*__pycache__*",
            "*.git*",
            "*.idea*",
            "*.zip",
            "*.jpg",
            "*.pth",
            "*.pt",
            "*.npy",
            "*.npz",
            "*.pkl",
        ),
        dirs_exist_ok=True,
    )
    return dst_dir


DEFAULT_CONSOLE_LOG_KEYS = [
    "lr",
    "loss",
    "regr_self_pts3d_avg",
    "regr_cross_pts3d_avg",
    "rgb_loss_avg",
    "pose_loss",
    "smpl_transl_avg",
    "shot_bce",
    "shot_acc",
    "shot_prob_gap",
    "shot_q_energy_cont",
    "shot_q_energy_jump",
    "shot_noop_loss",
    "v7_pose_pseudo_loss",
    "v7_pose_delta_t_err",
    "v7_pose_delta_r_err_deg",
    "v7_pose_alpha",
    "v82_pose_head_lora_l2",
    "v82_human_head_lora_l2",
]

DEFAULT_STEP_LOG_KEYS = [
    "loss",
    "lr",
    "regr_self_pts3d_avg",
    "regr_cross_pts3d_avg",
    "conf_loss_avg",
    "rgb_loss_avg",
    "pose_loss",
    "pose_loss_view2_AABB",
    "smpl_scores_avg",
    "smpl_rotmat_avg",
    "smpl_transl_avg",
    "smpl_shape_avg",
    "smpl_j3d_avg",
    "shot_bce",
    "shot_acc",
    "shot_prob_pos",
    "shot_prob_neg",
    "shot_prob_gap",
    "shot_label_pos_frac",
    "shot_q0_loss",
    "shot_q_energy_cont",
    "shot_q_energy_jump",
    "shot_noop_loss",
    "shot_noop_camera_pose",
    "shot_noop_pts3d_in_self_view",
    "shot_noop_pts3d_in_other_view",
    "shot_noop_smpl_transl",
    "v7_pose_pseudo_loss",
    "v7_pose_label_count",
    "v7_pose_delta_t_err",
    "v7_pose_delta_r_err_deg",
    "v7_pose_alpha_err",
    "v7_pose_r_human_err",
    "v7_pose_r_scene_err",
    "v7_pose_alpha",
    "v7_pose_r_human",
    "v7_pose_r_scene",
    "v7_pose_delta_t_norm",
    "v7_pose_delta_r_deg",
    "v82_pose_head_lora_l2",
    "v82_human_head_lora_l2",
]


def _safe_float(value):
    if isinstance(value, torch.Tensor):
        if value.ndim > 0:
            return None
        value = value.detach().float().item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (float, int)) and math.isfinite(float(value)):
        return float(value)
    return None


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _mean_loss_detail(loss_details, predicate):
    values = []
    for name, value in loss_details.items():
        if not predicate(name):
            continue
        value = _safe_float(value)
        if value is not None:
            values.append(value)
    if not values:
        return None
    return float(sum(values) / len(values))


def summarize_loss_details(loss_details):
    """Aggregate verbose per-view loss_details into stable diagnostic signals."""
    groups = {
        "regr_self_pts3d_avg": lambda k: "self_pts3d" in k,
        "regr_cross_pts3d_avg": lambda k: "Regr3DPose" in k and "_pts3d/" in k and "self_pts3d" not in k,
        "conf_loss_avg": lambda k: "_conf_loss" in k,
        "rgb_loss_avg": lambda k: k.startswith("RGBLoss_rgb"),
        "smpl_scores_avg": lambda k: k.startswith("SMPLLoss_scores"),
        "smpl_rotmat_avg": lambda k: k.startswith("SMPLLoss_rotmat"),
        "smpl_transl_avg": lambda k: k.startswith("SMPLLoss_transl"),
        "smpl_shape_avg": lambda k: k.startswith("SMPLLoss_shape"),
        "smpl_j3d_avg": lambda k: k.startswith("SMPLLoss_j3d"),
        "smpl_j2d_avg": lambda k: k.startswith("SMPLLoss_j2d"),
    }
    summary = {}
    for name, predicate in groups.items():
        value = _mean_loss_detail(loss_details, predicate)
        if value is not None:
            summary[name] = value

    shot_prob_pos = _safe_float(loss_details.get("shot_prob_pos"))
    shot_prob_neg = _safe_float(loss_details.get("shot_prob_neg"))
    if shot_prob_pos is not None and shot_prob_neg is not None:
        summary["shot_prob_gap"] = shot_prob_pos - shot_prob_neg
    return summary


def _get_list_arg(args, name, default):
    value = getattr(args, name, None)
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def _write_jsonl(path, record):
    with open(path, mode="a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _make_compact_record(source, keys):
    record = {}
    for key in keys:
        if key not in source:
            continue
        value = _safe_float(source[key])
        if value is not None:
            record[key] = value
    return record


def train(args):
    """
    训练总入口。

    这个函数负责完整训练生命周期：
    1. 初始化 Accelerate/DDP、随机种子和输出目录。
    2. 构建 train/val/test DataLoader。
    3. 构建模型、SMPLModel、loss、optimizer。
    4. 加载 pretrained 或自动 resume。
    5. 进入 epoch loop，按顺序执行 eval、保存、early stopping、train_one_epoch。
    """

    # Accelerator 封装了多卡 DDP、混合精度、梯度累积、梯度裁剪等逻辑。
    # torchrun 会启动多个进程，每个进程都会执行 train(args)，
    # accelerator 会根据环境变量自动判断 rank/world_size/local device。
    accelerator = Accelerator(
        gradient_accumulation_steps=args.accum_iter,
        mixed_precision="bf16",
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True),
            InitProcessGroupKwargs(timeout=timedelta(seconds=6000)),
        ],
    )
    device = accelerator.device

    # 非主进程默认不打印，避免多卡训练日志重复刷屏。
    setup_for_distributed(accelerator)

    # output_dir 是当前实验的根目录。注意：如果里面已有 checkpoint-last.pth，
    # 后面会自动 resume，所以新实验建议使用新的 output_dir。
    printer.info("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        # **========== 原始代码备份：每次启动训练都复制一份完整源码快照 ==========**
        # dst_dir = save_current_code(outdir=args.output_dir)
        # printer.info(f"Saving current code to {dst_dir}")
        # **========== 新代码：默认不复制源码快照，避免正式训练目录膨胀；需要时可设置 save_code=true ==========**
        if _as_bool(getattr(args, "save_code", False)):
            dst_dir = save_current_code(outdir=args.output_dir)
            printer.info(f"Saving current code to {dst_dir}")
        else:
            printer.info("Skipping code snapshot; set save_code=true to enable it")
        # **========== 结束 ==========**

    # auto resume
    # 如果命令行没有显式传 resume，但 output_dir 里有 checkpoint-last.pth，
    # 就自动恢复训练。这个机制很方便，但也容易误把新实验接到旧目录上。
    if not args.resume:
        last_ckpt_fname = os.path.join(args.output_dir, f"checkpoint-last.pth")
        args.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

    printer.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))

    # fix the seed
    # 多卡时每个进程 seed 稍微不同，避免不同 rank 采样完全一致。
    seed = args.seed + accelerator.state.process_index
    printer.info(
        f"Setting seed to {seed} for process {accelerator.state.process_index}"
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = args.benchmark

    # training dataset and loader
    # args.train_dataset 是一个字符串表达式，例如：
    # "800 @ AvatarReX_Video(...) + 800 @ AvatarReX_AABB(...)"。
    # get_data_loader 内部会 eval 这个字符串并构造真正的 PyTorch Dataset。
    printer.info("Building train dataset %s", args.train_dataset)
    #  dataset and loader
    data_loader_train = build_dataset(
        args.train_dataset,
        args.batch_size,
        args.num_workers,
        accelerator=accelerator,
        test=False,
        fixed_length=args.fixed_length
    )
    # test/val dataset 允许用 "+" 拼多个数据集，这里拆开后分别构建 loader，
    # 方便后续分别统计每个数据集的 loss。
    printer.info("Building test dataset %s", args.test_dataset)
    data_loader_test = {
        dataset.split("(")[0]: build_dataset(
            dataset,
            args.batch_size,
            args.num_workers,
            accelerator=accelerator,
            test=True,
            fixed_length=True
        )
        for dataset in args.test_dataset.split("+")
    }

    # validation dataset
    data_loader_val = None
    if hasattr(args, 'val_dataset') and args.val_dataset:
        printer.info("Building val dataset %s", args.val_dataset)
        data_loader_val = {
            dataset.split("(")[0]: build_dataset(
                dataset,
                args.batch_size,
                args.num_workers,
                accelerator=accelerator,
                test=True,
                fixed_length=True
            )
            for dataset in args.val_dataset.split("+")
        }

    # model
    # args.model 也是字符串表达式，来自 config/train.yaml。
    # 当前 Movie3R 典型值是 ARCroco3DStereo(ARCroco3DStereoConfig(... lora_rank=64))。
    printer.info("Loading model: %s", args.model)
    model: PreTrainedModel = eval(args.model)
    # SMPLModel 用于把 GT / prediction 的 SMPL 参数转换成 joints、vertices、render 等监督信号。
    smpl_model: SMPLModel = SMPLModel(
        device, 
        model_args={
            'patch_size': model.croco_args['patch_size'], 
            'mhmr_img_res': model.mhmr_img_res, 
            'bb_patch_size': model.bb_patch_size
        })
    printer.info(f"All model parameters: {sum(p.numel() for p in model.parameters())}")
    printer.info(
        f"Encoder parameters: {sum(p.numel() for p in model.enc_blocks.parameters())}"
    )
    printer.info(
        f"Decoder parameters: {sum(p.numel() for p in model.dec_blocks.parameters())}"
    )

    # train/test criterion 也是字符串表达式，eval 后得到真正的 loss module。
    # 当前训练 loss 大致是 3D point/pose loss + RGB loss + SMPL loss。
    printer.info(f">> Creating train criterion = {args.train_criterion}")
    train_criterion = eval(args.train_criterion).to(device)
    printer.info(
        f">> Creating test criterion = {args.test_criterion or args.train_criterion}"
    )
    test_criterion = eval(args.test_criterion or args.criterion).to(device)

    model.to(device)

    # gradient checkpointing 用更多计算换显存，适合大模型和多视角输入。
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.long_context:
        model.fixed_input_length = False

    # 新实验：从 pretrained Human3R/CUT3R 权重初始化。
    # resume 实验：跳过这里，后面 misc.load_model 会恢复完整训练状态。
    if args.pretrained and not args.resume:
        printer.info(f"Loading pretrained: {args.pretrained}")
        # **========== 原始代码备份：pretrained 直接加载到训练 device，merge_state_dict 会在 train() 内持续占用显存 ==========**
        # ckpt = torch.load(args.pretrained, map_location=device)
        # load_only_encoder = getattr(args, "load_only_encoder", False)
        # if load_only_encoder:
        #     filtered_state_dict = {
        #         k: v
        #         for k, v in ckpt["model"].items()
        #         if "enc_blocks" in k or "patch_embed" in k
        #     }
        #     merge_state_dict = strip_module(filtered_state_dict)
        # else:
        #     merge_state_dict = strip_module(ckpt["model"])
        # del ckpt  # in case it occupies memory
        #
        # if args.pretrained_mhmr:
        #     printer.info(f"Loading Multi-HMR pretrained: {args.pretrained_mhmr}")
        #     ckpt_mhmr = torch.load(args.pretrained_mhmr, map_location=device)
        #     merge_state_dict.update(strip_module_mhmr(ckpt_mhmr["model_state_dict"]))
        #     del ckpt_mhmr  # in case it occupies memory
        #
        # printer.info(
        #     model.load_state_dict(merge_state_dict, strict=False)
        # )
        # **========== 新代码：先加载到 CPU，load_state_dict 后释放临时权重，避免微调时额外常驻一份 GPU checkpoint ==========**
        ckpt = torch.load(args.pretrained, map_location="cpu")
        load_only_encoder = getattr(args, "load_only_encoder", False)
        if load_only_encoder:
            filtered_state_dict = {
                k: v
                for k, v in ckpt["model"].items()
                if "enc_blocks" in k or "patch_embed" in k
            }
            merge_state_dict = strip_module(filtered_state_dict)
        else:
            merge_state_dict = strip_module(ckpt["model"])
        del ckpt  # in case it occupies memory

        if args.pretrained_mhmr:
            printer.info(f"Loading Multi-HMR pretrained: {args.pretrained_mhmr}")
            ckpt_mhmr = torch.load(args.pretrained_mhmr, map_location="cpu")
            merge_state_dict.update(strip_module_mhmr(ckpt_mhmr["model_state_dict"]))
            del ckpt_mhmr  # in case it occupies memory

        load_result = model.load_state_dict(merge_state_dict, strict=False)
        printer.info(load_result)
        del merge_state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # **========== 结束 ==========**

    # # following timm: set wd as 0 for bias and norm layers
    # 只会把 requires_grad=True 的参数放进 optimizer。
    # 当前 freeze='shot_adaptation' 时，主模型冻结，optimizer 里只有 ShotToken/LoRA 参数。
    param_groups = misc.get_parameter_groups(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # print(optimizer)
    # NativeScaler 在这里主要负责 accelerator.backward、梯度裁剪和 optimizer.step。
    loss_scaler = NativeScaler(accelerator=accelerator)

    # prepare 是 Accelerate 的关键步骤：
    # - 多卡时把 model 包装成 DDP
    # - 包装 optimizer
    # - 包装 train dataloader，使每个 rank 拿到自己的数据切片
    accelerator.even_batches = False
    optimizer, model, data_loader_train = accelerator.prepare(
        optimizer, model, data_loader_train
    )

    def write_log_stats(epoch, train_stats, test_stats, val_stats=None):
        # 只在主进程写 JSON 日志，避免多卡同时写同一个 log.txt。
        if accelerator.is_main_process:
            if not train_stats and not test_stats and not val_stats:
                printer.info("Skipping empty epoch log before first train step")
                return

            if log_writer is not None:
                log_writer.flush()

            log_stats = dict(
                epoch=epoch, **{f"train_{k}": v for k, v in train_stats.items()}
            )
            for test_name in data_loader_test:
                if test_name not in test_stats:
                    continue
                log_stats.update(
                    {test_name + "_" + k: v for k, v in test_stats[test_name].items()}
                )
            # Add val stats if available
            if val_stats:
                for val_name in data_loader_val:
                    if val_name not in val_stats:
                        continue
                    log_stats.update(
                        {val_name + "_val_" + k: v for k, v in val_stats[val_name].items()}
                    )

            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

            compact_record = {"phase": "epoch", "epoch": epoch}
            compact_keys = _get_list_arg(args, "step_log_keys", DEFAULT_STEP_LOG_KEYS)
            train_source = {f"train_{k}": v for k, v in train_stats.items()}
            compact_record.update(_make_compact_record(train_source, [f"train_{k}" for k in compact_keys]))
            compact_record.update(
                _make_compact_record(
                    log_stats,
                    [k for k in log_stats if k.endswith("loss_avg") or k.endswith("loss_med")],
                )
            )
            _write_jsonl(os.path.join(args.output_dir, "metrics_epoch.jsonl"), compact_record)

    def save_model(epoch, fname, best_so_far):
        # 中间 checkpoint 保存，包括模型、optimizer、scaler、epoch、best_so_far 等训练状态。
        misc.save_model(
            accelerator=accelerator,
            args=args,
            model_without_ddp=model,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            fname=fname,
            best_so_far=best_so_far,
        )

    # 如果 args.resume 指向 checkpoint，这里会恢复模型和 optimizer 状态。
    best_so_far = misc.load_model(
        args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler
    )
    if best_so_far is None:
        best_so_far = float("inf")
    log_writer = (
        SummaryWriter(log_dir=args.output_dir) if accelerator.is_main_process else None
    )

    printer.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_stats = test_stats = {}

    # Early stopping state
    epochs_without_improvement = 0
    early_stop_patience = getattr(args, 'early_stopping_patience', 10)

    for epoch in range(args.start_epoch, args.epochs + 1):

        # Save immediately the last checkpoint
        # epoch 开头先保存上一轮 last，这样即使后面训练中断，也有最近 checkpoint。
        if epoch > args.start_epoch and getattr(args, "save_last_checkpoint", True):
            if (
                args.save_freq
                and np.allclose(epoch / args.save_freq, int(epoch / args.save_freq))
                or epoch == args.epochs
            ):
                save_model(epoch - 1, "last", best_so_far)

        # Validation (run before training, except epoch 0)
        # 注意：eval 发生在当前 epoch 的训练之前。
        # 因此 epoch=0 且 eval_freq=1 时，会先跑一轮 val/test 再开始第 0 轮训练。
        val_stats = {}
        if data_loader_val is not None and epoch >= 0 and args.eval_freq > 0 and epoch % args.eval_freq == 0:
            val_stats = {}
            for val_name, valset in data_loader_val.items():
                stats = test_one_epoch(
                    model,
                    test_criterion,
                    valset,
                    accelerator,
                    device,
                    epoch,
                    log_writer=log_writer,
                    args=args,
                    prefix=val_name + "_val",
                    smpl_model=smpl_model,
                )
                val_stats[val_name] = stats

        # Test on multiple datasets
        # test 和 val 都通过 test_one_epoch 执行，区别只是 prefix 和数据集来源。
        new_best = False
        if epoch >= 0 and args.eval_freq > 0 and epoch % args.eval_freq == 0:
            test_stats = {}
            for test_name, testset in data_loader_test.items():
                stats = test_one_epoch(
                    model,
                    test_criterion,
                    testset,
                    accelerator,
                    device,
                    epoch,
                    log_writer=log_writer,
                    args=args,
                    prefix=test_name,
                    smpl_model=smpl_model,
                )
                test_stats[test_name] = stats

                # Save best based on val loss if available, else test loss
                monitor_loss = None
                if val_stats and val_name in val_stats:
                    monitor_loss = val_stats[val_name]["loss_med"]
                else:
                    monitor_loss = stats["loss_med"]

                if monitor_loss < best_so_far:
                    best_so_far = monitor_loss
                    new_best = True

        # Save more stuff
        # 每个 epoch 写一行 JSON 到 output_dir/log.txt，同时刷新 TensorBoard。
        write_log_stats(epoch, train_stats, test_stats, val_stats)

        if epoch > args.start_epoch:
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch - 1, str(epoch), best_so_far)
            if new_best:
                save_model(epoch - 1, "best", best_so_far)

        # Early stopping check
        # early stopping 依赖 eval_freq；如果 eval_freq=0，就不会触发 early stopping。
        if data_loader_val is not None and epoch >= 0 and args.eval_freq > 0 and epoch % args.eval_freq == 0:
            if new_best:
                epochs_without_improvement = 0
                printer.info(f"Validation improved: {best_so_far:.4f}")
            else:
                epochs_without_improvement += args.eval_freq
                printer.info(f"No improvement for {epochs_without_improvement} epochs")
                if epochs_without_improvement >= early_stop_patience:
                    printer.info(f"Early stopping triggered after {epoch} epochs")
                    break

        if epoch >= args.epochs:
            break  # exit after writing last test to disk

        # Train
        # 真正的一轮训练在 train_one_epoch 里完成。
        train_stats = train_one_epoch(
            model,
            train_criterion,
            data_loader_train,
            optimizer,
            accelerator,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
            smpl_model=smpl_model,
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    printer.info("Training time {}".format(total_time_str))

    if getattr(args, "save_final_checkpoint", True):
        save_final_model(accelerator, args, args.epochs, model, best_so_far=best_so_far)
    else:
        printer.info("Skipping final checkpoint because save_final_checkpoint=false")


def save_final_model(accelerator, args, epoch, model_without_ddp, best_so_far=None):
    # 训练结束时额外保存一个 checkpoint-final.pth。
    # 它主要用于记录最终模型权重；resume 训练通常依赖 checkpoint-last.pth。
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "checkpoint-final.pth"
    to_save = {
        "args": args,
        "model": (
            model_without_ddp
            if isinstance(model_without_ddp, dict)
            else model_without_ddp.cpu().state_dict()
        ),
        "epoch": epoch,
    }
    if best_so_far is not None:
        to_save["best_so_far"] = best_so_far
    printer.info(f">> Saving model to {checkpoint_path} ...")
    misc.save_on_master(accelerator, to_save, checkpoint_path)


def build_dataset(dataset, batch_size, num_workers, accelerator, test=False, fixed_length=False):
    # 统一封装 DataLoader 创建逻辑。
    # train loader 会 shuffle/drop_last；test/val loader 不 shuffle、不 drop_last。
    # 多卡时 sampler 会根据 accelerator.num_processes 把数据切给不同 rank。
    split = ["Train", "Test"][test]
    printer.info(f"Building {split} Data loader for dataset: {dataset}")
    loader = get_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        shuffle=not (test),
        drop_last=not (test),
        accelerator=accelerator,
        fixed_length=fixed_length
    )
    return loader


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    epoch: int,
    loss_scaler,
    args,
    log_writer=None,
    smpl_model: SMPLModel = None
):
    """
    执行一个 epoch 的训练。

    单个 iteration 的核心顺序：
    1. 从 DataLoader 取 batch。
    2. 根据 epoch_f 调整 learning rate。
    3. loss_of_one_batch 执行 model forward + criterion。
    4. accelerator.backward 反传。
    5. clip grad、optimizer.step、optimizer.zero_grad。
    6. 记录终端日志、TensorBoard scalar，可选写可视化图。
    """
    assert torch.backends.cuda.matmul.allow_tf32 == True

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.display_keys = _get_list_arg(args, "console_log_keys", DEFAULT_CONSOLE_LOG_KEYS)
    header = "Epoch: [{}]".format(epoch)
    accum_iter = args.accum_iter

    def save_model(epoch, fname, best_so_far):
        misc.save_model(
            accelerator=accelerator,
            args=args,
            model_without_ddp=model,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            fname=fname,
            best_so_far=best_so_far,
        )

    if log_writer is not None:
        printer.info("log_dir: {}".format(log_writer.log_dir))

    # 对支持 set_epoch 的 dataset/sampler，显式设置 epoch，保证多卡 shuffle 可复现。
    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if (
        hasattr(data_loader, "batch_sampler")
        and hasattr(data_loader.batch_sampler, "batch_sampler")
        and hasattr(data_loader.batch_sampler.batch_sampler, "set_epoch")
    ):
        data_loader.batch_sampler.batch_sampler.set_epoch(epoch)

    optimizer.zero_grad()

    # metric_logger.log_every 包装了 dataloader，会周期性打印当前 loss、ETA、显存等信息。
    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, accelerator, header)
    ):
        iter_start_time = time.time()
        # accelerator.accumulate 根据 args.accum_iter 决定何时同步梯度/更新参数。
        # 当前 accum_iter=1，因此每个 batch 都会更新一次参数。
        with accelerator.accumulate(model):
            # epoch_f 是浮点 epoch，用于 iteration-level learning rate schedule。
            epoch_f = epoch + data_iter_step / len(data_loader)
            step = int(epoch_f * len(data_loader))
            # we use a per iteration (instead of per epoch) lr scheduler
            if data_iter_step % accum_iter == 0:
                misc.adjust_learning_rate(optimizer, epoch_f, args)
            if not args.long_context:
                # loss_of_one_batch 是训练 step 的核心：
                # smpl_model.update_smpl_gt(batch) -> model(batch) -> criterion(batch, preds)。
                result = loss_of_one_batch(
                    batch,
                    model,
                    criterion,
                    accelerator,
                    symmetrize_batch=False,
                    use_amp=bool(args.amp),
                    smpl_model=smpl_model
                )
            else:
                NotImplementedError("Long context is not supported")
            has_msk = "msk" in result["pred"][0]
            loss, loss_details = result["loss"]  # criterion returns two values
            loss_value = float(loss)

            if not math.isfinite(loss_value):
                print(
                    f"Loss is {loss_value}, stopping training, loss details: {loss_details}"
                )
                sys.exit(1)
            if not result.get("already_backprop", False):
                # 这里完成 backward、梯度裁剪和 optimizer.step。
                # 对 Movie3R LoRA 训练来说，只有 ShotToken/LoRA 参数有梯度会被更新。
                loss_scaler(
                    loss,
                    optimizer,
                    parameters=model.parameters(),
                    update_grad=True,
                    clip_grad=1.0,
                )
                optimizer.zero_grad()

            # batch 是 list[view_dict]，curr_num_view 通常等于 config.num_views。
            is_metric = batch[0]["is_metric"]
            curr_num_view = len(batch)

            del loss
            tb_vis_img = (data_iter_step + 1) % accum_iter == 0 and (
                (step + 1) % (args.print_img_freq)
            ) == 0
            if not tb_vis_img:
                del batch
            else:
                torch.cuda.empty_cache()

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(epoch=epoch_f)
            metric_logger.update(lr=lr)
            metric_logger.update(step=step)

            # loss_details 里包含各个子 loss，例如 pointmap、RGB、SMPL、mask 等。
            loss_summary = summarize_loss_details(loss_details)
            metric_logger.update(loss=loss_value, **loss_details, **loss_summary)

            structured_log_freq = int(getattr(args, "structured_log_freq", args.print_freq))
            if structured_log_freq > 0 and (data_iter_step + 1) % structured_log_freq == 0:
                if accelerator.is_main_process:
                    elapsed = time.time() - iter_start_time
                    global_batch_size = int(args.batch_size) * int(accelerator.num_processes)
                    step_source = {"loss": loss_value, "lr": lr, **loss_details, **loss_summary}
                    step_record = {
                        "phase": "train",
                        "epoch": float(epoch_f),
                        "epoch_int": int(epoch),
                        "iter": int(data_iter_step),
                        "step": int(step),
                        "global_batch_size": global_batch_size,
                        "iter_time_sec": elapsed,
                        "samples_per_sec": global_batch_size / elapsed if elapsed > 0 else 0.0,
                    }
                    if torch.cuda.is_available():
                        step_record["max_mem_mb"] = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                    step_record.update(
                        _make_compact_record(
                            step_source,
                            _get_list_arg(args, "step_log_keys", DEFAULT_STEP_LOG_KEYS),
                        )
                    )
                    _write_jsonl(os.path.join(args.output_dir, "train_steps.jsonl"), step_record)

            if (data_iter_step + 1) % accum_iter == 0 and (
                (data_iter_step + 1) % (accum_iter * args.print_freq)
            ) == 0:
                # 多卡时先 gather 再 mean，保证 TensorBoard 上是全局平均 loss。
                loss_value_reduce = accelerator.gather(
                    torch.tensor(loss_value).to(accelerator.device)
                ).mean()  # MUST BE EXECUTED BY ALL NODES

                if log_writer is None:
                    continue
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int(epoch_f * 1000)
                log_writer.add_scalar("train_loss", loss_value_reduce, step)
                log_writer.add_scalar("train_lr", lr, step)
                log_writer.add_scalar("train_iter", epoch_1000x, step)
                for name, val in loss_summary.items():
                    log_writer.add_scalar("train_summary/" + name, val, step)
                for name, val in loss_details.items():
                    if isinstance(val, torch.Tensor):
                        if val.ndim > 0:
                            continue
                    if isinstance(val, dict):
                        continue
                    log_writer.add_scalar("train_" + name, val, step)

            if tb_vis_img:
                # 可视化很耗时/显存，正式训练通常通过 print_img_freq=999999 关闭。
                if log_writer is None:
                    continue
                with torch.no_grad():
                    depths_self, gt_depths_self = get_render_results(
                        batch, result["pred"], self_view=True
                    )
                    depths_cross, gt_depths_cross = get_render_results(
                        batch, result["pred"], self_view=False
                    )
                    gt_msks, pr_msks, gt_hms, pr_hms, gt_smpls, pr_smpls = get_render_smpl(
                        batch, result["pred"], smpl_model, loss_details, has_msk=has_msk
                    )
                    for k in range(len(batch)):
                        loss_details[f"self_pred_depth_{k+1}"] = depths_self[k].detach().cpu()
                        loss_details[f"self_gt_depth_{k+1}"] = gt_depths_self[k].detach().cpu()
                        loss_details[f"pred_depth_{k+1}"] = depths_cross[k].detach().cpu()
                        loss_details[f"gt_depth_{k+1}"] = gt_depths_cross[k].detach().cpu()           
                        loss_details[f"pred_hm_{k+1}"] = pr_hms[k].detach().cpu()
                        loss_details[f"gt_hm_{k+1}"] = gt_hms[k].detach().cpu()
                        loss_details[f"pred_smpl_rend_{k+1}"] = pr_smpls[k].detach().cpu()
                        loss_details[f"gt_smpl_rend_{k+1}"] = gt_smpls[k].detach().cpu()
                        if has_msk:
                            loss_details[f"pred_msk_{k+1}"] = pr_msks[k].detach().cpu()
                            loss_details[f"gt_msk_{k+1}"] = gt_msks[k].detach().cpu()

                imgs_stacked_dict = get_vis_imgs_new(
                    loss_details, 
                    args.num_imgs_vis, 
                    curr_num_view, 
                    is_metric=is_metric, 
                    has_msk=has_msk)
                for name, imgs_stacked in imgs_stacked_dict.items():
                    log_writer.add_images(
                        "train" + "/" + name, imgs_stacked, step, dataformats="HWC"
                    )
                del batch

        if (
            data_iter_step % int(args.save_freq * len(data_loader)) == 0
            and data_iter_step != 0
            and data_iter_step != len(data_loader) - 1
        ):
            # 按 step 保存中间 checkpoint。
            # 注意 save_freq 不能设为 0，否则 int(args.save_freq * len(data_loader)) 会导致除零。
            print("saving at step", data_iter_step)
            save_model(epoch - 1, "last", float("inf"))

    # gather the stats from all processes
    # 多卡训练结束后同步各 rank 的指标，返回全局平均值。
    metric_logger.synchronize_between_processes(accelerator)
    printer.info("Averaged stats: %s", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    accelerator: Accelerator,
    device: torch.device,
    epoch: int,
    args,
    log_writer=None,
    prefix="test",
    smpl_model: SMPLModel = None
):
    """
    执行一个 validation/test epoch。

    与 train_one_epoch 的区别：
    - 使用 model.eval() 和 @torch.no_grad()。
    - 不做 backward / optimizer.step。
    - 汇总 avg 和 median 两套指标。
    - 可选写 TensorBoard 可视化。
    """

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    metric_logger.display_keys = _get_list_arg(args, "console_log_keys", DEFAULT_CONSOLE_LOG_KEYS)
    header = "Test Epoch: [{}]".format(epoch)

    if log_writer is not None:
        printer.info("log_dir: {}".format(log_writer.log_dir))

    # eval 阶段固定 epoch=0，保证测试集顺序稳定。
    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(0)
    if (
        hasattr(data_loader, "batch_sampler")
        and hasattr(data_loader.batch_sampler, "batch_sampler")
        and hasattr(data_loader.batch_sampler.batch_sampler, "set_epoch")
    ):
        data_loader.batch_sampler.batch_sampler.set_epoch(0)

    for _, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, accelerator, header)
    ):
        # eval loader 没有经过 accelerator.prepare，这里显式搬到当前 device。
        batch = todevice(batch, device)
        result = loss_of_one_batch(
            batch,
            model,
            criterion,
            accelerator,
            symmetrize_batch=False,
            use_amp=bool(args.amp),
            smpl_model=smpl_model
        )

        has_msk = "msk" in result["pred"][0]
        loss_value, loss_details = result["loss"]  # criterion returns two values
        loss_summary = summarize_loss_details(loss_details)
        metric_logger.update(loss=float(loss_value), **loss_details, **loss_summary)

    printer.info("Averaged stats: %s", metric_logger)

    # 同时返回平均值和中位数。best checkpoint 当前主要使用 loss_med 作为监控指标。
    aggs = [("avg", "global_avg"), ("med", "median")]
    results = {
        f"{k}_{tag}": getattr(meter, attr)
        for k, meter in metric_logger.meters.items()
        for tag, attr in aggs
    }

    if log_writer is not None:
        for name, val in results.items():
            if isinstance(val, torch.Tensor):
                if val.ndim > 0:
                    continue
            if isinstance(val, dict):
                continue
            log_writer.add_scalar(prefix + "_" + name, val, 1000 * epoch)

        # Skip visualization if print_img_freq is set to a very large value (disabled)
        # print_img_freq>=10000 时跳过可视化，减少正式训练时的评估开销。
        tb_vis_img = args.print_img_freq < 10000
        if not tb_vis_img:
            del loss_details, loss_value, batch
            torch.cuda.empty_cache()
            return results

        depths_self, gt_depths_self = get_render_results(
            batch, result["pred"], self_view=True
        )
        depths_cross, gt_depths_cross = get_render_results(
            batch, result["pred"], self_view=False
        )
        gt_msks, pr_msks, gt_hms, pr_hms, gt_smpls, pr_smpls = get_render_smpl(
            batch, result["pred"], smpl_model, loss_details, has_msk=has_msk
        )
        for k in range(len(batch)):
            loss_details[f"self_pred_depth_{k+1}"] = depths_self[k].detach().cpu()
            loss_details[f"self_gt_depth_{k+1}"] = gt_depths_self[k].detach().cpu()
            loss_details[f"pred_depth_{k+1}"] = depths_cross[k].detach().cpu()
            loss_details[f"gt_depth_{k+1}"] = gt_depths_cross[k].detach().cpu()
            loss_details[f"pred_hm_{k+1}"] = pr_hms[k].detach().cpu()
            loss_details[f"gt_hm_{k+1}"] = gt_hms[k].detach().cpu()
            loss_details[f"pred_smpl_rend_{k+1}"] = pr_smpls[k].detach().cpu()
            loss_details[f"gt_smpl_rend_{k+1}"] = gt_smpls[k].detach().cpu()
            if has_msk:
                loss_details[f"pred_msk_{k+1}"] = pr_msks[k].detach().cpu()
                loss_details[f"gt_msk_{k+1}"] = gt_msks[k].detach().cpu()

        imgs_stacked_dict = get_vis_imgs_new(
            loss_details,
            args.num_imgs_vis,
            args.num_test_views,
            is_metric=batch[0]["is_metric"],
            has_msk=has_msk
        )
        for name, imgs_stacked in imgs_stacked_dict.items():
            log_writer.add_images(
                prefix + "/" + name, imgs_stacked, 1000 * epoch, dataformats="HWC"
            )

    del loss_details, loss_value, batch
    torch.cuda.empty_cache()

    return results


def batch_append(original_list, new_list):
    for sublist, new_item in zip(original_list, new_list):
        sublist.append(new_item)
    return original_list


def gen_mask_indicator(img_mask_list, ray_mask_list, num_views, h, w):
    output = []
    for img_mask, ray_mask in zip(img_mask_list, ray_mask_list):
        out = torch.zeros((h, w * num_views, 3))
        for i in range(num_views):
            if img_mask[i] and not ray_mask[i]:
                offset = 0
            elif not img_mask[i] and ray_mask[i]:
                offset = 1
            else:
                offset = 0.5
            out[:, i * w : (i + 1) * w] += offset
        output.append(out)
    return output


def vis_and_cat(
    gt_imgs,
    pred_imgs,
    gt_msks,
    pred_msks,
    gt_hms,
    pred_hms,
    gt_smpl_rends,
    pred_smpl_rends,
    cross_gt_depths,
    cross_pred_depths,
    self_gt_depths,
    self_pred_depths,
    cross_conf,
    self_conf,
    ray_indicator,
    is_metric,
    has_msk=False
):
    cross_depth_gt_min = torch.quantile(cross_gt_depths, 0.01).item()
    cross_depth_gt_max = torch.quantile(cross_gt_depths, 0.99).item()
    cross_depth_pred_min = torch.quantile(cross_pred_depths, 0.01).item()
    cross_depth_pred_max = torch.quantile(cross_pred_depths, 0.99).item()
    cross_depth_min = min(cross_depth_gt_min, cross_depth_pred_min)
    cross_depth_max = max(cross_depth_gt_max, cross_depth_pred_max)

    cross_gt_depths_vis = colorize(
        cross_gt_depths,
        range=(
            (cross_depth_min, cross_depth_max)
            if is_metric
            else (cross_depth_gt_min, cross_depth_gt_max)
        ),
        append_cbar=True,
    )
    cross_pred_depths_vis = colorize(
        cross_pred_depths,
        range=(
            (cross_depth_min, cross_depth_max)
            if is_metric
            else (cross_depth_pred_min, cross_depth_pred_max)
        ),
        append_cbar=True,
    )

    self_depth_gt_min = torch.quantile(self_gt_depths, 0.01).item()
    self_depth_gt_max = torch.quantile(self_gt_depths, 0.99).item()
    self_depth_pred_min = torch.quantile(self_pred_depths, 0.01).item()
    self_depth_pred_max = torch.quantile(self_pred_depths, 0.99).item()
    self_depth_min = min(self_depth_gt_min, self_depth_pred_min)
    self_depth_max = max(self_depth_gt_max, self_depth_pred_max)

    self_gt_depths_vis = colorize(
        self_gt_depths,
        range=(
            (self_depth_min, self_depth_max)
            if is_metric
            else (self_depth_gt_min, self_depth_gt_max)
        ),
        append_cbar=True,
    )
    self_pred_depths_vis = colorize(
        self_pred_depths,
        range=(
            (self_depth_min, self_depth_max)
            if is_metric
            else (self_depth_pred_min, self_depth_pred_max)
        ),
        append_cbar=True,
    )
    if len(cross_conf) > 0:
        cross_conf_vis = colorize(cross_conf, append_cbar=True)
    if len(self_conf) > 0:
        self_conf_vis = colorize(self_conf, append_cbar=True)
    gt_imgs_vis = torch.zeros_like(cross_gt_depths_vis)
    gt_imgs_vis[: gt_imgs.shape[0], : gt_imgs.shape[1]] = gt_imgs
    pred_imgs_vis = torch.zeros_like(cross_gt_depths_vis)
    pred_imgs_vis[: pred_imgs.shape[0], : pred_imgs.shape[1]] = pred_imgs
    if has_msk:
        gt_msks_vis = torch.zeros_like(cross_gt_depths_vis)
        gt_msks_vis[: gt_msks.shape[0], : gt_msks.shape[1]] = gt_msks
        pred_msks_vis = torch.zeros_like(cross_gt_depths_vis)
        pred_msks_vis[: pred_msks.shape[0], : pred_msks.shape[1]] = pred_msks
    gt_hms_vis = torch.zeros_like(cross_gt_depths_vis)
    gt_hms_vis[: gt_hms.shape[0], : gt_hms.shape[1]] = gt_hms
    pred_hms_vis = torch.zeros_like(cross_gt_depths_vis)
    pred_hms_vis[: pred_hms.shape[0], : pred_hms.shape[1]] = pred_hms
    gt_smpl_rends_vis = torch.zeros_like(cross_gt_depths_vis)
    gt_smpl_rends_vis[: gt_smpl_rends.shape[0], : gt_smpl_rends.shape[1]] = gt_smpl_rends
    pred_smpl_rends_vis = torch.zeros_like(cross_gt_depths_vis)
    pred_smpl_rends_vis[: pred_smpl_rends.shape[0], : pred_smpl_rends.shape[1]] = pred_smpl_rends
    ray_indicator_vis = torch.cat(
        [
            ray_indicator,
            torch.zeros(
                ray_indicator.shape[0],
                cross_pred_depths_vis.shape[1] - ray_indicator.shape[1],
                3,
            ),
        ],
        dim=1,
    )
    if has_msk:
        out = torch.cat(
            [
                ray_indicator_vis,
                gt_imgs_vis,
                pred_imgs_vis,
                gt_msks_vis,
                pred_msks_vis,
                gt_hms_vis,
                pred_hms_vis,
                gt_smpl_rends_vis,
                pred_smpl_rends_vis,
                self_gt_depths_vis,
                self_pred_depths_vis,
                self_conf_vis,
                cross_gt_depths_vis,
                cross_pred_depths_vis,
                cross_conf_vis,
            ],
            dim=0,
        )
    else:
        out = torch.cat(
            [
                ray_indicator_vis,
                gt_imgs_vis,
                pred_imgs_vis,
                gt_hms_vis,
                pred_hms_vis,
                gt_smpl_rends_vis,
                pred_smpl_rends_vis,
                self_gt_depths_vis,
                self_pred_depths_vis,
                self_conf_vis,
                cross_gt_depths_vis,
                cross_pred_depths_vis,
                cross_conf_vis,
            ],
            dim=0,
        )
    return out


def get_vis_imgs_new(loss_details, num_imgs_vis, num_views, is_metric, has_msk=False):
    ret_dict = {}
    gt_img_list = [[] for _ in range(num_imgs_vis)]
    pred_img_list = [[] for _ in range(num_imgs_vis)]

    cross_gt_depth_list = [[] for _ in range(num_imgs_vis)]
    cross_pred_depth_list = [[] for _ in range(num_imgs_vis)]

    self_gt_depth_list = [[] for _ in range(num_imgs_vis)]
    self_pred_depth_list = [[] for _ in range(num_imgs_vis)]

    gt_msk_list = [[] for _ in range(num_imgs_vis)]
    pred_msk_list = [[] for _ in range(num_imgs_vis)]
    gt_hm_list = [[] for _ in range(num_imgs_vis)]
    pred_hm_list = [[] for _ in range(num_imgs_vis)]
    gt_smpl_rend_list = [[] for _ in range(num_imgs_vis)]
    pred_smpl_rend_list = [[] for _ in range(num_imgs_vis)]

    cross_view_conf_list = [[] for _ in range(num_imgs_vis)]
    self_view_conf_list = [[] for _ in range(num_imgs_vis)]
    cross_view_conf_exits = False
    self_view_conf_exits = False

    img_mask_list = [[] for _ in range(num_imgs_vis)]
    ray_mask_list = [[] for _ in range(num_imgs_vis)]

    if num_views > 30:
        stride = 5
    elif num_views > 20:
        stride = 3
    elif num_views > 10:
        stride = 2
    else:
        stride = 1
    for i in range(0, num_views, stride):
        gt_imgs = 0.5 * (loss_details[f"gt_img{i+1}"] + 1)[:num_imgs_vis].detach().cpu()
        width = gt_imgs.shape[2]
        pred_imgs = (
            0.5 * (loss_details[f"pred_rgb_{i+1}"] + 1)[:num_imgs_vis].detach().cpu()
        )
        gt_img_list = batch_append(gt_img_list, gt_imgs.unbind(dim=0))
        pred_img_list = batch_append(pred_img_list, pred_imgs.unbind(dim=0))

        cross_pred_depths = (
            loss_details[f"pred_depth_{i+1}"][:num_imgs_vis].detach().cpu()
        )
        cross_gt_depths = (
            loss_details[f"gt_depth_{i+1}"]
            .to(gt_imgs.device)[:num_imgs_vis]
            .detach()
            .cpu()
        )
        cross_pred_depth_list = batch_append(
            cross_pred_depth_list, cross_pred_depths.unbind(dim=0)
        )
        cross_gt_depth_list = batch_append(
            cross_gt_depth_list, cross_gt_depths.unbind(dim=0)
        )

        self_gt_depths = (
            loss_details[f"self_gt_depth_{i+1}"][:num_imgs_vis].detach().cpu()
        )
        self_pred_depths = (
            loss_details[f"self_pred_depth_{i+1}"][:num_imgs_vis].detach().cpu()
        )
        self_gt_depth_list = batch_append(
            self_gt_depth_list, self_gt_depths.unbind(dim=0)
        )
        self_pred_depth_list = batch_append(
            self_pred_depth_list, self_pred_depths.unbind(dim=0)
        )

        if has_msk:
            gt_msks = (
                loss_details[f"gt_msk_{i+1}"][:num_imgs_vis].detach().cpu()
            )
            pred_msks = (
                loss_details[f"pred_msk_{i+1}"][:num_imgs_vis].detach().cpu()
            )
        gt_hms = (
            loss_details[f"gt_hm_{i+1}"][:num_imgs_vis].detach().cpu()
        )
        pred_hms = (
            loss_details[f"pred_hm_{i+1}"][:num_imgs_vis].detach().cpu()
        )
        gt_smpl_rends = (
            loss_details[f"gt_smpl_rend_{i+1}"][:num_imgs_vis].detach().cpu()
        )
        pred_smpl_rends = (
            loss_details[f"pred_smpl_rend_{i+1}"][:num_imgs_vis].detach().cpu()
        )
        if has_msk:
            gt_msk_list = batch_append(gt_msk_list, gt_msks.unbind(dim=0))
            pred_msk_list = batch_append(pred_msk_list, pred_msks.unbind(dim=0))
        gt_hm_list = batch_append(gt_hm_list, gt_hms.unbind(dim=0))
        pred_hm_list = batch_append(pred_hm_list, pred_hms.unbind(dim=0))
        gt_smpl_rend_list = batch_append(
            gt_smpl_rend_list, gt_smpl_rends.unbind(dim=0))
        pred_smpl_rend_list = batch_append(
            pred_smpl_rend_list, pred_smpl_rends.unbind(dim=0))

        if f"conf_{i+1}" in loss_details:
            cross_view_conf = loss_details[f"conf_{i+1}"][:num_imgs_vis].detach().cpu()
            cross_view_conf_list = batch_append(
                cross_view_conf_list, cross_view_conf.unbind(dim=0)
            )
            cross_view_conf_exits = True

        if f"self_conf_{i+1}" in loss_details:
            self_view_conf = (
                loss_details[f"self_conf_{i+1}"][:num_imgs_vis].detach().cpu()
            )
            self_view_conf_list = batch_append(
                self_view_conf_list, self_view_conf.unbind(dim=0)
            )
            self_view_conf_exits = True

        img_mask_list = batch_append(
            img_mask_list,
            loss_details[f"img_mask_{i+1}"][:num_imgs_vis].detach().cpu().unbind(dim=0),
        )
        ray_mask_list = batch_append(
            ray_mask_list,
            loss_details[f"ray_mask_{i+1}"][:num_imgs_vis].detach().cpu().unbind(dim=0),
        )

    # each element in the list is [H, num_views * W, (3)], the size of the list is num_imgs_vis
    gt_img_list = [torch.cat(sublist, dim=1) for sublist in gt_img_list]
    pred_img_list = [torch.cat(sublist, dim=1) for sublist in pred_img_list]
    cross_pred_depth_list = [
        torch.cat(sublist, dim=1) for sublist in cross_pred_depth_list
    ]
    cross_gt_depth_list = [torch.cat(sublist, dim=1) for sublist in cross_gt_depth_list]
    self_gt_depth_list = [torch.cat(sublist, dim=1) for sublist in self_gt_depth_list]
    self_pred_depth_list = [
        torch.cat(sublist, dim=1) for sublist in self_pred_depth_list
    ]
    if has_msk:
        gt_msk_list = [torch.cat(sublist, dim=1) for sublist in gt_msk_list]
        pred_msk_list = [torch.cat(sublist, dim=1) for sublist in pred_msk_list]
    gt_hm_list = [torch.cat(sublist, dim=1) for sublist in gt_hm_list]
    pred_hm_list = [torch.cat(sublist, dim=1) for sublist in pred_hm_list]
    gt_smpl_rend_list = [torch.cat(sublist, dim=1) for sublist in gt_smpl_rend_list]
    pred_smpl_rend_list = [torch.cat(sublist, dim=1) for sublist in pred_smpl_rend_list]
    cross_view_conf_list = (
        [torch.cat(sublist, dim=1) for sublist in cross_view_conf_list]
        if cross_view_conf_exits
        else []
    )
    self_view_conf_list = (
        [torch.cat(sublist, dim=1) for sublist in self_view_conf_list]
        if self_view_conf_exits
        else []
    )
    # each elment in the list is [num_views,], the size of the list is num_imgs_vis
    img_mask_list = [torch.stack(sublist, dim=0) for sublist in img_mask_list]
    ray_mask_list = [torch.stack(sublist, dim=0) for sublist in ray_mask_list]

    ray_indicator = gen_mask_indicator(
        img_mask_list, ray_mask_list, len(img_mask_list[0]), 30, width
    )

    for i in range(num_imgs_vis):
        out = vis_and_cat(
            gt_img_list[i],
            pred_img_list[i],
            gt_msk_list[i],
            pred_msk_list[i],
            gt_hm_list[i],
            pred_hm_list[i],
            gt_smpl_rend_list[i],
            pred_smpl_rend_list[i],
            cross_gt_depth_list[i],
            cross_pred_depth_list[i],
            self_gt_depth_list[i],
            self_pred_depth_list[i],
            cross_view_conf_list[i],
            self_view_conf_list[i],
            ray_indicator[i],
            is_metric[i],
            has_msk=has_msk
        )
        ret_dict[f"imgs_{i}"] = out
    return ret_dict


@hydra.main(
    version_base=None,
    config_path=str(os.path.dirname(os.path.abspath(__file__))) + "/../config",
    config_name="train.yaml",
)
def run(cfg: OmegaConf):
    # Hydra 入口。启动命令中的 overrides 会在这里合并到 config/train.yaml。
    # 例如：python train.py epochs=30 batch_size=2 output_dir=...
    OmegaConf.resolve(cfg)
    # 创建 logdir 后进入真正的训练主函数。
    logdir = pathlib.Path(cfg.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    train(cfg)


if __name__ == "__main__":
    run()
