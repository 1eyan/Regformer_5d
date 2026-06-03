#!/usr/bin/env python3
"""E2E V9 training for query-context seismic interpolation.

The active training path is direct reconstruction:

    masked_patch + coords + observed_mask -> V9 EncDec -> reconstructed traces

Loss is focused on query/missing traces by default.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config.data_config import get_parser
from config.segy_config import print_info as print_segy_config
from dataset import DatasetH5_all_queryctx
from model import create_gated_model_v9_encdec, trace_time_chunk, trace_time_unchunk
from utils.coord_utils import build_rope_frequency_config


def str2bool(v):
    if isinstance(v, bool):
        return v
    if str(v).lower() in ("yes", "true", "t", "y", "1"):
        return True
    if str(v).lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def optional_float(v):
    if v is None:
        return None
    text = str(v).strip().lower()
    if text in ("", "none", "null", "auto"):
        return None
    return float(v)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(parents=[get_parser()], conflict_handler="resolve")

    parser.add_argument("--model_name", type=str, default="e2e_v9")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=515)
    parser.add_argument("--data_type", type=str, default="df_field1031_5d")
    parser.add_argument("--results_dir", type=str, default="./resultsE2E")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"])

    parser.add_argument("--target_mode", choices=["self", "supervised"], default="self")
    parser.add_argument("--h5File_tgt", type=str, default=None)
    parser.add_argument("--chunk_length", type=int, default=256)
    parser.add_argument("--overlap_ratio", type=float, default=0.125)
    parser.add_argument("--query_loss_weight", type=float, default=1.0)
    parser.add_argument("--context_loss_weight", type=float, default=0.0)
    parser.add_argument("--energy_loss_weight", type=float, default=2.0)
    parser.add_argument("--hf_grad_loss_weight", type=float, default=0.2)
    parser.add_argument("--phase_loss_weight", type=float, default=0.0)
    parser.add_argument("--coord_aug_scale", type=float, default=0.0,
                        help="Coord augmentation strength (>0 = enable rotation+scaling+centering). "
                             "0=disabled (default). Suggested: 0.01~0.05")

    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_encoder_layers", type=int, default=None)
    parser.add_argument("--num_decoder_layers", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--elementwise_attn_output_gate", type=str2bool, default=True)
    parser.add_argument("--headwise_attn_output_gate", type=str2bool, default=False)
    parser.add_argument("--use_qk_norm", type=str2bool, default=False)
    parser.add_argument("--qkv_bias", type=str2bool, default=False)
    parser.add_argument("--rms_norm_eps", type=float, default=1e-8)
    parser.add_argument("--hidden_act", type=str, default="gelu")
    parser.add_argument("--use_coord_encoding", type=str2bool, default=True)
    parser.add_argument("--use_rope", type=str2bool, default=True)
    parser.add_argument("--coord_dim", type=int, default=6)
    parser.add_argument("--num_attn_res_blocks", type=int, default=2)
    parser.add_argument("--encode_observed_only", type=str2bool, default=True,
                        help="If True, encoder only processes observed tokens; "
                             "if False, encoder processes the full sequence.")
    parser.add_argument("--rope_freq_mode", choices=["default", "physical"], default="default")
    parser.add_argument("--lambda_phys_x", type=optional_float, default=None)
    parser.add_argument("--lambda_phys_y", type=optional_float, default=None)
    parser.add_argument("--rope_nyquist_safety", type=float, default=1.0)

    args = parser.parse_args()
    if getattr(args, "use_phys_omega", False) and args.rope_freq_mode == "default":
        args.rope_freq_mode = "physical"
    return args


def _as_array(sample: Dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(sample[key], dtype=np.float32)


def collate_queryctx(samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    max_tr = max(int(s["data"].shape[0]) for s in samples)
    time_len = int(samples[0]["data"].shape[1])
    batch_size = len(samples)

    out = {
        "data": np.zeros((batch_size, max_tr, time_len), dtype=np.float32),
        "masked_patch": np.zeros((batch_size, max_tr, time_len), dtype=np.float32),
        "sx_patch": np.zeros((batch_size, max_tr), dtype=np.float32),
        "sy_patch": np.zeros((batch_size, max_tr), dtype=np.float32),
        "rx_patch": np.zeros((batch_size, max_tr), dtype=np.float32),
        "ry_patch": np.zeros((batch_size, max_tr), dtype=np.float32),
        "trace_mask": np.zeros((batch_size, max_tr), dtype=np.float32),
        "loss_mask": np.zeros((batch_size, max_tr), dtype=np.float32),
        "valid_mask": np.zeros((batch_size, max_tr), dtype=np.float32),
    }

    for b, sample in enumerate(samples):
        n_tr = int(sample["data"].shape[0])
        out["data"][b, :n_tr] = _as_array(sample, "data")
        out["masked_patch"][b, :n_tr] = _as_array(sample, "masked_patch")
        for key in ("sx_patch", "sy_patch", "rx_patch", "ry_patch"):
            out[key][b, :n_tr] = _as_array(sample, key)
        if "trace_mask" in sample:
            out["trace_mask"][b, :n_tr] = _as_array(sample, "trace_mask")
        else:
            out["trace_mask"][b, :n_tr] = (~np.asarray(sample["is_query"], dtype=bool)).astype(np.float32)
        if "loss_mask" in sample:
            out["loss_mask"][b, :n_tr] = _as_array(sample, "loss_mask")
        else:
            out["loss_mask"][b, :n_tr] = np.asarray(sample["is_query"], dtype=np.float32)
        out["valid_mask"][b, :n_tr] = 1.0

    return {key: torch.from_numpy(value) for key, value in out.items()}


def coords01_from_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    coords = torch.stack(
        [
            batch["sx_patch"],
            batch["sy_patch"],
            batch["rx_patch"],
            batch["ry_patch"],
        ],
        dim=-1,
    ).float().to(device)
    return (0.5 * (coords + 1.0)).clamp(0.0, 1.0)


def repeat_trace_mask_for_chunks(mask: torch.Tensor, n_chunks: int) -> torch.Tensor:
    return mask.float().repeat(1, int(n_chunks))


def _weighted_token_mean(
    per_token_value: torch.Tensor,
    token_weight: torch.Tensor,
) -> torch.Tensor:
    """Helper: compute weighted mean over tokens.
    per_token_value: (B, L) or (B, L, T); token_weight: (B, L)
    """
    token_weight = token_weight.float()
    if per_token_value.dim() == 3:
        per_token_value = per_token_value.float() * token_weight.unsqueeze(-1)
        return per_token_value.sum() / (token_weight.sum() * per_token_value.shape[-1]).clamp(min=1.0)
    per_token_value = per_token_value.float() * token_weight
    return per_token_value.sum() / token_weight.sum().clamp(min=1.0)


def weighted_time_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    token_weight: torch.Tensor,
) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    sq = (pred - target) ** 2
    per_token_mse = sq.mean(dim=-1)
    return _weighted_token_mean(per_token_mse, token_weight)


def weighted_energy_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    token_weight: torch.Tensor,
) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    pred_energy = torch.mean(pred ** 2, dim=-1)
    target_energy = torch.mean(target ** 2, dim=-1)
    sq = (pred_energy - target_energy) ** 2
    return _weighted_token_mean(sq, token_weight)


def weighted_time_gradient_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    token_weight: torch.Tensor,
) -> torch.Tensor:
    if pred.shape[-1] <= 1:
        return pred.new_tensor(0.0, dtype=torch.float32)
    pred = pred.float()
    target = target.float()
    grad_pred = pred[..., 1:] - pred[..., :-1]
    grad_target = target[..., 1:] - target[..., :-1]
    sq = (grad_pred - grad_target) ** 2
    per_token_grad_mse = sq.mean(dim=-1)
    return _weighted_token_mean(per_token_grad_mse, token_weight)


def weighted_normalized_correlation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    token_weight: torch.Tensor,
) -> torch.Tensor:
    """逐道归一化互相关相位约束。

    对每道独立计算 pred 与 target 的 Pearson 相关系数，
    loss = 1 - corr，范围 [0, 2]。
    - corr=1（波形一致）→ loss=0
    - corr=0（不相关）  → loss=1
    - corr=-1（极性反）→ loss=2

    逐道计算，不依赖道间连续性，适用于 4D 随机采样 patch。
    MSE 约束振幅精度，互相关约束波形形状/相位对齐。
    """
    pred = pred.float()
    target = target.float()
    token_weight = token_weight.float()

    # 去直流
    pred_c = pred - pred.mean(dim=-1, keepdim=True)
    target_c = target - target.mean(dim=-1, keepdim=True)

    cov = (pred_c * target_c).sum(dim=-1)       # (B, L)
    var_pred = (pred_c ** 2).sum(dim=-1)         # (B, L)
    var_target = (target_c ** 2).sum(dim=-1)

    # 振幅接近 0 的道 → 相关系数无定义 → 设 corr=1 不惩罚
    valid = (var_pred > 1e-6) & (var_target > 1e-6)
    corr = torch.where(
        valid,
        cov / (var_pred * var_target).clamp(min=1e-6).sqrt(),
        torch.ones_like(cov),
    )
    per_token_loss = 1.0 - corr  # (B, L), [0, 2]
    return _weighted_token_mean(per_token_loss, token_weight)


def forward_loss(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, Dict[str, float]]:
    x_obs = batch["masked_patch"].float().to(device)
    x_gt = batch["data"].float().to(device)
    coords = coords01_from_batch(batch, device)
    trace_mask = batch["trace_mask"].float().to(device) * batch["valid_mask"].float().to(device)
    loss_mask = batch["loss_mask"].float().to(device) * batch["valid_mask"].float().to(device)
    valid_mask = batch["valid_mask"].float().to(device)

    x_chunk, coords_chunk, time_bounds, chunk_info = trace_time_chunk(
        x_obs, coords, args.chunk_length, args.overlap_ratio
    )
    target_chunk, _, _, _ = trace_time_chunk(
        x_gt, coords, args.chunk_length, args.overlap_ratio
    )

    n_chunks = int(chunk_info["n_chunks"])
    observed_token = repeat_trace_mask_for_chunks(trace_mask, n_chunks)
    query_token = repeat_trace_mask_for_chunks(loss_mask, n_chunks)
    valid_token = repeat_trace_mask_for_chunks(valid_mask, n_chunks)
    token_weight = (
        args.query_loss_weight * query_token
        + args.context_loss_weight * observed_token
    ) * valid_token
    if float(token_weight.detach().sum().item()) <= 0.0:
        token_weight = valid_token

    time_bounds = time_bounds.to(device).float() / max(float(chunk_info["time_length"]), 1.0)
    pred = model(x_chunk, coords_chunk, time_bounds, mask=observed_token, valid_mask=valid_token)
    mse_loss = weighted_time_mse(pred, target_chunk, token_weight)
    energy_loss = weighted_energy_mse(pred, target_chunk, token_weight)
    grad_loss = weighted_time_gradient_mse(pred, target_chunk, token_weight)
    phase_loss = weighted_normalized_correlation_loss(pred, target_chunk, token_weight)
    weighted_energy = float(args.energy_loss_weight) * energy_loss
    weighted_grad = float(args.hf_grad_loss_weight) * grad_loss
    weighted_phase = float(args.phase_loss_weight) * phase_loss
    loss = mse_loss + weighted_energy + weighted_grad + weighted_phase
    metrics = {
        "loss_mse": float(mse_loss.detach().item()),
        "loss_energy": float(weighted_energy.detach().item()),
        "loss_grad": float(weighted_grad.detach().item()),
        "loss_phase": float(weighted_phase.detach().item()),
        "loss_phase_raw": float(phase_loss.detach().item()),
        "loss_energy_raw": float(energy_loss.detach().item()),
        "loss_grad_raw": float(grad_loss.detach().item()),
        "query_tokens": float(query_token.sum().detach().item()),
        "context_tokens": float(observed_token.sum().detach().item()),
        "valid_tokens": float(valid_token.sum().detach().item()),
    }
    return loss, metrics


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    return create_gated_model_v9_encdec(
        input_dim=args.chunk_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        output_dim=args.chunk_length,
        headwise_attn_output_gate=args.headwise_attn_output_gate,
        elementwise_attn_output_gate=args.elementwise_attn_output_gate,
        use_qk_norm=args.use_qk_norm,
        qkv_bias=args.qkv_bias,
        rms_norm_eps=args.rms_norm_eps,
        hidden_act=args.hidden_act,
        use_coord_encoding=args.use_coord_encoding,
        use_rope=args.use_rope,
        coord_dim=args.coord_dim,
        num_attn_res_blocks=args.num_attn_res_blocks,
        encode_observed_only=args.encode_observed_only,
        rope_omega_bands=getattr(args, "rope_omega_bands", None),
    )


def prepare_rope_frequency(args: argparse.Namespace, dataset) -> Dict[str, Any]:
    head_dim = int(args.d_model) // int(args.n_heads)
    part_dim = head_dim // int(args.coord_dim)
    n_freqs = part_dim // 2
    cfg = build_rope_frequency_config(
        coord_stats=getattr(dataset, "coord_stats", {}),
        coord_dim=int(args.coord_dim),
        n_freqs=n_freqs,
        mode=args.rope_freq_mode,
        lambda_phys_x=args.lambda_phys_x,
        lambda_phys_y=args.lambda_phys_y,
        nyquist_safety=args.rope_nyquist_safety,
    )
    args.rope_frequency_config = cfg
    args.rope_omega_bands = cfg.get("omega_bands")
    return cfg


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        results_folder: str,
        dl: DataLoader,
        val_dl: DataLoader,
        args: argparse.Namespace,
        accelerator: Accelerator,
    ):
        self.model = model
        self.args = args
        self.accelerator = accelerator
        self.device = accelerator.device
        self.results_folder = Path(results_folder)
        self.ckp_folder = self.results_folder / "checkpoints"
        self.log_folder = self.results_folder / "logs"

        if accelerator.is_main_process:
            self.ckp_folder.mkdir(exist_ok=True, parents=True)
            self.log_folder.mkdir(exist_ok=True, parents=True)
            self.writer = SummaryWriter(log_dir=str(self.log_folder))
            self.log_file = open(self.log_folder / "training_log.txt", "a", encoding="utf-8")
        else:
            self.writer = None
            self.log_file = None

        self.dl = dl
        self.val_dl = val_dl
        self.opt = AdamW(
            [{"params": model.parameters(), "lr": args.lr}],
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=1e-4,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt,
            T_max=max(1, args.epochs - args.warmup_epochs),
            eta_min=min(args.lr, 5e-5),
        )

        self.model, self.opt, self.dl, self.val_dl = accelerator.prepare(
            self.model, self.opt, self.dl, self.val_dl
        )

        if accelerator.is_main_process:
            self._save_training_config()
            self._log_setup_info()
            self._log_first_batch_stats()

    def _log(self, msg: str) -> None:
        if self.accelerator.is_main_process and self.log_file is not None:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_file.write(f"[{ts}] {msg}\n")
            self.log_file.flush()
            print(msg)

    def _log_setup_info(self) -> None:
        """记录训练环境、数据集、模型和优化器的完整初始化信息。"""
        args = self.args
        ds = self.dl.dataset
        world_size = self.accelerator.num_processes
        device = self.accelerator.device

        # 1. 训练环境
        self._log("=" * 60)
        self._log("Training Setup")
        self._log("=" * 60)
        self._log(f"  Timestamp:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"  World size:      {world_size} GPU(s)")
        self._log(f"  Device:          {device}")
        self._log(f"  Mixed precision: {args.mixed_precision}")
        self._log(f"  Seed:            {args.seed}")

        # 2. 数据集统计
        total_samples = len(ds)
        epoch_repeat = getattr(ds, "epoch_repeat", 1)
        actual_train_samples = total_samples
        if getattr(ds, "num_anchors", None) is not None:
            actual_train_samples = f"{ds.num_anchors} anchors x {epoch_repeat} repeats = {total_samples}"

        self._log("-" * 40)
        self._log("Dataset")
        self._log("-" * 40)
        self._log(f"  Class:           {type(ds).__name__}")
        self._log(f"  Total samples:   {total_samples}")
        if getattr(ds, "num_anchors", None) is not None:
            self._log(f"  Anchor samples:  {ds.num_anchors}")
            self._log(f"  Epoch repeat:    {epoch_repeat}")
        self._log(f"  Effective train: {actual_train_samples}")
        self._log(f"  time_ps:         {ds.time_ps}")
        self._log(f"  trace_ps:        {ds.trace_ps}")
        self._log(f"  train_num_query: {args.train_num_query}")
        self._log(f"  patch_beta:      {args.patch_beta}")
        self._log(f"  target_mode:     {args.target_mode}")
        self._log(f"  trace_sort_keys: {getattr(args, 'trace_sort_keys', 'default')}")
        self._log(f"  h5File:          {args.h5File}")
        self._log(f"  h5File_regular:  {args.h5File_regular}")
        self._log(f"  dataset_neighbors: {args.dataset_neighbors_train}")

        # 3. 数据加载器
        sampler_name = type(self.dl.sampler).__name__ if self.dl.sampler else "None"
        val_sampler_name = type(self.val_dl.sampler).__name__ if self.val_dl.sampler else "None"
        self._log("-" * 40)
        self._log("DataLoader")
        self._log("-" * 40)
        self._log(f"  Batch size (per GPU):   {args.batch_size}")
        self._log(f"  Global batch size:      {args.batch_size * world_size}")
        self._log(f"  Train steps/epoch:      {len(self.dl)}")
        self._log(f"  Val steps (max):        {len(self.val_dl)}")
        self._log(f"  Train sampler:          {sampler_name}")
        self._log(f"  Val sampler:            {val_sampler_name}")
        self._log(f"  Num workers:            {args.num_workers}")
        self._log(f"  Pin memory:             {torch.cuda.is_available()}")
        self._log(f"  Chunk length:           {args.chunk_length}")
        self._log(f"  Overlap ratio:          {args.overlap_ratio}")

        # 4. 模型配置
        n_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        head_dim = args.d_model // args.n_heads
        part_dim = head_dim // args.coord_dim if args.coord_dim > 0 else head_dim
        self._log("-" * 40)
        self._log("Model Config")
        self._log("-" * 40)
        self._log(f"  Architecture:    GatedSeismicInterpolationTransformerV9EncDec")
        self._log(f"  d_model:         {args.d_model}")
        self._log(f"  n_heads:         {args.n_heads}  (head_dim={head_dim})")
        self._log(f"  num_enc_layers:  {args.num_encoder_layers or args.num_layers}")
        self._log(f"  num_dec_layers:  {args.num_decoder_layers or args.num_layers}")
        self._log(f"  d_ff:            {args.d_ff}")
        self._log(f"  dropout:         {args.dropout}")
        self._log(f"  coord_dim:       {args.coord_dim}  (part_dim={part_dim})")
        self._log(f"  use_rope:        {args.use_rope}")
        self._log(f"  use_coord_enc:   {args.use_coord_encoding}")
        self._log(f"  elementwise_gate:{args.elementwise_attn_output_gate}")
        self._log(f"  use_qk_norm:     {args.use_qk_norm}")
        self._log(f"  encode_obs_only: {args.encode_observed_only}")
        self._log(f"  Total params:    {n_params:,}")
        self._log(f"  Trainable params:{trainable_params:,}")

        # 5. RoPE 配置
        rope_cfg = getattr(args, "rope_frequency_config", None)
        if rope_cfg:
            self._log("-" * 40)
            self._log("RoPE Frequency")
            self._log("-" * 40)
            self._log(f"  Mode:            {rope_cfg.get('rope_freq_mode', 'default')}")
            self._log(f"  n_freqs:         {rope_cfg.get('n_freqs', 'N/A')}")
            omega = rope_cfg.get("omega", {})
            self._log(f"  omega_x:         {omega.get('x', 'N/A')}")
            self._log(f"  omega_y:         {omega.get('y', 'N/A')}")
            if rope_cfg.get("warnings"):
                for w in rope_cfg["warnings"]:
                    self._log(f"  [WARN] {w}")

        # 6. 优化器与损失
        self._log("-" * 40)
        self._log("Optimizer & Loss")
        self._log("-" * 40)
        self._log(f"  Optimizer:       AdamW")
        self._log(f"  LR:              {args.lr:.3e}")
        self._log(f"  Betas:           (0.9, 0.95)")
        self._log(f"  Weight decay:    1e-4")
        self._log(f"  Warmup epochs:   {args.warmup_epochs}")
        self._log(f"  Cosine T_max:    {max(1, args.epochs - args.warmup_epochs)}")
        self._log(f"  Grad clip norm:  1.0")
        self._log(f"  Accum steps:     {args.accumulation_steps}")
        self._log(f"  query_loss_w:    {args.query_loss_weight}")
        self._log(f"  context_loss_w:  {args.context_loss_weight}")
        self._log(f"  energy_loss_w:   {args.energy_loss_weight}")
        self._log(f"  grad_loss_w:     {args.hf_grad_loss_weight}")
        self._log("=" * 60)

    @torch.no_grad()
    def _log_first_batch_stats(self) -> None:
        """采样第一个 batch，记录 trace 数、query/context 比例等实际数据分布。"""
        try:
            batch = next(iter(self.dl))
        except StopIteration:
            return

        valid_mask = batch["valid_mask"].float()
        trace_mask = batch["trace_mask"].float() * valid_mask
        loss_mask = batch["loss_mask"].float() * valid_mask

        n_samples = int(valid_mask.shape[0])
        traces_per_sample = [int(valid_mask[i].sum().item()) for i in range(n_samples)]
        queries_per_sample = [int(loss_mask[i].sum().item()) for i in range(n_samples)]
        contexts_per_sample = [int(trace_mask[i].sum().item()) for i in range(n_samples)]

        avg_tr = sum(traces_per_sample) / max(len(traces_per_sample), 1)
        avg_q = sum(queries_per_sample) / max(len(queries_per_sample), 1)
        avg_c = sum(contexts_per_sample) / max(len(contexts_per_sample), 1)

        self._log("First Batch Statistics")
        self._log("-" * 40)
        self._log(f"  Batch samples:   {n_samples}")
        self._log(f"  Avg traces:      {avg_tr:.1f}")
        self._log(f"  Avg queries:     {avg_q:.1f}")
        self._log(f"  Avg contexts:    {avg_c:.1f}")
        self._log(f"  Query ratio:     {avg_q/max(avg_tr,1)*100:.1f}%")
        self._log(f"  Context ratio:   {avg_c/max(avg_tr,1)*100:.1f}%")
        self._log(f"  Trace shape:     {batch['data'].shape}")
        self._log(f"  Time samples:    {batch['data'].shape[-1]}")
        self._log("=" * 60)

    def _save_training_config(self) -> None:
        ds = self.dl.dataset
        world_size = self.accelerator.num_processes
        cfg = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "environment": {
                "world_size": world_size,
                "device": str(self.accelerator.device),
                "mixed_precision": self.args.mixed_precision,
                "seed": self.args.seed,
            },
            "training_args": vars(self.args),
            "model": {
                "class": "GatedSeismicInterpolationTransformerV9EncDec",
                "model_type": "e2e_encdec_v9",
                "input_dim": self.args.chunk_length,
                "d_model": self.args.d_model,
                "n_heads": self.args.n_heads,
                "num_layers": self.args.num_layers,
                "num_encoder_layers": self.args.num_encoder_layers,
                "num_decoder_layers": self.args.num_decoder_layers,
                "d_ff": self.args.d_ff,
                "dropout": self.args.dropout,
                "encode_observed_only": self.args.encode_observed_only,
                "total_params": sum(p.numel() for p in self.model.parameters()),
                "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            },
            "dataloader": {
                "batch_size_per_gpu": self.args.batch_size,
                "global_batch_size": self.args.batch_size * world_size,
                "train_steps_per_epoch": len(self.dl),
                "val_steps": len(self.val_dl),
                "num_workers": self.args.num_workers,
                "pin_memory": torch.cuda.is_available(),
                "chunk_length": self.args.chunk_length,
                "overlap_ratio": self.args.overlap_ratio,
            },
            "dataset": {
                "class": type(ds).__name__,
                "total_samples": len(ds),
                "num_anchors": getattr(ds, "num_anchors", None),
                "epoch_repeat": getattr(ds, "epoch_repeat", 1),
                "target_mode": self.args.target_mode,
                "h5File": self.args.h5File,
                "h5File_regular": self.args.h5File_regular,
                "h5File_tgt": self.args.h5File_tgt,
                "dataset_neighbors_train": self.args.dataset_neighbors_train,
                "time_ps": ds.time_ps,
                "trace_ps": ds.trace_ps,
                "train_num_query": self.args.train_num_query,
                "patch_beta": self.args.patch_beta,
                "use_p_scale": self.args.use_p_scale,
                "trace_sort_keys": getattr(self.args, "trace_sort_keys", None),
                "coord_stats": getattr(ds, "coord_stats", None),
            },
            "loss_weights": {
                "query": self.args.query_loss_weight,
                "context": self.args.context_loss_weight,
                "energy": self.args.energy_loss_weight,
                "grad": self.args.hf_grad_loss_weight,
                "phase": self.args.phase_loss_weight,
            },
            "rope_frequency_config": getattr(self.args, "rope_frequency_config", None),
        }
        with open(self.log_folder / "training_config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False,
                      default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))

    def save(self, epoch: int, train_loss: float, val_loss: float | None) -> None:
        if not self.accelerator.is_main_process:
            return
        unwrapped = self.accelerator.unwrap_model(self.model)
        path = self.ckp_folder / f"model-{epoch}.pth"
        torch.save(
            {
                "model_state_dict": unwrapped.state_dict(),
                "optimizer_state_dict": self.opt.state_dict(),
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": None if val_loss is None else float(val_loss),
                "config": vars(self.args),
            },
            path,
        )
        self._log(f"Saved checkpoint: {path}")

    @staticmethod
    def _avg_metrics(metric_sums: Dict[str, float], count: int) -> Dict[str, float]:
        if count <= 0:
            return {}
        return {key: value / float(count) for key, value in metric_sums.items()}

    @torch.no_grad()
    def visualize(self, epoch: int, max_samples: int = 4) -> None:
        self.model.eval()

        if self.accelerator.is_main_process:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            vis_dir = self.results_folder / "vis" / f"epoch_{epoch:04d}"
            vis_dir.mkdir(parents=True, exist_ok=True)
            self._log(f"Saving visualizations to {vis_dir}")

        saved = 0
        for batch in self.val_dl:
            if saved >= max_samples:
                break

            x_obs = batch["masked_patch"].float().to(self.device)
            x_gt = batch["data"].float().to(self.device)
            coords = coords01_from_batch(batch, self.device)
            trace_mask = batch["trace_mask"].float().to(self.device) * batch["valid_mask"].float().to(self.device)
            valid_mask = batch["valid_mask"].float().to(self.device)

            x_chunk, coords_chunk, time_bounds, chunk_info = trace_time_chunk(
                x_obs, coords, self.args.chunk_length, self.args.overlap_ratio
            )
            n_chunks = int(chunk_info["n_chunks"])
            observed_token = repeat_trace_mask_for_chunks(trace_mask, n_chunks)
            valid_token = repeat_trace_mask_for_chunks(valid_mask, n_chunks)
            time_bounds_norm = time_bounds.to(self.device).float() / max(float(chunk_info["time_length"]), 1.0)

            with self.accelerator.autocast():
                pred_chunk = self.model(x_chunk, coords_chunk, time_bounds_norm,
                                         mask=observed_token, valid_mask=valid_token)
            pred = trace_time_unchunk(pred_chunk.float(), chunk_info).cpu().numpy()
            target = x_gt.cpu().numpy()
            inp = x_obs.cpu().numpy()

            if self.accelerator.is_main_process:
                for b in range(int(min(batch["data"].shape[0], 1))):
                    n_tr = int(valid_mask[b].sum().item())
                    if n_tr < 2:
                        continue
                    inp_b = inp[b, :n_tr]
                    pred_b = pred[b, :n_tr]
                    tgt_b = target[b, :n_tr]
                    resid_b = pred_b - tgt_b

                    vmax = max(
                        abs(inp_b).max(), abs(pred_b).max(),
                        abs(tgt_b).max(), abs(resid_b).max(), 1e-6,
                    )
                    fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)
                    for ax, img, title in zip(
                        axes,
                        [inp_b, pred_b, tgt_b, resid_b],
                        ["Masked Input", "Prediction", "Target", "Residual"],
                    ):
                        ax.imshow(img.T, aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax)
                        ax.set_title(title)
                        ax.set_xlabel("Trace")
                        ax.set_ylabel("Time Sample")
                    fig.savefig(str(vis_dir / f"sample_{saved}.png"), dpi=120, bbox_inches="tight")
                    plt.close(fig)

            saved += 1

        self.model.train()

    def evaluate(self, max_batches: int = 50) -> tuple[float, Dict[str, float]]:
        self.model.eval()
        losses = []
        metric_sums: Dict[str, float] = {}
        for i, batch in enumerate(self.val_dl):
            if i >= max_batches:
                break
            with self.accelerator.autocast():
                loss, metrics = forward_loss(self.model, batch, self.args, self.device)
            losses.append(float(loss.detach().item()))
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
        self.model.train()
        avg_loss = float(sum(losses) / len(losses)) if losses else float("nan")
        return avg_loss, self._avg_metrics(metric_sums, len(losses))

    def train(self) -> None:
        for epoch in tqdm(
            range(self.args.epochs),
            total=self.args.epochs,
            desc="Training E2E V9",
            disable=not self.accelerator.is_main_process,
        ):
            if hasattr(self.dl.sampler, "set_epoch"):
                self.dl.sampler.set_epoch(epoch)

            self.model.train()
            self.opt.zero_grad(set_to_none=True)
            losses = []
            metric_sums: Dict[str, float] = {}
            for step, batch in enumerate(self.dl):
                with self.accelerator.autocast():
                    loss, metrics = forward_loss(self.model, batch, self.args, self.device)
                    scaled_loss = loss / max(1, self.args.accumulation_steps)
                self.accelerator.backward(scaled_loss)

                do_step = ((step + 1) % self.args.accumulation_steps == 0) or (step + 1 == len(self.dl))
                if do_step:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.opt.step()
                    self.opt.zero_grad(set_to_none=True)
                losses.append(float(loss.detach().item()))
                for key, value in metrics.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(value)

            if epoch < self.args.warmup_epochs:
                lr = self.args.lr * float(epoch + 1) / max(1, self.args.warmup_epochs)
                for group in self.opt.param_groups:
                    group["lr"] = lr
            else:
                self.lr_scheduler.step()
                lr = self.opt.param_groups[0]["lr"]

            avg_loss = float(sum(losses) / len(losses)) if losses else float("nan")
            train_metrics = self._avg_metrics(metric_sums, len(losses))
            val_loss = None
            val_metrics = {}
            if epoch == 0 or epoch % 10 == 0:
                val_loss, val_metrics = self.evaluate()
            if epoch == 0 or epoch % 100 == 0:
                self.visualize(epoch + 1)
                self.accelerator.wait_for_everyone()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if self.accelerator.is_main_process:
                # Build detailed epoch log line
                epoch_log = (
                    f"Epoch {epoch + 1}/{self.args.epochs} "
                    f"train={avg_loss:.6f} "
                    f"val={val_loss if val_loss is not None else 'N/A'} "
                    f"lr={lr:.3e} "
                    f"mse={train_metrics.get('loss_mse', float('nan')):.6f} "
                    f"energy={train_metrics.get('loss_energy', float('nan')):.6f} "
                    f"grad={train_metrics.get('loss_grad', float('nan')):.6f}"
                    f" phase={train_metrics.get('loss_phase', float('nan')):.6f}"
                )
                # Add token statistics if available
                qtok = train_metrics.get('query_tokens')
                ctok = train_metrics.get('context_tokens')
                vtok = train_metrics.get('valid_tokens')
                if qtok is not None and vtok is not None and vtok > 0:
                    epoch_log += f" q_ratio={qtok/vtok*100:.1f}%"
                if ctok is not None and vtok is not None and vtok > 0:
                    epoch_log += f" c_ratio={ctok/vtok*100:.1f}%"
                self._log(epoch_log)

                self.writer.add_scalar("Loss/train", avg_loss, epoch)
                self.writer.add_scalar("LR", lr, epoch)
                for key in ("loss_mse", "loss_energy", "loss_grad", "loss_phase",
                            "loss_energy_raw", "loss_grad_raw", "loss_phase_raw"):
                    if key in train_metrics:
                        self.writer.add_scalar(f"Loss/train_{key}", train_metrics[key], epoch)
                if val_loss is not None:
                    self.writer.add_scalar("Loss/val", val_loss, epoch)
                    for key in ("loss_mse", "loss_energy", "loss_grad", "loss_phase",
                                "loss_energy_raw", "loss_grad_raw", "loss_phase_raw"):
                        if key in val_metrics:
                            self.writer.add_scalar(f"Loss/val_{key}", val_metrics[key], epoch)

            if self.accelerator.is_main_process and (
                epoch == 0 or (epoch + 1) % self.args.save_every == 0
            ):
                self.save(epoch + 1, avg_loss, val_loss)
            self.accelerator.wait_for_everyone()

    def __del__(self):
        if hasattr(self, "log_file") and self.log_file is not None:
            self.log_file.close()


def main() -> None:
    args = parse_args()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    if rank == 0:
        print_segy_config()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    trace_sort_keys = tuple(k for k in str(args.trace_sort_keys).split(",") if k)

    if rank != 0:
        real_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
    try:
        dataset = DatasetH5_all_queryctx(
            h5File=args.h5File,
            h5File_regular=args.h5File_regular,
            h5File_tgt=args.h5File_tgt,
            dataset_neighbors=args.dataset_neighbors_train,
            train=True,
            train_num_query=args.train_num_query,
            train_context_size=args.train_context_size,
            patch_beta=args.patch_beta,
            force_anchor_query=args.force_anchor_query,
            trace_sort_keys=trace_sort_keys,
            use_p_scale=args.use_p_scale,
            time_ps=args.time_ps,
            trace_ps=args.trace_ps,
            epoch_repeat=args.epoch_repeat,
            target_mode=args.target_mode,
            coord_aug_scale=args.coord_aug_scale,
        )
        val_dataset = None
        if args.dataset_neighbors_test is not None:
            val_dataset = DatasetH5_all_queryctx(
                h5File=args.h5File,
                h5File_regular=args.h5File_regular,
                h5File_tgt=args.h5File_tgt,
                dataset_neighbors=args.dataset_neighbors_test,
                train=False,
                trace_sort_keys=trace_sort_keys,
                use_p_scale=args.use_p_scale,
                time_ps=args.time_ps,
                trace_ps=args.trace_ps,
                target_mode="self",
            )
    finally:
        if rank != 0:
            sys.stdout.close()
            sys.stdout = real_stdout

    if rank == 0:
        sample0 = dataset[0]
        print(
            f"[e2e_v9] target_mode={args.target_mode} "
            f"time_ps={dataset.time_ps} trace_ps={dataset.trace_ps} "
            f"sample_shape={sample0['data'].shape} samples={len(dataset)}"
        )

    rope_cfg = prepare_rope_frequency(args, dataset)
    if rank == 0:
        print(
            f"[e2e_v9] rope_freq_mode={rope_cfg['rope_freq_mode']} "
            f"n_freqs={rope_cfg['n_freqs']} "
            f"warnings={len(rope_cfg.get('warnings', []))}"
        )
        for warning in rope_cfg.get("warnings", []):
            print(f"[e2e_v9][rope] {warning}")

    train_sampler = DistributedSampler(dataset) if world_size > 1 else None
    val_ds = val_dataset if val_dataset is not None else dataset
    val_sampler = DistributedSampler(val_ds, shuffle=False) if world_size > 1 else None
    dl = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        sampler=train_sampler,
        collate_fn=collate_queryctx,
        pin_memory=torch.cuda.is_available(),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, min(args.num_workers, 2)),
        sampler=val_sampler,
        collate_fn=collate_queryctx,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(args).to(accelerator.device)
    if rank == 0:
        print(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    result_dir = Path(args.results_dir) / f"{args.model_name}_datatype_{args.data_type}_queryctx"
    if accelerator.is_main_process:
        result_dir.mkdir(exist_ok=True, parents=True)

    trainer = Trainer(
        model=model,
        results_folder=str(result_dir),
        dl=dl,
        val_dl=val_dl,
        args=args,
        accelerator=accelerator,
    )
    trainer.train()


if __name__ == "__main__":
    main()
