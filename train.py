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
from model import create_gated_model_v9_encdec, trace_time_chunk
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
    sq = (pred - target_chunk) ** 2
    denom = (token_weight.sum() * pred.shape[-1]).clamp(min=1.0)
    loss = (sq * token_weight.unsqueeze(-1)).sum() / denom
    metrics = {
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
            self._log("Model: V9 EncDec E2E")
            self._log(f"Train dataset: {len(self.dl.dataset)} samples")
            self._log(f"Steps/epoch:   {len(self.dl)}")
            self._log(f"Batch size:    {args.batch_size}")
            self._log(f"Learning rate: {args.lr}")

    def _log(self, msg: str) -> None:
        if self.accelerator.is_main_process and self.log_file is not None:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_file.write(f"[{ts}] {msg}\n")
            self.log_file.flush()
            print(msg)

    def _save_training_config(self) -> None:
        ds = self.dl.dataset
        cfg = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
                "total_params": sum(p.numel() for p in self.model.parameters()),
                "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            },
            "dataset": {
                "class": "DatasetH5_all_queryctx",
                "target_mode": self.args.target_mode,
                "h5File": self.args.h5File,
                "h5File_regular": self.args.h5File_regular,
                "h5File_tgt": self.args.h5File_tgt,
                "dataset_neighbors_train": self.args.dataset_neighbors_train,
                "time_ps": self.args.time_ps,
                "trace_ps": self.args.trace_ps,
                "train_num_query": self.args.train_num_query,
                "patch_beta": self.args.patch_beta,
                "use_p_scale": self.args.use_p_scale,
                "coord_stats": getattr(ds, "coord_stats", None),
            },
            "rope_frequency_config": getattr(self.args, "rope_frequency_config", None),
        }
        with open(self.log_folder / "training_config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False, default=str)

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

    @torch.no_grad()
    def evaluate(self, max_batches: int = 50) -> float:
        self.model.eval()
        losses = []
        for i, batch in enumerate(self.val_dl):
            if i >= max_batches:
                break
            with self.accelerator.autocast():
                loss, _ = forward_loss(self.model, batch, self.args, self.device)
            losses.append(float(loss.detach().item()))
        self.model.train()
        return float(sum(losses) / len(losses)) if losses else float("nan")

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
            for step, batch in enumerate(self.dl):
                with self.accelerator.autocast():
                    loss, _ = forward_loss(self.model, batch, self.args, self.device)
                    scaled_loss = loss / max(1, self.args.accumulation_steps)
                self.accelerator.backward(scaled_loss)

                do_step = ((step + 1) % self.args.accumulation_steps == 0) or (step + 1 == len(self.dl))
                if do_step:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.opt.step()
                    self.opt.zero_grad(set_to_none=True)
                losses.append(float(loss.detach().item()))

            if epoch < self.args.warmup_epochs:
                lr = self.args.lr * float(epoch + 1) / max(1, self.args.warmup_epochs)
                for group in self.opt.param_groups:
                    group["lr"] = lr
            else:
                self.lr_scheduler.step()
                lr = self.opt.param_groups[0]["lr"]

            avg_loss = float(sum(losses) / len(losses)) if losses else float("nan")
            val_loss = self.evaluate() if (epoch == 0 or epoch % 10 == 0) else None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if self.accelerator.is_main_process:
                self._log(
                    f"Epoch {epoch + 1}/{self.args.epochs} "
                    f"train={avg_loss:.6f} "
                    f"val={val_loss if val_loss is not None else 'N/A'} "
                    f"lr={lr:.3e}"
                )
                self.writer.add_scalar("Loss/train", avg_loss, epoch)
                self.writer.add_scalar("LR", lr, epoch)
                if val_loss is not None:
                    self.writer.add_scalar("Loss/val", val_loss, epoch)

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
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
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
    val_sampler = DistributedSampler(dataset, shuffle=False) if world_size > 1 else None
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
        dataset,
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
