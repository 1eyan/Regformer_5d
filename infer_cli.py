#!/usr/bin/env python3
"""E2E V9 inference CLI for query-context seismic interpolation and SEGY fill."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist

from config.segy_config import (
    get_key_columns,
    load_config as load_segy_config,
    print_info as print_segy_config,
)
from dataset import DatasetH5_all_queryctx
from infer import fit_trace, run_queryctx_inference
from model import create_gated_model_v9_encdec
from utils.coord_utils import build_rope_frequency_config
from utils import (
    build_lookup,
    read_segy_data,
    read_segy_headers,
    sort_output_segy,
    write_segy_data,
    write_segy_data_incremental,
)


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def optional_float(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null", "auto"}:
        return None
    return float(value)


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("e2e_v9_infer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "infer.log", encoding="utf-8"),
    ):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def setup_ddp() -> Tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def load_checkpoint(model: torch.nn.Module, path: str, strict: bool, logger: logging.Logger) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt.get("model", ckpt))) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("module.") for k in state):
        state = OrderedDict((k.replace("module.", "", 1), v) for k, v in state.items())
    result = model.load_state_dict(state, strict=strict)
    logger.info("checkpoint loaded: missing=%s unexpected=%s", result.missing_keys, result.unexpected_keys)


def load_training_config(checkpoint_path: str) -> Dict[str, Any]:
    ckpt = Path(checkpoint_path)
    for config_path in (
        ckpt.parent.parent / "logs" / "training_config.json",
        ckpt.parent / "logs" / "training_config.json",
    ):
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


def _first_of(*keys: str, cfg: dict, default=None):
    scopes = [
        cfg,
        cfg.get("training_args", {}),
        cfg.get("model", {}),
        cfg.get("dataset", {}),
        cfg.get("dataset_args", {}),
    ]
    for scope in scopes:
        for key in keys:
            if key in scope and scope[key] is not None:
                return scope[key]
    return default


def apply_training_config(args: argparse.Namespace, train_cfg: Dict[str, Any]) -> None:
    if not train_cfg:
        return
    int_keys = [
        "time_ps",
        "trace_ps",
        "chunk_length",
        "d_model",
        "n_heads",
        "num_layers",
        "d_ff",
        "coord_dim",
        "num_attn_res_blocks",
    ]
    float_keys = ["overlap_ratio", "dropout", "rms_norm_eps"]
    bool_keys = [
        "elementwise_attn_output_gate",
        "headwise_attn_output_gate",
        "use_qk_norm",
        "qkv_bias",
        "use_coord_encoding",
        "use_rope",
        "use_p_scale",
        "encode_observed_only",
    ]
    for key in int_keys:
        value = _first_of(key, cfg=train_cfg)
        if value is not None:
            setattr(args, key, int(value))
    for key in float_keys:
        value = _first_of(key, cfg=train_cfg)
        if value is not None:
            setattr(args, key, float(value))
    for key in ("lambda_phys_x", "lambda_phys_y", "rope_nyquist_safety"):
        value = _first_of(key, cfg=train_cfg)
        if value is not None:
            setattr(args, key, optional_float(value) if key.startswith("lambda_") else float(value))
    for key in bool_keys:
        value = _first_of(key, cfg=train_cfg)
        if value is not None:
            setattr(args, key, bool(value))

    for key in ("num_encoder_layers", "num_decoder_layers"):
        value = _first_of(key, cfg=train_cfg)
        setattr(args, key, None if value in (None, "None") else int(value))
    hidden_act = _first_of("hidden_act", cfg=train_cfg)
    if hidden_act is not None:
        args.hidden_act = str(hidden_act)
    rope_freq_mode = _first_of("rope_freq_mode", cfg=train_cfg)
    if rope_freq_mode is not None:
        args.rope_freq_mode = str(rope_freq_mode)
    else:
        use_phys_omega = _first_of("use_phys_omega", cfg=train_cfg)
        if use_phys_omega is not None and str2bool(use_phys_omega):
            args.rope_freq_mode = "physical"
    rope_cfg = _first_of("rope_frequency_config", cfg=train_cfg)
    if rope_cfg is not None:
        args.rope_frequency_config = rope_cfg
        args.rope_omega_bands = rope_cfg.get("omega_bands")

    # Restore trace_sort_keys from training config (critical for patch ordering consistency)
    trace_sort_keys = _first_of("trace_sort_keys", cfg=train_cfg)
    if trace_sort_keys is not None:
        if isinstance(trace_sort_keys, str):
            args.trace_sort_keys = tuple(k.strip() for k in trace_sort_keys.split(",") if k.strip())
        elif isinstance(trace_sort_keys, (list, tuple)):
            args.trace_sort_keys = tuple(str(k).strip() for k in trace_sort_keys if str(k).strip())
    else:
        args.trace_sort_keys = None


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


def prepare_rope_frequency(args: argparse.Namespace, dataset, logger: logging.Logger) -> Dict[str, Any]:
    cfg = getattr(args, "rope_frequency_config", None)
    if cfg and cfg.get("omega_bands") is not None:
        omega_bands = cfg["omega_bands"]
        # Check for string corruption from json.dump(default=str) on numpy arrays
        if isinstance(omega_bands, str):
            logger.warning(
                "omega_bands in checkpoint is a string (corrupted by JSON serialization). "
                "Recomputing from scratch ..."
            )
        else:
            args.rope_freq_mode = cfg.get("rope_freq_mode", args.rope_freq_mode)
            args.rope_omega_bands = omega_bands
            logger.info("using RoPE frequency config restored from checkpoint: mode=%s", args.rope_freq_mode)
            return cfg

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
    logger.info(
        "prepared RoPE frequency config: mode=%s n_freqs=%d warnings=%d",
        cfg["rope_freq_mode"],
        cfg["n_freqs"],
        len(cfg.get("warnings", [])),
    )
    for warning in cfg.get("warnings", []):
        logger.warning("RoPE frequency: %s", warning)
    return cfg


def save_reports(
    output_dir: Path,
    headers,
    written,
    unfilled,
    still_missing,
    observed_changed,
    unmatched,
    summary,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    header_by_idx = {int(h["trace_idx"]): h["key"] for h in headers}
    for name, indices in (
        ("filled_missing_keys.csv", written),
        ("unfilled_missing_keys.csv", unfilled),
        ("still_missing_after_write_keys.csv", still_missing),
        ("observed_changed_keys.csv", observed_changed),
    ):
        with open(output_dir / name, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["trace_idx", *get_key_columns()])
            for idx in indices:
                writer.writerow([idx, *header_by_idx.get(int(idx), ("", "", "", ""))])
    with open(output_dir / "unmatched_prediction_keys.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(get_key_columns())
        writer.writerows(unmatched)


def fill_segy(args, headers, missing_global, pred_sum, pred_count, logger, label_data=None) -> dict:
    mask_data = read_segy_data(args.mask_path)
    lookup = build_lookup(headers)
    out = mask_data.copy()
    ns = mask_data.shape[1]
    written, unmatched = set(), []

    for key, total in pred_sum.items():
        indices = lookup.get(key)
        if not indices:
            unmatched.append(key)
            continue
        trace = fit_trace(total / max(pred_count[key], 1), ns, time_ps=args.time_ps)
        wrote = False
        for trace_idx in indices:
            if missing_global[trace_idx]:
                out[trace_idx] = trace
                written.add(trace_idx)
                wrote = True
        if not wrote:
            unmatched.append(key)

    write_start = time.perf_counter()
    write_segy_data(args.mask_path, args.output_segy, out)
    writeback_seconds = time.perf_counter() - write_start

    written_sorted = sorted(written)
    missing_indices = set(np.flatnonzero(missing_global).tolist())
    unfilled = sorted(missing_indices - written)
    still_missing = np.flatnonzero(
        missing_global & np.all(np.abs(out) <= args.missing_eps, axis=1)
    ).tolist()
    observed_changed = np.flatnonzero(
        (~missing_global) & np.any(np.abs(out - mask_data) > args.missing_eps, axis=1)
    ).tolist()

    residual_stats = {}
    if label_data is not None:
        residual = np.zeros_like(out)
        for trace_idx in written_sorted:
            residual[trace_idx] = out[trace_idx] - label_data[trace_idx]
        write_segy_data(args.mask_path, args.output_residual_segy, residual)
        filled_residuals = residual[written_sorted]
        residual_stats = {
            "residual_max_abs": float(np.max(np.abs(filled_residuals))) if filled_residuals.size else 0.0,
            "residual_mean_abs": float(np.mean(np.abs(filled_residuals))) if filled_residuals.size else 0.0,
            "output_residual_segy": args.output_residual_segy,
        }

    summary = {
        "key_columns": list(get_key_columns()),
        "segy_traces": int(mask_data.shape[0]),
        "segy_samples": int(mask_data.shape[1]),
        "missing_total": int(len(missing_indices)),
        "written": int(len(written_sorted)),
        "unfilled": int(len(unfilled)),
        "still_missing_after_write": int(len(still_missing)),
        "observed_changed": int(len(observed_changed)),
        "prediction_keys": int(len(pred_sum)),
        "unmatched_prediction_keys": int(len(unmatched)),
        "writeback_seconds": round(writeback_seconds, 3),
        "output_segy": args.output_segy,
        **residual_stats,
    }
    save_reports(Path(args.output_dir), headers, written_sorted, unfilled, still_missing, observed_changed, unmatched, summary)
    logger.info("writeback summary: %s", summary)
    if args.strict_fill and (unfilled or still_missing or observed_changed):
        raise RuntimeError(
            f"strict fill failed: unfilled={len(unfilled)} "
            f"still_missing={len(still_missing)} observed_changed={len(observed_changed)}"
        )
    return summary


def _make_periodic_fill_callback(
    output_segy: str,
    mask_path: str,
    mask_data: np.ndarray,
    headers: list,
    missing_global: np.ndarray,
    time_ps: int,
    logger: logging.Logger,
):
    """Return a closure that checkpoints current predictions into *output_segy*.

    First invocation copies *mask_path* → *output_segy* (template).
    Subsequent ones open the existing file in ``r+`` and write only traces
    that are marked missing and have accumulated predictions.

    Signature ``(pred_sum, pred_count, flush_count)`` matches the
    ``flush_callback`` expected by ``run_queryctx_inference``.
    """
    import shutil as _shutil
    from pathlib import Path as _Path

    lookup = build_lookup(headers)
    out = mask_data.copy()
    ns = mask_data.shape[1]
    _initialized = False

    def _callback(pred_sum, pred_count, flush_count):
        nonlocal _initialized

        if not _initialized:
            _Path(output_segy).parent.mkdir(parents=True, exist_ok=True)
            _shutil.copy2(mask_path, output_segy)
            _initialized = True

        seen: dict = {}
        for key, total in pred_sum.items():
            for trace_idx in lookup.get(key, []):
                if missing_global[trace_idx]:
                    avg = total / max(pred_count[key], 1)
                    trace = fit_trace(avg, ns, time_ps=time_ps)
                    out[trace_idx] = trace
                    seen[trace_idx] = trace

        if seen:
            write_segy_data_incremental(
                output_segy,
                np.array(list(seen.keys()), dtype=np.intp),
                np.array(list(seen.values()), dtype=np.float32),
            )
        logger.info("periodic fill [flush %d]: wrote %d traces to %s",
                     flush_count, len(seen), output_segy)

    return _callback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E2E V9 queryctx inference and SEGY fill")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--h5_irregular", required=True)
    parser.add_argument("--h5_regular", required=True)
    parser.add_argument("--h5_mask", required=True)
    parser.add_argument("--h5_tgt", default=None)
    parser.add_argument("--mask_path", required=True)
    parser.add_argument("--label_segy", default=None)
    parser.add_argument("--dataset_neighbors_infer", required=True)

    parser.add_argument("--output_dir", default="gen_fill_results")
    parser.add_argument("--output_segy", default=None)
    parser.add_argument("--output_residual_segy", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--fill_interval", type=int, default=0,
                        help="Periodic SEGY checkpoint every N batch flushes "
                             "(0=disabled). Only rank 0 writes in DDP mode.")
    parser.add_argument("--pred_clamp", type=optional_float, default=None,
                        help="Hard clamp model predictions to [-v, v] before inverse "
                             "amplitude scaling (default: None = no clamp).")
    parser.add_argument("--time_ps", type=int, default=1251)
    parser.add_argument("--trace_ps", type=int, default=128)
    parser.add_argument("--chunk_length", type=int, default=256)
    parser.add_argument("--overlap_ratio", type=float, default=0.125)
    parser.add_argument("--missing_eps", type=float, default=1e-10)
    parser.add_argument("--header_mode", choices=["fixed", "self_computed"], default="fixed")
    parser.add_argument("--non_strict_load", dest="strict_load", action="store_false")
    parser.add_argument("--strict_fill", action="store_true", default=False)

    parser.add_argument("--model_type", choices=["e2e_encdec_v9"], default="e2e_encdec_v9")
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
    parser.add_argument("--use_p_scale", type=str2bool, default=False)
    parser.add_argument("--rope_freq_mode", choices=["default", "physical"], default="default")
    parser.add_argument("--lambda_phys_x", type=optional_float, default=None)
    parser.add_argument("--lambda_phys_y", type=optional_float, default=None)
    parser.add_argument("--rope_nyquist_safety", type=float, default=1.0)

    parser.add_argument("--segy_config", type=str, default=None)
    parser.add_argument("--sort_segy", type=str2bool, default=False)
    parser.add_argument("--visualize", type=str2bool, default=False)
    parser.add_argument("--vis_batches", type=int, default=0)

    args = parser.parse_args()
    args.output_dir = str(Path(args.output_dir).resolve())
    args.output_segy = args.output_segy or str(Path(args.output_dir) / "filled_missing.sgy")
    args.output_residual_segy = args.output_residual_segy or str(Path(args.output_dir) / "residual.sgy")
    return args


def _cleanup_stale_rank_results(rank_tmp: Path, logger: logging.Logger, is_main: bool) -> None:
    if rank_tmp.exists():
        import shutil

        if is_main:
            logger.info("cleaning up stale rank results: %s", rank_tmp)
        shutil.rmtree(rank_tmp, ignore_errors=True)


def _merge_rank_results(rank_tmp: Path, rank, world_size, is_main, logger, pred_sum, pred_count):
    if world_size <= 1:
        return pred_sum, pred_count

    import pickle
    import shutil

    rank_base = rank_tmp
    rank_dir = rank_base / f"rank_{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)

    rank_keys = list(pred_sum.keys())
    with open(rank_dir / "pred_keys.pkl", "wb") as f:
        pickle.dump(rank_keys, f, protocol=pickle.HIGHEST_PROTOCOL)
    np.savez(rank_dir / "pred_sum.npz", **{f"arr_{i}": pred_sum[k] for i, k in enumerate(rank_keys)})
    with open(rank_dir / "pred_count.json", "w", encoding="utf-8") as f:
        json.dump({"__".join(map(str, k)): int(v) for k, v in pred_count.items()}, f)
    (rank_base / f".rank_{rank}_done").touch()

    if not is_main:
        return pred_sum, pred_count

    max_wait = 86400
    waited = 0
    pending = set(range(world_size))
    logger.info("waiting for all ranks to finish inference...")
    while pending:
        time.sleep(2)
        waited += 2
        pending = {r for r in pending if not (rank_base / f".rank_{r}_done").exists()}
        if waited % 60 == 0:
            logger.info("still waiting for ranks %s (%.0f s elapsed)", sorted(pending), waited)
        if waited > max_wait:
            raise RuntimeError(f"File barrier timed out. Missing ranks: {sorted(pending)}")

    merged_sum, merged_count = {}, defaultdict(int)
    for r in range(world_size):
        rank_result_dir = rank_base / f"rank_{r}"
        with open(rank_result_dir / "pred_keys.pkl", "rb") as f:
            rank_keys = pickle.load(f)
        with np.load(rank_result_dir / "pred_sum.npz") as npz:
            for i, key in enumerate(rank_keys):
                arr = npz[f"arr_{i}"]
                if key in merged_sum:
                    merged_sum[key] += arr
                else:
                    merged_sum[key] = arr.copy()
        with open(rank_result_dir / "pred_count.json", encoding="utf-8") as f:
            rank_counts = json.load(f)
        for key_str, value in rank_counts.items():
            key = tuple(int(x) for x in key_str.split("__"))
            merged_count[key] += int(value)

    shutil.rmtree(rank_base)
    logger.info("merge complete: %d unique keys", len(merged_sum))
    return merged_sum, merged_count


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0

    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
        logger = setup_logger(Path(args.output_dir)) if is_main else logging.getLogger("e2e_v9_infer")
        if not is_main:
            logger.setLevel(logging.WARNING)
    else:
        device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
        logger = setup_logger(Path(args.output_dir))

    if args.segy_config is not None:
        load_segy_config(args.segy_config)

    train_cfg = load_training_config(args.checkpoint)
    apply_training_config(args, train_cfg)

    if is_main:
        logger.info("args: %s", vars(args))
        logger.info("world_size=%d", world_size)
        if train_cfg:
            logger.info("loaded training_config.json from checkpoint directory")
        print_segy_config()

    # Compute shared temp directory for cross-rank result exchange.
    # Uses local storage (typically /tmp) rather than output_dir to avoid
    # NFS I/O pressure from 8 concurrent writers.
    import hashlib as _hashlib
    import tempfile as _tempfile
    _RANK_TMP = Path(_tempfile.gettempdir()) / (
        "infer_merge_" + _hashlib.md5(args.output_dir.encode()).hexdigest()[:12]
    )
    _cleanup_stale_rank_results(_RANK_TMP, logger, is_main)
    total_start = time.perf_counter()

    if is_main:
        mask_data = read_segy_data(args.mask_path)
        headers = read_segy_headers(args.mask_path, args.header_mode)
        if len(headers) != mask_data.shape[0]:
            raise ValueError(f"header_count={len(headers)} != segy_traces={mask_data.shape[0]}")
        missing_global = np.all(np.abs(mask_data) <= args.missing_eps, axis=1)
        logger.info(
            "template SEGY: traces=%d samples=%d missing=%d",
            mask_data.shape[0],
            mask_data.shape[1],
            int(missing_global.sum()),
        )
        label_data = None
        if args.label_segy:
            label_data = read_segy_data(args.label_segy)
            if label_data.shape != mask_data.shape:
                raise ValueError(f"label shape {label_data.shape} != mask shape {mask_data.shape}")
            logger.info("label SEGY loaded: %s shape=%s", args.label_segy, label_data.shape)
    else:
        headers = None
        missing_global = None
        label_data = None

    logger.info("building DatasetH5_all_queryctx inference dataset")
    trace_sort_keys = getattr(args, "trace_sort_keys", None)
    dataset = DatasetH5_all_queryctx(
        h5File=args.h5_irregular,
        h5File_regular=args.h5_regular,
        h5File_tgt=args.h5_tgt,
        dataset_neighbors=args.dataset_neighbors_infer,
        train=False,
        use_p_scale=args.use_p_scale,
        time_ps=args.time_ps,
        trace_ps=args.trace_ps,
        target_mode="self",
        trace_sort_keys=trace_sort_keys,
    )
    logger.info("queryctx dataset ready: samples=%d time_ps=%d", len(dataset), dataset.time_ps)
    rope_cfg = prepare_rope_frequency(args, dataset, logger)
    if is_main and rope_cfg.get("rope_freq_mode") == "physical":
        logger.info(
            "physical RoPE lambda_min=%s span=%s",
            rope_cfg.get("lambda_min", {}),
            rope_cfg.get("span", {}),
        )

    model = build_model(args).to(device).eval()
    load_checkpoint(model, args.checkpoint, args.strict_load, logger)
    if is_main:
        logger.info(
            "model=%s params=%d device=%s chunk_length=%d overlap=%.3f",
            args.model_type,
            sum(p.numel() for p in model.parameters()),
            device,
            args.chunk_length,
            args.overlap_ratio,
        )

    # ---- Periodic fill callback (optional) ----
    fill_callback = None
    if is_main and args.fill_interval > 0:
        fill_callback = _make_periodic_fill_callback(
            output_segy=args.output_segy,
            mask_path=args.mask_path,
            mask_data=mask_data,
            headers=headers,
            missing_global=missing_global,
            time_ps=args.time_ps,
            logger=logger,
        )

    pred_sum, pred_count, inference_seconds, infer_stats = run_queryctx_inference(
        dataset=dataset,
        model=model,
        device=device,
        batch_size=args.batch_size,
        chunk_length=args.chunk_length,
        overlap_ratio=args.overlap_ratio,
        visualize=args.visualize,
        vis_dir=str(Path(args.output_dir) / "vis"),
        vis_max=args.vis_batches,
        progress=is_main,
        logger=logger,
        rank=rank,
        world_size=world_size,
        flush_callback=fill_callback,
        flush_interval=args.fill_interval,
        pred_clamp=args.pred_clamp,
    )

    pred_sum, pred_count = _merge_rank_results(
        _RANK_TMP, rank, world_size, is_main, logger, pred_sum, pred_count
    )

    if is_main:
        summary = fill_segy(args, headers, missing_global, pred_sum, pred_count, logger, label_data=label_data)
        summary.update(infer_stats)
        summary["prediction_keys"] = int(len(pred_sum))
        summary["num_gpus"] = int(world_size)
        summary["chunk_length"] = int(args.chunk_length)
        summary["overlap_ratio"] = float(args.overlap_ratio)
        summary["inference_seconds"] = round(float(inference_seconds), 3)
        summary["total_seconds"] = round(time.perf_counter() - total_start, 3)

        if args.sort_segy:
            sorted_path = args.output_segy.replace(".sgy", "_sorted.sgy")
            logger.info("sorting output SEG-Y: %s", sorted_path)
            sort_output_segy(args.output_segy, sorted_path)
            summary["output_segy_sorted"] = sorted_path

        (Path(args.output_dir) / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("done in %.2fs | output=%s", summary["total_seconds"], args.output_segy)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
