"""E2E V9 query-context inference engine.

This module batches DatasetH5_all_queryctx samples, runs a direct V9 EncDec
forward pass, and accumulates predictions for query traces by SEG-Y key.
"""

from __future__ import annotations

import json
import inspect
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from config.segy_config import get_key_columns
    from model import trace_time_chunk, trace_time_unchunk
except ImportError:
    from .config.segy_config import get_key_columns
    from .model import trace_time_chunk, trace_time_unchunk

try:
    from tqdm import tqdm
except Exception:

    class _TqdmFallback:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **kwargs):
            return None

    def tqdm(iterable, **kwargs):
        return _TqdmFallback(iterable)


def add_prediction(pred_sum, pred_count, key, trace):
    trace = np.asarray(trace, dtype=np.float64).reshape(-1)
    if key not in pred_sum:
        pred_sum[key] = trace
    else:
        pred_sum[key] += trace
    pred_count[key] += 1


def fit_trace(trace: np.ndarray, ns: int, time_ps: Optional[int] = None) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float32).reshape(-1)
    if time_ps is not None and trace.size != ns:
        if ns > time_ps:
            return np.pad(trace, (ns - time_ps, 0), constant_values=0).astype(np.float32)
        if ns < time_ps:
            return trace[time_ps - ns:].astype(np.float32)
        return trace.astype(np.float32)
    if trace.size > ns:
        return trace[:ns]
    if trace.size < ns:
        return np.pad(trace, (0, ns - trace.size), constant_values=0).astype(np.float32)
    return trace


def _coords_to_01(coords):
    return (0.5 * (coords + 1.0)).clamp(0.0, 1.0)


def _repeat_trace_mask_for_chunks(mask, n_chunks: int):
    return mask.float().repeat(1, int(n_chunks))


def _model_accepts_valid_mask(model) -> bool:
    forward = getattr(model, "forward", None)
    if forward is None:
        return False
    try:
        return "valid_mask" in inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return bool(getattr(model, "accepts_valid_mask", True))


def _run_model_forward(model, x_chunk, coords_chunk, time_bounds, token_mask, valid_token):
    if _model_accepts_valid_mask(model):
        return model(
            x_chunk,
            coords_chunk,
            time_bounds,
            mask=token_mask,
            valid_mask=valid_token,
        )
    return model(x_chunk, coords_chunk, time_bounds, mask=token_mask)


def run_e2e_batch(
    model,
    x_norm: np.ndarray,
    coords: np.ndarray,
    trace_mask: np.ndarray,
    valid_mask: np.ndarray,
    device,
    chunk_length: int,
    overlap_ratio: float,
    pred_clamp: Optional[float] = None,
) -> np.ndarray:
    import torch

    x_t = torch.from_numpy(x_norm.astype(np.float32)).to(device)
    c_t = torch.from_numpy(coords.astype(np.float32)).to(device)
    c_t = _coords_to_01(c_t)
    trace_mask_t = torch.from_numpy(trace_mask.astype(np.float32)).to(device)
    valid_mask_t = torch.from_numpy(valid_mask.astype(np.float32)).to(device)

    x_chunk, c_chunk, time_bounds, chunk_info = trace_time_chunk(
        x_t, c_t, int(chunk_length), float(overlap_ratio)
    )
    token_mask = _repeat_trace_mask_for_chunks(trace_mask_t, int(chunk_info["n_chunks"]))
    valid_token = _repeat_trace_mask_for_chunks(valid_mask_t, int(chunk_info["n_chunks"]))
    time_bounds = time_bounds.to(device).float() / max(float(chunk_info["time_length"]), 1.0)
    with torch.inference_mode():
        pred_chunk = _run_model_forward(
            model,
            x_chunk,
            c_chunk,
            time_bounds,
            token_mask,
            valid_token,
        )
    pred = trace_time_unchunk(pred_chunk.detach().cpu(), chunk_info)
    if pred_clamp is not None:
        # Optional hard clamp to normalized range before inverse scaling.
        # When None (default), model outputs are passed through without
        # truncation, preserving possible amplitude excursions beyond the
        # training clipping threshold.
        pred = pred.clamp(-float(pred_clamp), float(pred_clamp))
    return pred.numpy().astype(np.float32)


def run_queryctx_inference(
    dataset,
    model,
    device,
    batch_size: int = 1,
    chunk_length: int = 256,
    overlap_ratio: float = 0.125,
    visualize: bool = False,
    vis_dir: Optional[str] = None,
    vis_max: int = 0,
    progress: bool = True,
    logger=None,
    rank: int = 0,
    world_size: int = 1,
    flush_callback=None,
    flush_interval: int = 0,
    pred_clamp: Optional[float] = None,
) -> Tuple[Dict, Dict, float, Dict[str, Any]]:
    import torch

    pred_sum: Dict = {}
    pred_count = defaultdict(int)
    total_missing = 0
    total_traces = 0
    is_main = rank == 0

    vis_path = None
    if vis_dir is not None:
        vis_path = Path(vis_dir)
        vis_path.mkdir(parents=True, exist_ok=True)
    vis_limit = vis_max if vis_max > 0 else float("inf")
    _flush_count = 0

    all_indices = list(range(len(dataset)))
    if world_size > 1:
        all_indices = all_indices[rank::world_size]
    batch_size = max(1, int(batch_size))

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    sample_buf: list = []
    total_samples = len(all_indices)
    desc = f"e2e v9 inference [rank {rank}]" if world_size > 1 else "e2e v9 inference"
    iterator = tqdm(
        all_indices,
        desc=desc,
        unit="sample",
        disable=not (progress and is_main),
        smoothing=0.01,
    )

    def _build_batch(samples: list):
        batch_count = len(samples)
        max_tr = max(s[0]["data"].shape[0] for s in samples)
        time_len = samples[0][0]["data"].shape[1]

        x_batch = np.zeros((batch_count, max_tr, time_len), dtype=np.float32)
        c_batch = np.zeros((batch_count, max_tr, 4), dtype=np.float32)
        trace_mask = np.zeros((batch_count, max_tr), dtype=np.float32)
        valid_mask = np.zeros((batch_count, max_tr), dtype=np.float32)
        scales = np.zeros(batch_count, dtype=np.float32)
        meta_list = []

        for b, (sample, sample_idx) in enumerate(samples):
            n_tr = int(sample["data"].shape[0])
            is_query = np.asarray(sample["is_query"], dtype=bool)
            x_batch[b, :n_tr] = np.asarray(sample["masked_patch"], dtype=np.float32)
            c_batch[b, :n_tr, 0] = np.asarray(sample["sx_patch"], dtype=np.float32)
            c_batch[b, :n_tr, 1] = np.asarray(sample["sy_patch"], dtype=np.float32)
            c_batch[b, :n_tr, 2] = np.asarray(sample["rx_patch"], dtype=np.float32)
            c_batch[b, :n_tr, 3] = np.asarray(sample["ry_patch"], dtype=np.float32)
            trace_mask[b, :n_tr] = np.asarray(
                sample.get("trace_mask", (~is_query).astype(np.float32)),
                dtype=np.float32,
            )
            valid_mask[b, :n_tr] = 1.0
            scales[b] = float(sample["amp_scale"])
            meta_list.append(
                {
                    "n_tr": n_tr,
                    "is_query": is_query,
                    "trace_obs": trace_mask[b, :n_tr].copy(),
                    "masked_patch_raw": np.asarray(
                        sample.get("masked_patch_raw", sample["masked_patch"]),
                        dtype=np.float32,
                    ),
                    "data_raw": np.asarray(sample.get("data_raw", sample["data"]), dtype=np.float32),
                    "patch_info": sample.get("patch_info", {}),
                    "sample_idx": sample_idx,
                }
            )

        return x_batch, c_batch, trace_mask, valid_mask, scales, meta_list

    def _flush():
        nonlocal total_missing, total_traces, pred_sum, pred_count, _flush_count
        if not sample_buf:
            return

        x_batch, c_batch, trace_mask, valid_mask, scales, meta_list = _build_batch(sample_buf)
        pred = run_e2e_batch(
            model=model,
            x_norm=x_batch,
            coords=c_batch,
            trace_mask=trace_mask,
            valid_mask=valid_mask,
            device=device,
            chunk_length=chunk_length,
            overlap_ratio=overlap_ratio,
            pred_clamp=pred_clamp,
        )
        pred = pred * scales[:, None, None]

        key_cols = get_key_columns()
        for b, meta in enumerate(meta_list):
            n_tr = meta["n_tr"]
            is_query = meta["is_query"]
            pred_b = pred[b, :n_tr]
            missing_count = int(is_query.sum())
            total_missing += missing_count
            total_traces += n_tr

            if visualize and is_main and meta["sample_idx"] < vis_limit:
                _visualize_sample(
                    meta["masked_patch_raw"][:n_tr],
                    pred_b,
                    meta["data_raw"][:n_tr],
                    meta["trace_obs"],
                    int(meta["sample_idx"]),
                    vis_path,
                )

            patch_info = meta["patch_info"]
            key_arrays = {col: patch_info.get(col) for col in key_cols}
            for j in range(n_tr):
                if is_query[j]:
                    key = tuple(
                        int(key_arrays[col][j]) if key_arrays[col] is not None else 0
                        for col in key_cols
                    )
                    add_prediction(pred_sum, pred_count, key, pred_b[j])

        sample_buf.clear()
        _flush_count += 1
        if (
            flush_callback is not None
            and flush_interval > 0
            and _flush_count % flush_interval == 0
        ):
            flush_callback(pred_sum, pred_count, _flush_count)

    for idx in iterator:
        sample = dataset[idx]
        sample_buf.append((sample, idx))
        if len(sample_buf) >= batch_size:
            _flush()
        if progress and is_main:
            is_query = np.asarray(sample["is_query"], dtype=bool)
            iterator.set_postfix(
                sample=idx,
                traces=int(sample["data"].shape[0]),
                missing=int(is_query.sum()),
            )

    _flush()

    if flush_callback is not None and flush_interval > 0:
        flush_callback(pred_sum, pred_count, _flush_count)

    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start_time

    if logger is not None:
        logger.info(
            "e2e v9 inference done: %.2fs | samples=%d traces=%d missing=%d keys=%d",
            seconds,
            total_samples,
            total_traces,
            total_missing,
            len(pred_sum),
        )

    return pred_sum, pred_count, seconds, {
        "dataset_samples": int(len(dataset)),
        "dataset_traces": int(total_traces),
        "dataset_missing": int(total_missing),
        "prediction_keys": int(len(pred_sum)),
    }


def _visualize_sample(
    masked_patch: np.ndarray,
    pred: np.ndarray,
    data: np.ndarray,
    trace_obs: np.ndarray,
    sample_idx: int,
    vis_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    residual = pred - data
    vmax = float(
        max(
            abs(masked_patch.min()),
            abs(masked_patch.max()),
            abs(pred.min()),
            abs(pred.max()),
            abs(data.min()),
            abs(residual.min()),
            abs(residual.max()),
            1e-6,
        )
    )

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)
    im = None
    for ax, img, title in zip(
        axes,
        [masked_patch, pred, data, residual],
        ["Masked Input", "E2E V9 Prediction", "Target", "Residual (Pred - Target)"],
    ):
        im = ax.imshow(img.T, aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax, origin="upper")
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time Sample")
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    n_missing = int((trace_obs < 0.5).sum())
    fig.suptitle(
        f"Sample {sample_idx} | {trace_obs.shape[0]} traces | {n_missing} missing",
        fontsize=12,
    )
    fig.savefig(vis_dir / f"sample_{sample_idx:04d}.png", dpi=120)
    plt.close(fig)


def load_training_config(checkpoint_path: str) -> Dict[str, Any]:
    ckpt = Path(checkpoint_path)
    config_path = ckpt.parent.parent / "logs" / "training_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    config_path = ckpt.parent / "logs" / "training_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}
