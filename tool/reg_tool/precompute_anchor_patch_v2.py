#!/usr/bin/env python3
"""Production script for ``anchor_patch_debug.ipynb`` — thin CLI over ``core.py``.

The heavy logic lives in ``core.py`` and ``patch_sampler.py``.  This file only
keeps the argument parser, lazy-dependency loading, and the ``main()``
orchestration loop.

Defaults mirror the notebook source:
- base dir: ``../h5/dongfang`` relative to this file
- patch dir: ``patchV2``
- train anchor selector: ``value_based_anchor_sampling``
- train trusted points: all raw observations
- infer query mask: regular ``mask == True``
- infer context filter: none
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple


# ── lazy imports (keep --help fast) ─────────────────────────────────


def load_runtime_dependencies() -> None:
    """Import heavy scientific dependencies after argparse has handled --help."""
    global np
    global File
    global generate_binning_keys
    global raw_obs_valid_mask_from_regular_trusted_mask
    global read_coord4, read_trace_data, read_regular_mask
    global _make_context_selector
    global _stable_unique_index_list
    global diverse_topk
    global resolve_block_tuple
    global build_grid_index_map_4d_from_coord_grid
    global make_grid_blocks_from_index_map_4d
    global make_query_mask
    global save_norm_stats
    global object_array, summarize_query_context
    global validate_outputs
    global normalize_coords
    global precompute_infer_patches_4d
    global precompute_train_patches_2d

    if "np" in globals():
        return

    import numpy as _np
    from h5py import File as _File

    try:
        from .core import (
            generate_binning_keys as _generate_binning_keys,
            raw_obs_valid_mask_from_regular_trusted_mask as _raw_obs_valid_mask,
            read_coord4 as _read_coord4,
            read_trace_data as _read_trace_data,
            read_regular_mask as _read_regular_mask,
            resolve_block_tuple as _resolve_block_tuple,
            build_grid_index_map_4d_from_coord_grid as _build_grid_index_map_4d,
            make_query_mask as _make_query_mask,
            save_norm_stats as _save_norm_stats,
            object_array as _object_array,
            summarize_query_context as _summarize_query_context,
            validate_outputs as _validate_outputs,
        )
        from .patch_sampler import (
            _make_context_selector as _make_context_selector,
            _stable_unique_index_list as _stable_unique_index_list,
            diverse_topk as _diverse_topk,
            make_grid_blocks_from_index_map_4d as _make_grid_blocks_from_index_map_4d,
            normalize_coords as _normalize_coords,
            precompute_infer_patches_4d as _precompute_infer_patches_4d,
            precompute_train_patches_2d as _precompute_train_patches_2d,
        )
    except ImportError:
        from core import (
            generate_binning_keys as _generate_binning_keys,
            raw_obs_valid_mask_from_regular_trusted_mask as _raw_obs_valid_mask,
            read_coord4 as _read_coord4,
            read_trace_data as _read_trace_data,
            read_regular_mask as _read_regular_mask,
            resolve_block_tuple as _resolve_block_tuple,
            build_grid_index_map_4d_from_coord_grid as _build_grid_index_map_4d,
            make_query_mask as _make_query_mask,
            save_norm_stats as _save_norm_stats,
            object_array as _object_array,
            summarize_query_context as _summarize_query_context,
            validate_outputs as _validate_outputs,
        )
        from patch_sampler import (
            _make_context_selector as _make_context_selector,
            _stable_unique_index_list as _stable_unique_index_list,
            diverse_topk as _diverse_topk,
            make_grid_blocks_from_index_map_4d as _make_grid_blocks_from_index_map_4d,
            normalize_coords as _normalize_coords,
            precompute_infer_patches_4d as _precompute_infer_patches_4d,
            precompute_train_patches_2d as _precompute_train_patches_2d,
        )

    np = _np
    File = _File
    generate_binning_keys = _generate_binning_keys
    raw_obs_valid_mask_from_regular_trusted_mask = _raw_obs_valid_mask
    read_coord4 = _read_coord4
    read_trace_data = _read_trace_data
    read_regular_mask = _read_regular_mask
    _make_context_selector = _make_context_selector
    _stable_unique_index_list = _stable_unique_index_list
    diverse_topk = _diverse_topk
    resolve_block_tuple = _resolve_block_tuple
    build_grid_index_map_4d_from_coord_grid = _build_grid_index_map_4d
    make_grid_blocks_from_index_map_4d = _make_grid_blocks_from_index_map_4d
    make_query_mask = _make_query_mask
    save_norm_stats = _save_norm_stats
    object_array = _object_array
    summarize_query_context = _summarize_query_context
    validate_outputs = _validate_outputs
    normalize_coords = _normalize_coords
    precompute_infer_patches_4d = _precompute_infer_patches_4d
    precompute_train_patches_2d = _precompute_train_patches_2d


# ── CLI helpers ─────────────────────────────────────────────────────


def default_base_dir() -> Path:
    return (Path(__file__).resolve().parent / ".." / "h5" / "dongfang").resolve()


def parse_metric_weights(text: str) -> Tuple[float, float, float, float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(values) != 4:
        raise argparse.ArgumentTypeError("metric weights must be four comma-separated numbers")
    return tuple(values)  # type: ignore[return-value]


def as_path(path: Optional[str], default: Path) -> Path:
    return Path(path).expanduser().resolve() if path else default.resolve()


# ── argument parser ─────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute the notebook V2 train pool and infer query/context patch files."
    )
    parser.add_argument("--base-dir", type=str, default=str(default_base_dir()))
    parser.add_argument("--raw-h5", type=str, default=None)
    parser.add_argument("--target-h5", type=str, default=None)
    parser.add_argument("--regular-h5", type=str, default=None)
    parser.add_argument("--group-key", type=str, default="1551")
    parser.add_argument("--patch-dir", type=str, default=None)
    parser.add_argument("--regular-mask-key", type=str, default="mask")

    parser.add_argument("--num-anchors", type=int, default=None)
    parser.add_argument("--anchor-stride", type=int, default=128)
    parser.add_argument("--k-patch", type=int, default=256)
    parser.add_argument("--top-l", type=int, default=None)
    parser.add_argument("--num-query", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metric-weights", type=parse_metric_weights, default=(1.0, 1.0, 1.0, 1.0))
    parser.add_argument(
        "--train-anchor-selector",
        choices=[
            "farthest_point_sampling",
            "facility_location_anchor_sampling",
            "value_based_anchor_sampling",
        ],
        default="value_based_anchor_sampling",
    )
    parser.add_argument(
        "--train-trusted-source",
        choices=["all", "regular_mask"],
        default="all",
        help="Notebook default is all; regular_mask uses raw rows whose 4D keys map to regular mask=True.",
    )
    parser.add_argument("--value-local-top-l", type=int, default=None)
    parser.add_argument("--value-suppression", choices=["subtractive", "multiplicative"], default="subtractive")
    parser.add_argument("--value-suppression-lambda", type=float, default=1.0)
    parser.add_argument("--value-score-tol", type=float, default=0.0)
    parser.add_argument("--value-knn-gpu-batch-rows", type=int, default=512)
    parser.add_argument("--value-knn-gpu-device", type=str, default="cuda:0")
    parser.add_argument("--value-knn-full-matrix-max-n", type=int, default=4096)
    parser.add_argument("--train-knn-use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-suppression-use-gpu", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--block-size", type=int, nargs=4, default=None)
    parser.add_argument("--stride", type=int, nargs=4, default=None)
    parser.add_argument("--block-divisors", type=int, nargs=4, default=(6, 21, 7, 5))
    parser.add_argument("--stride-divisors", type=int, nargs=4, default=(6, 21, 7, 5))
    parser.add_argument("--on-grid-collision", choices=["raise", "last"], default="raise")
    parser.add_argument(
        "--query-mask-mode",
        choices=["regular_true", "regular_false", "all", "none"],
        default="regular_false",
        help="Use regular_false for interpolation: regular mask True means observed/trusted, so False means missing query targets.",
    )
    parser.add_argument(
        "--infer-obs-valid-source",
        choices=["none", "regular_mask"],
        default="none",
        help="Notebook default is none.",
    )
    parser.add_argument("--infer-top-l", type=int, default=None)
    parser.add_argument("--max-query-per-patch", type=int, default=32)
    parser.add_argument("--gpu-query-chunk-size", type=int, default=64)
    parser.add_argument("--infer-gpu-device", type=str, default="cuda:0")
    parser.add_argument("--infer-use-gpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-full-query-coverage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--greedy-fill-uncovered", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-infer", action="store_true")
    parser.add_argument("--save-legacy-anchor-files", action="store_true")
    parser.add_argument("--save-grid-index-map", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--summary-json", type=str, default=None)
    parser.add_argument(
        "--train-mode",
        choices=["anchor", "block"],
        default="anchor",
        help="anchor: legacy anchor-based pool; block: 4D block aligned with inference",
    )
    parser.add_argument(
        "--min-obs-per-block",
        type=int,
        default=4,
        help="Minimum raw observations per block to keep as training pool (block mode only)",
    )
    parser.add_argument(
        "--block-pool-margin",
        type=int,
        default=1,
        help="Expand block boundary by this many logical-grid cells when building training pool (block mode only). "
             "Use 1+ when stride==block_size (no overlap) to cover edge queries whose nearest context may fall in adjacent blocks.",
    )
    parser.add_argument(
        "--train-query-mask-mode",
        choices=["regular_true", "regular_false", "all", "none"],
        default="regular_true",
        help="Query mask for training holdout: regular_true = holdout from observed grid points (has reliable targets).",
    )
    parser.add_argument(
        "--grid-holdout-ratio",
        type=float,
        default=0.3,
        help="Fraction of regular_true grid points per block used as holdout queries (block mode only).",
    )
    parser.add_argument(
        "--emit-regular-holdout",
        action="store_true",
        help="In anchor mode, additionally emit train_regular_holdout_query_context.npz.",
    )
    parser.add_argument(
        "--context-budget",
        type=int,
        default=None,
        help="Hard cap on context size per holdout patch. Default=k_patch.",
    )
    return parser


# ── main ────────────────────────────────────────────────────────────


def main() -> None:
    args = build_arg_parser().parse_args()
    load_runtime_dependencies()

    base_dir = as_path(args.base_dir, default_base_dir())
    raw_h5 = as_path(args.raw_h5, base_dir / "raw5d_data1104.h5")
    regular_h5 = as_path(args.regular_h5, base_dir / "reg5dbin_label1031.h5")
    target_h5 = args.target_h5
    patch_dir = as_path(args.patch_dir, base_dir / "patchV2")
    patch_dir.mkdir(parents=True, exist_ok=True)

    print("base_dir:", base_dir)
    print("raw_h5:", raw_h5)
    print("regular_h5:", regular_h5)
    print("patch_dir:", patch_dir)

    with File(raw_h5, "r") as f_raw, File(regular_h5, "r") as f_reg:
        raw_group = f_raw[args.group_key]
        regular_group = f_reg[args.group_key]
        print("raw keys:", list(raw_group.keys()))
        print("regular keys:", list(regular_group.keys()))

        trace_obs = read_trace_data(raw_group)
        coord_obs = read_coord4(raw_group)
        coord_grid = read_coord4(regular_group)
        regular_mask = read_regular_mask(
                regular_group, args.regular_mask_key, coord_grid.shape[0],
                target_h5, group_key=args.group_key,
            )
        print("regular mask true count:", int(regular_mask.sum()))

        raw_keys = generate_binning_keys(raw_group).astype(np.int64)
        reg_keys = generate_binning_keys(regular_group).astype(np.int64)

    if trace_obs.shape[0] != coord_obs.shape[0] or raw_keys.shape[0] != coord_obs.shape[0]:
        raise ValueError("raw data, raw coordinates, and raw binning keys must have the same length")
    if reg_keys.shape[0] != coord_grid.shape[0]:
        raise ValueError("regular coordinates and regular binning keys must have the same length")

    print("trace_obs shape:", trace_obs.shape)
    print("coord_obs shape:", coord_obs.shape)
    print("coord_grid shape:", coord_grid.shape)
    print("regular mask true count:", int(regular_mask.sum()))

    raw_obs_valid = raw_obs_valid_mask_from_regular_trusted_mask(
        raw_binning_keys=raw_keys,
        reg_binning_keys=reg_keys,
        regular_trusted_mask=regular_mask,
    )
    trusted_from_mask = np.flatnonzero(raw_obs_valid).astype(np.int64)
    np.save(patch_dir / "raw_obs_valid_mask.npy", raw_obs_valid)
    np.save(patch_dir / "trusted_row_indices_from_regular_mask.npy", trusted_from_mask)
    print("raw_obs_valid true count:", int(raw_obs_valid.sum()))

    coord_obs_norm, coord_grid_norm, norm_stats = normalize_coords(coord_obs, coord_grid)

    # compute grid_steps for physical RoPE
    def _grid_step(arr):
        u = np.sort(np.unique(arr))
        if u.size < 2:
            return None
        d = np.diff(u)
        d = d[d > 1e-9]
        return float(np.median(d)) if d.size > 0 else None

    gs_sx = _grid_step(coord_grid[:, 0])
    gs_sy = _grid_step(coord_grid[:, 1])
    gs_rx = _grid_step(coord_grid[:, 2])
    gs_ry = _grid_step(coord_grid[:, 3])
    Lx = float(max(coord_grid[:, 0].max() - coord_grid[:, 0].min(),
                   coord_grid[:, 2].max() - coord_grid[:, 2].min()))
    Ly = float(max(coord_grid[:, 1].max() - coord_grid[:, 1].min(),
                   coord_grid[:, 3].max() - coord_grid[:, 3].min()))
    grid_steps = {
        "grid_step_sx": gs_sx, "grid_step_sy": gs_sy,
        "grid_step_rx": gs_rx, "grid_step_ry": gs_ry,
        "Lx": Lx if Lx > 0 else None, "Ly": Ly if Ly > 0 else None,
    }
    save_norm_stats(patch_dir / "coord_norm_stats.npz", norm_stats, grid_steps=grid_steps)
    np.save(patch_dir / "coord_obs_norm.npy", coord_obs_norm)
    np.save(patch_dir / "coord_grid_norm.npy", coord_grid_norm)
    print("coord_obs_norm range:", float(coord_obs_norm.min()), float(coord_obs_norm.max()))
    print("coord_grid_norm range:", float(coord_grid_norm.min()), float(coord_grid_norm.max()))

    grid_index_map_4d, grid_info = build_grid_index_map_4d_from_coord_grid(
        coord_grid_norm,
        on_collision=args.on_grid_collision,
    )
    dims_4d = (grid_info["nsx"], grid_info["nsy"], grid_info["nrx"], grid_info["nry"])
    valid_grid_cells = int(np.count_nonzero(grid_index_map_4d >= 0))
    if valid_grid_cells != coord_grid.shape[0] and args.on_grid_collision == "raise":
        raise ValueError("grid index map valid cell count does not match coord_grid rows")
    if args.save_grid_index_map:
        np.save(patch_dir / "grid_index_map_4d.npy", grid_index_map_4d)
        np.savez(
            patch_dir / "grid_index_map_4d_levels.npz",
            sx_levels=grid_info["sx_levels"],
            sy_levels=grid_info["sy_levels"],
            rx_levels=grid_info["rx_levels"],
            ry_levels=grid_info["ry_levels"],
        )
    print("grid_index_map_4d shape:", grid_index_map_4d.shape)
    print("grid_index_map_4d valid cells:", valid_grid_cells)

    metric_weights = list(args.metric_weights)
    k_patch = int(args.k_patch)
    top_l = int(args.top_l) if args.top_l is not None else k_patch + 128
    infer_top_l = int(args.infer_top_l) if args.infer_top_l is not None else k_patch * 2
    num_anchors = (
        int(args.num_anchors)
        if args.num_anchors is not None
        else max(1, int(trace_obs.shape[0]) // int(args.anchor_stride))
    )
    value_local_top_l = args.value_local_top_l
    if value_local_top_l is None:
        value_local_top_l = top_l

    summary: Dict[str, Any] = {
        "base_dir": str(base_dir),
        "raw_h5": str(raw_h5),
        "regular_h5": str(regular_h5),
        "patch_dir": str(patch_dir),
        "group_key": args.group_key,
        "n_obs": int(coord_obs.shape[0]),
        "n_grid": int(coord_grid.shape[0]),
        "grid_shape_4d": [int(x) for x in dims_4d],
        "regular_mask_true": int(regular_mask.sum()),
        "raw_obs_valid_true": int(raw_obs_valid.sum()),
        "k_patch": k_patch,
        "top_l": top_l,
        "infer_top_l": infer_top_l,
        "num_anchors": num_anchors,
        "metric_weights": metric_weights,
    }

    # Build query masks (shared by train and infer branches)
    train_query_mask = make_query_mask(
        mode=args.train_query_mask_mode,
        regular_mask=regular_mask,
        n_grid=coord_grid.shape[0],
    )
    infer_query_mask = make_query_mask(
        mode=args.query_mask_mode,
        regular_mask=regular_mask,
        n_grid=coord_grid.shape[0],
    )
    context_budget = int(args.context_budget) if args.context_budget is not None else k_patch

    if not args.skip_train:
        if args.train_mode == "block":
            # ---- Block-based training: regular holdout query-context ----
            # For each block, holdout a fraction of regular_true grid points as
            # queries, select context from global obs pool (same as inference),
            # inject margin raw obs, then apply leak prevention and hard budget.
            # Output: train_regular_holdout_query_context.npz (infer_query_context schema)
            block_size = resolve_block_tuple(args.block_size, args.block_divisors, dims_4d, "block_size")
            stride = resolve_block_tuple(args.stride, args.stride_divisors, dims_4d, "stride")

            sx_levels = grid_info["sx_levels"]
            sy_levels = grid_info["sy_levels"]
            rx_levels = grid_info["rx_levels"]
            ry_levels = grid_info["ry_levels"]
            isx_obs = np.clip(np.searchsorted(sx_levels, coord_obs_norm[:, 0]), 0, len(sx_levels) - 1)
            isy_obs = np.clip(np.searchsorted(sy_levels, coord_obs_norm[:, 1]), 0, len(sy_levels) - 1)
            irx_obs = np.clip(np.searchsorted(rx_levels, coord_obs_norm[:, 2]), 0, len(rx_levels) - 1)
            iry_obs = np.clip(np.searchsorted(ry_levels, coord_obs_norm[:, 3]), 0, len(ry_levels) - 1)

            blocks = make_grid_blocks_from_index_map_4d(
                grid_index_map_4d, block_size=block_size, stride=stride
            )

            select_contexts = _make_context_selector(
                obs_coords=coord_obs_norm,
                k_patch=k_patch,
                top_l=top_l,
                metric_weights=metric_weights,
                beta=args.beta,
                obs_valid_mask=None,
                use_gpu=False,
            )

            # Pre-build raw_key → obs_idx mapping for leak prevention
            raw_key_to_obs = {}
            for i in range(raw_keys.shape[0]):
                key_t = tuple(raw_keys[i].tolist())
                if key_t not in raw_key_to_obs:
                    raw_key_to_obs[key_t] = []
                raw_key_to_obs[key_t].append(i)

            grid_query_list = []
            context_list = []
            anchor_list = []
            block_ids = []
            center_ids = []

            holdout_rng = np.random.default_rng(args.seed)

            for block in blocks:
                point_idx = block["grid_point_indices"]
                if point_idx.size == 0:
                    continue

                # 1. Take regular_true grid points in this block (observed, have targets)
                grid_true_in_block = point_idx[train_query_mask[point_idx]]
                if grid_true_in_block.size == 0:
                    continue

                # 2. Holdout a fraction as queries
                holdout_size = max(1, int(grid_true_in_block.size * args.grid_holdout_ratio))
                holdout_idx = holdout_rng.choice(grid_true_in_block, holdout_size, replace=False)

                # 3. Lexsort by 4D coordinates
                block_grid_coords = coord_grid_norm[holdout_idx]
                sort_order = np.lexsort([block_grid_coords[:, i] for i in (3, 2, 1, 0)])
                sorted_query = holdout_idx[sort_order]

                # 4. Chunk by max_query_per_patch
                chunk_size = max(args.max_query_per_patch, 1)
                n_chunks = max(1, (sorted_query.size + chunk_size - 1) // chunk_size)
                chunks = np.array_split(sorted_query, n_chunks)

                # 5. Take margin raw obs for context injection
                block_grid_coords_all = coord_grid_norm[point_idx]
                isx = np.clip(np.searchsorted(sx_levels, block_grid_coords_all[:, 0]), 0, len(sx_levels) - 1)
                isy = np.clip(np.searchsorted(sy_levels, block_grid_coords_all[:, 1]), 0, len(sy_levels) - 1)
                irx = np.clip(np.searchsorted(rx_levels, block_grid_coords_all[:, 2]), 0, len(rx_levels) - 1)
                iry = np.clip(np.searchsorted(ry_levels, block_grid_coords_all[:, 3]), 0, len(ry_levels) - 1)

                margin = int(args.block_pool_margin)
                isx_min = max(0, int(isx.min()) - margin)
                isx_max = min(len(sx_levels), int(isx.max()) + 1 + margin)
                isy_min = max(0, int(isy.min()) - margin)
                isy_max = min(len(sy_levels), int(isy.max()) + 1 + margin)
                irx_min = max(0, int(irx.min()) - margin)
                irx_max = min(len(rx_levels), int(irx.max()) + 1 + margin)
                iry_min = max(0, int(iry.min()) - margin)
                iry_max = min(len(ry_levels), int(iry.max()) + 1 + margin)

                mask_in_block = (
                    (isx_obs >= isx_min) & (isx_obs < isx_max) &
                    (isy_obs >= isy_min) & (isy_obs < isy_max) &
                    (irx_obs >= irx_min) & (irx_obs < irx_max) &
                    (iry_obs >= iry_min) & (iry_obs < iry_max)
                )
                obs_in_block = np.flatnonzero(mask_in_block).astype(np.int64)

                for chunk in chunks:
                    if chunk.size == 0:
                        continue

                    # anchor = chunk grid point mean (using coord_grid_norm)
                    anchor_coord = np.mean(coord_grid_norm[chunk], axis=0).astype(np.float32)

                    # select context (same as inference)
                    context_batch = select_contexts(anchor_coord[None, :])
                    if len(context_batch) != 1:
                        raise RuntimeError(
                            "context selector returned inconsistent batch size in block training mode"
                        )
                    context_idx = np.asarray(context_batch[0], dtype=np.int64).reshape(-1)

                    # Merge context + margin obs as candidates
                    all_candidates = _stable_unique_index_list([context_idx, obs_in_block])

                    # Leak prevention: exclude raw obs whose binning key matches
                    # any held-out query grid key
                    heldout_keys = set(tuple(reg_keys[g].tolist()) for g in chunk.tolist())
                    leaked_obs = set()
                    for hk in heldout_keys:
                        if hk in raw_key_to_obs:
                            leaked_obs.update(raw_key_to_obs[hk])
                    safe_mask = np.ones(all_candidates.size, dtype=bool)
                    for li, idx in enumerate(all_candidates.tolist()):
                        if idx in leaked_obs:
                            safe_mask[li] = False
                    safe_candidates = all_candidates[safe_mask]

                    if safe_candidates.size == 0:
                        continue

                    # Hard budget: diverse_topk selects back to context_budget
                    final_context = diverse_topk(
                        center_coord=anchor_coord,
                        candidate_idx=safe_candidates,
                        all_coords=coord_obs_norm,
                        k=min(context_budget, safe_candidates.size),
                        metric_weights=metric_weights,
                        beta=args.beta,
                    ).astype(np.int64)

                    if final_context.size < args.min_obs_per_block:
                        continue

                    grid_query_list.append(chunk.astype(np.int64))
                    context_list.append(final_context)
                    anchor_list.append(np.asarray([int(chunk[len(chunk) // 2])], dtype=np.int64))
                    block_ids.append(int(block["block_id"]))
                    center_ids.append(int(block["block_center_grid_index"].item()))

            if not grid_query_list:
                raise ValueError("No valid block holdout patches generated")

            np.savez(
                patch_dir / "train_regular_holdout_query_context.npz",
                grid_query_idx_list=object_array(grid_query_list),
                context_idx_list=object_array(context_list),
                block_id=np.asarray(block_ids, dtype=np.int64),
                block_center_grid_idx=np.asarray(center_ids, dtype=np.int64),
                anchor_grid_idx_list=object_array(anchor_list),
            )
            print("train_regular_holdout_query_context samples:", len(grid_query_list))
            summary["train_holdout_samples"] = len(grid_query_list)
        else:
            # ---- Legacy anchor-based training pools ----
            if args.train_trusted_source == "all":
                trusted_idx = np.arange(coord_obs.shape[0], dtype=np.int64)
            else:
                trusted_idx = trusted_from_mask
            if trusted_idx.size == 0:
                raise ValueError("no trusted training observations are available")

            print("building train patches")
            print("train trusted source:", args.train_trusted_source, "count:", trusted_idx.size)
            train_pack = precompute_train_patches_2d(
                coord_obs_norm=coord_obs_norm,
                trace_obs=trace_obs,
                trusted_idx=trusted_idx,
                num_anchors=num_anchors,
                k_patch=k_patch,
                top_l=top_l,
                metric_weights=metric_weights,
                beta=args.beta,
                anchor_selector=args.train_anchor_selector,
                facility_nearest_l=top_l,
                value_local_top_l=value_local_top_l,
                value_suppression=args.value_suppression,
                value_suppression_lambda=args.value_suppression_lambda,
                value_score_tol=args.value_score_tol,
                value_knn_use_gpu=args.train_knn_use_gpu,
                value_knn_gpu_batch_rows=args.value_knn_gpu_batch_rows,
                value_knn_gpu_device=args.value_knn_gpu_device,
                value_knn_full_matrix_max_n=args.value_knn_full_matrix_max_n,
                value_suppression_use_gpu=args.train_suppression_use_gpu,
                num_query=args.num_query,
                seed=args.seed,
                pool_size=args.pool_size,
            )
            np.savez(
                patch_dir / "train_pool_idx_2d.npz",
                pool_idx_2d=train_pack["patch_idx_2d"],
                anchor_idx=train_pack["anchor_idx"],
            )
            if args.save_legacy_anchor_files:
                np.savez(patch_dir / "anchor_train_patch_idx_2d.npz", **{"0": train_pack["patch_idx_2d"]})
                np.savez(patch_dir / "anchor_train_context_idx_2d.npz", **{"0": train_pack["context_idx_2d"]})
                np.savez(patch_dir / "anchor_train_query_idx_2d.npz", **{"0": train_pack["query_idx_2d"]})
                np.save(patch_dir / "anchor_train_anchor_idx.npy", train_pack["anchor_idx"])
                np.save(patch_dir / "anchor_train_anchor_coord.npy", train_pack["anchor_coord"])
            print("train_pool_idx_2d:", train_pack["patch_idx_2d"].shape)
            summary["train_pool_shape"] = [int(x) for x in train_pack["patch_idx_2d"].shape]

    if not args.skip_infer:
        block_size = resolve_block_tuple(args.block_size, args.block_divisors, dims_4d, "block_size")
        stride = resolve_block_tuple(args.stride, args.stride_divisors, dims_4d, "stride")
        obs_valid_mask = None
        if args.infer_obs_valid_source == "regular_mask":
            obs_valid_mask = raw_obs_valid

        print("building infer patches")
        print("block_size:", block_size, "stride:", stride)
        print("query_mask_mode:", args.query_mask_mode)
        if infer_query_mask is not None:
            print("query targets:", int(infer_query_mask.sum()))
        print("infer obs_valid source:", args.infer_obs_valid_source)
        infer_pack = precompute_infer_patches_4d(
            coord_obs_norm=coord_obs_norm,
            coord_grid_norm=coord_grid_norm,
            grid_shape_4d=None,
            block_size=block_size,
            stride=stride,
            k_patch=k_patch,
            top_l=infer_top_l,
            metric_weights=metric_weights,
            beta=args.beta,
            grid_query_mask=infer_query_mask,
            require_full_query_coverage=args.require_full_query_coverage,
            grid_index_map_4d=grid_index_map_4d,
            max_query_per_patch=args.max_query_per_patch,
            greedy_fill_uncovered=args.greedy_fill_uncovered,
            obs_valid_mask=obs_valid_mask,
            use_gpu=args.infer_use_gpu,
            gpu_device=args.infer_gpu_device,
            gpu_query_chunk_size=args.gpu_query_chunk_size,
        )
        query_list = infer_pack["grid_query_idx_list"]
        context_list = infer_pack["patch_idx_list"]
        input_missing_ratio = summarize_query_context(query_list, context_list)
        np.savez(
            patch_dir / "infer_query_context.npz",
            grid_query_idx_list=object_array(query_list),
            context_idx_list=object_array(context_list),
            block_id=infer_pack["block_id"],
            block_center_grid_idx=infer_pack["block_center_grid_idx"],
            anchor_grid_idx_list=object_array(infer_pack["anchor_grid_idx_list"]),
        )
        np.savez(
            patch_dir / "infer_query_context_stats.npz",
            patch_query_count=infer_pack["patch_query_count"],
            patch_context_count=infer_pack["patch_context_count"],
            patch_input_count=infer_pack["patch_input_count"],
            patch_query_ratio=infer_pack["patch_query_ratio"],
            input_missing_ratio=input_missing_ratio,
        )
        print("infer_query_context samples:", len(query_list))
        if len(query_list):
            print(
                "query count min/max/mean:",
                int(infer_pack["patch_query_count"].min()),
                int(infer_pack["patch_query_count"].max()),
                float(infer_pack["patch_query_count"].mean()),
            )
            print(
                "input missing ratio min/max/mean:",
                float(input_missing_ratio.min()),
                float(input_missing_ratio.max()),
                float(input_missing_ratio.mean()),
            )
        summary["infer_samples"] = int(len(query_list))
        summary["block_size"] = [int(x) for x in block_size]
        summary["stride"] = [int(x) for x in stride]
        summary["query_mask_mode"] = args.query_mask_mode

    validation = validate_outputs(
        patch_dir=patch_dir,
        n_obs=coord_obs.shape[0],
        n_grid=coord_grid.shape[0],
        check_train=not args.skip_train and args.train_mode != "block",
        check_infer=not args.skip_infer,
    )
    summary["validation"] = validation
    print("validation:", validation)

    summary_path = Path(args.summary_json).expanduser().resolve() if args.summary_json else patch_dir / "precompute_anchor_patch_v2_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("summary_json:", summary_path)
    print("All checks passed.")


if __name__ == "__main__":
    main()
