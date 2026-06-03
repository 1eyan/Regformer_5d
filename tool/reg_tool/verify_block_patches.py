#!/usr/bin/env python3
"""Verify block-based training patches align with inference patches."""

import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Verify block-based training pool alignment with inference patches."
    )
    parser.add_argument("patch_dir", type=str, help="Directory containing precomputed npz files")
    parser.add_argument(
        "--check-samples",
        type=int,
        default=20,
        help="Number of inference samples to check for Jaccard alignment",
    )
    parser.add_argument(
        "--suggest-query-count",
        action="store_true",
        help="Print only the suggested train_num_query (machine-readable)",
    )
    args = parser.parse_args()

    p = Path(args.patch_dir)

    # Load training pools
    train_path = p / "train_pool_idx_2d_block.npz"
    if not train_path.exists():
        print(f"ERROR: {train_path} not found")
        return 1

    train = np.load(train_path)
    pool_idx = train["pool_idx_2d"]
    train_block_ids = train["block_id"]

    # Build bid -> list of obs sets (one block may have multiple chunked pools)
    train_by_bid: dict = {}
    for i in range(len(pool_idx)):
        bid = int(train_block_ids[i])
        s = set(pool_idx[i][pool_idx[i] >= 0])
        if bid not in train_by_bid:
            train_by_bid[bid] = []
        train_by_bid[bid].append(s)

    # Load inference patches
    infer_path = p / "infer_query_context.npz"
    if not infer_path.exists():
        print(f"ERROR: {infer_path} not found")
        return 1

    infer = np.load(infer_path, allow_pickle=True)
    ctx_list = infer["context_idx_list"]
    block_ids = infer["block_id"]

    # Load inference stats
    stats_path = p / "infer_query_context_stats.npz"
    if not stats_path.exists():
        print(f"ERROR: {stats_path} not found")
        return 1

    stats = np.load(stats_path)

    n_pools = len(pool_idx)
    n_infer = len(ctx_list)

    if args.suggest_query_count:
        avg_missing = stats["patch_query_count"].mean()
        print(int(round(avg_missing * 0.7)))
        return 0

    print("=" * 60)
    print("Block Patch Alignment Verification")
    print("=" * 60)

    # 1. Basic stats
    pool_sizes = [len(r[r >= 0]) for r in pool_idx]
    n_unique_blocks = len(train_by_bid)
    pools_per_block = n_pools / max(1, n_unique_blocks)
    print(f"Train pools:              {n_pools} (from {n_unique_blocks} unique blocks, {pools_per_block:.1f} pools/block)")
    print(
        f"Train pool size:          mean={np.mean(pool_sizes):.1f}, "
        f"std={np.std(pool_sizes):.1f}, min={np.min(pool_sizes)}, max={np.max(pool_sizes)}"
    )
    print(f"Infer samples:            {n_infer}")
    print(f"Infer query count:        mean={stats['patch_query_count'].mean():.1f}")
    print(f"Infer context count:      mean={stats['patch_context_count'].mean():.1f}")
    print(f"Infer missing ratio:      mean={stats['patch_query_ratio'].mean():.3f}")

    # 2. Recall alignment (same block_id)
    # Recall = |infer_context ∩ train_pool| / |infer_context|
    # This is the key metric: how much of the inference context is covered by train pool.
    recalls = []
    train_only_ratios = []
    infer_only_ratios = []
    matched_count = 0
    unmatched_bids = []
    for i in range(min(args.check_samples, n_infer)):
        bid = int(block_ids[i])
        train_set_list = train_by_bid.get(bid)
        if train_set_list is None:
            unmatched_bids.append(bid)
            continue
        matched_count += 1
        ctx = set(np.asarray(ctx_list[i], dtype=np.int64).reshape(-1))
        # Union all training pools for this block (model sees all of them during training)
        train_set = set().union(*train_set_list)
        inter = len(train_set & ctx)
        if len(ctx) > 0:
            recalls.append(inter / len(ctx))
        train_only = len(train_set - ctx)
        infer_only = len(ctx - train_set)
        total_union = len(train_set | ctx)
        if total_union > 0:
            train_only_ratios.append(train_only / total_union)
            infer_only_ratios.append(infer_only / total_union)

    if recalls:
        print(f"\nRecall (first {args.check_samples} checked, {matched_count} matched):")
        print(
            f"  mean={np.mean(recalls):.3f}, "
            f"min={np.min(recalls):.3f}, "
            f"max={np.max(recalls):.3f}"
        )
        print(
            f"  Train-only ratio:  mean={np.mean(train_only_ratios):.3f} "
            f"(obs in train pool but not infer context)"
        )
        print(
            f"  Infer-only ratio:  mean={np.mean(infer_only_ratios):.3f} "
            f"(obs in infer context but not train pool ← FALSE NEGATIVE)"
        )
        if np.mean(recalls) < 0.7:
            print("  WARNING: Low recall! Inference context is not well covered by train pool.")
            print("  Likely cause: stride==block_size (no overlap) + edge queries need context from adjacent blocks.")
            print("  Fix: increase --block-pool-margin (e.g., 1 or 2) in precompute.")
        elif np.mean(recalls) > 0.9:
            print("  OK: Strong coverage between train pool and infer context.")
    else:
        print("\nWARNING: No overlapping block_id found for recall check.")

    if unmatched_bids:
        n_uniq = len(set(unmatched_bids))
        print(
            f"\nNOTE: {len(unmatched_bids)} infer samples ({n_uniq} unique blocks) "
            f"had no matching train pool (filtered by --min-obs-per-block)."
        )

    # 3. Suggest train_num_query
    avg_missing = stats["patch_query_count"].mean()
    conservative = int(round(avg_missing * 0.7))
    aggressive = int(round(avg_missing))
    print(f"\nSuggested train_num_query:")
    print(f"  Conservative (0.7x):    {conservative}")
    print(f"  Aggressive (1.0x):      {aggressive}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
