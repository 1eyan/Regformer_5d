"""
Extract observation system from a SEG-Y file and plot shot/receiver geometry.

Byte positions come from the active segy_config preset (same mechanism as
mk_irr_mask.py).  No standalone JSON file needed.

Usage:
    INPUT_SGY=/path/to/data.sgy SEGY_CONFIG=a002 python utils/obs_system.py
    INPUT_SGY=/path/to/data.sgy SEGY_CONFIG=field1031 python utils/obs_system.py

Output:
    - terminal:  shot/receiver count, line/stake ranges, fold estimate
    - PNG file:  {stem}_obs_system.png  (next to the SEG-Y file)
"""

import os
import struct
import logging
import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is on sys.path (so "from config import segy_config" works)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import segy_config

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', force=True)
log = logging.getLogger("obs_system")

# Colour palette
C_SHOT = "#2c7bb6"
C_RECV = "#d7191c"

# Shot / receiver identity fields — priority: line+stake > grid > float
SHOT_LINE_STAKE = ("shot_line", "shot_no")
RECV_LINE_STAKE = ("recv_line", "recv_no")
SHOT_GRID_FIELDS = ("shot_x_grid", "shot_y_grid")
RECV_GRID_FIELDS = ("recv_x_grid", "recv_y_grid")
SHOT_FLOAT_FIELDS = ("shot_x", "shot_y")
RECV_FLOAT_FIELDS = ("rec_x", "rec_y")


# ---------------------------------------------------------------------------
# Direct SEG-Y trace header reader (no seisio dependency)
# ---------------------------------------------------------------------------

# struct format mapping: short-type → (nbytes, struct_fmt)
_HDR_FMTS = {
    "i": (4, ">i"),
    "f": (4, ">f"),
    "h": (2, ">h"),
    "H": (2, ">H"),
}


def _read_headers_direct(path: str) -> tuple[dict[str, np.ndarray], int, int]:
    """Read all trace headers from a SEG-Y file using raw byte positions.

    Returns
    -------
    headall : dict[str, np.ndarray]
        One 1-D array per field in the active ``byte_pos`` config.
    nt : int
        Number of traces.
    ns : int
        Number of samples per trace (from binary header).
    """
    byte_pos = segy_config.get_byte_pos()
    types = segy_config.get_thdef_types()

    with open(path, "rb") as f:
        # --- binary file header ---
        f.seek(3200)
        bin_header = f.read(400)
        ns = struct.unpack(">H", bin_header[20:22])[0]
        fmt_code = struct.unpack(">H", bin_header[24:26])[0]
        if fmt_code in (1, 2, 5):
            bps = 4
        elif fmt_code == 3:
            bps = 2
        else:
            bps = 1

        # --- read all 240-byte trace headers ---
        f.seek(3600)
        raw_headers: list[bytes] = []
        while True:
            hdr = f.read(240)
            if len(hdr) < 240:
                break
            raw_headers.append(hdr)
            ns_trace = struct.unpack(">H", hdr[114:116])[0] or ns
            f.seek(int(ns_trace) * bps, os.SEEK_CUR)

    nt = len(raw_headers)

    if nt == 0:
        return {}, 0, ns

    # Concatenate all headers → single byte buffer for efficient strided access
    buf = b"".join(raw_headers)

    headall = {}
    for field, pos_1b in byte_pos.items():
        tipo = types.get(field, "i")
        size, fmt_str = _HDR_FMTS.get(tipo, (4, ">i"))
        dtype = np.dtype(fmt_str)
        vals = np.empty(nt, dtype=dtype)
        for i in range(nt):
            offset = i * 240 + pos_1b - 1
            vals[i] = struct.unpack_from(fmt_str, buf, offset)[0]
        headall[field] = vals

    return headall, nt, ns


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def _describe(name: str, xy: np.ndarray, ids: np.ndarray | None = None) -> dict:
    """Return summary stats for a set of 2D points, optionally with line/stake IDs."""
    if len(xy) == 0:
        return {"name": name, "count": 0}
    xs, ys = xy[:, 0], xy[:, 1]
    s = {
        "name": name,
        "count": len(xy),
        "x_range": (int(xs.min()), int(xs.max())),
        "y_range": (int(ys.min()), int(ys.max())),
        "x_unique": int(np.unique(xs).size),
        "y_unique": int(np.unique(ys).size),
    }
    if ids is not None:
        s["line_min"] = int(ids[:, 0].min())
        s["line_max"] = int(ids[:, 0].max())
        s["stake_min"] = int(ids[:, 1].min())
        s["stake_max"] = int(ids[:, 1].max())
        s["n_lines"] = int(np.unique(ids[:, 0]).size)
        s["n_stakes"] = int(np.unique(ids[:, 1]).size)
    return s


def _print_stats(stats: dict):
    """Print a formatted stats block."""
    print(f"\n{'=' * 55}")
    print(f"  {stats['name']}")
    print(f"{'=' * 55}")
    print(f"  Unique points : {stats['count']}")
    if stats["count"] == 0:
        return
    print(f"  X range       : {stats['x_range'][0]:>12d}  — {stats['x_range'][1]:>12d}")
    print(f"  Y range       : {stats['y_range'][0]:>12d}  — {stats['y_range'][1]:>12d}")
    if "n_lines" in stats:
        print(f"  Lines range   : {stats['n_lines']:>4d}  ({stats['line_min']} — {stats['line_max']})")
        print(f"  Stakes range  : {stats['n_stakes']:>4d}  ({stats['stake_min']} — {stats['stake_max']})")
    print(f"  Unique X vals : {stats['x_unique']}")
    print(f"  Unique Y vals : {stats['y_unique']}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_obs(shot_xy: np.ndarray, rcv_xy: np.ndarray, stem: str, out_dir: str):
    """Three-panel observation-system figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Observation System — {stem}", fontsize=14, y=1.02)

    panels = [
        (0, "Shot Points",          shot_xy, C_SHOT, "^"),
        (1, "Receiver Points",      rcv_xy,  C_RECV, "."),
        (2, "Shot + Receiver",      None,    None,   None),
    ]

    for col, title, xy, color, marker in panels[:2]:
        ax = axes[col]
        ax.scatter(xy[:, 0], xy[:, 1], s=10, c=color, marker=marker,
                   alpha=0.7, edgecolors="none")
        ax.set_title(f"{title}  ({len(xy)})")
        ax.set_xlabel("Easting")
        ax.set_ylabel("Northing")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")

    # Overlay
    ax = axes[2]
    ax.scatter(shot_xy[:, 0], shot_xy[:, 1], s=18, c=C_SHOT, marker="^",
               alpha=0.8, label=f"Shots ({len(shot_xy)})")
    ax.scatter(rcv_xy[:, 0],  rcv_xy[:, 1],  s=8,  c=C_RECV, marker=".",
               alpha=0.6, label=f"Receivers ({len(rcv_xy)})")
    ax.set_title("Overlay")
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.legend(markerscale=1.5)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{stem}_obs_system.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("Saved plot → %s", out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract and plot observation system from SEG-Y")
    parser.add_argument("input_sgy", nargs="?", default=os.environ.get("INPUT_SGY", ""),
                        help="Path to input SEG-Y (or set $INPUT_SGY)")
    parser.add_argument("--preset", default=os.environ.get("SEGY_CONFIG", "a002"),
                        help="SEGY config preset (default: a002)")
    parser.add_argument("--outdir", default="",
                        help="Output directory for PNG (default: same dir as SEG-Y)")
    args = parser.parse_args()

    if not args.input_sgy:
        log.error("No input SEG-Y. Pass path or set INPUT_SGY env var.")
        return

    # ---- Load config ----
    try:
        segy_config.load_config(args.preset)
    except ValueError as e:
        log.error(e)
        return

    bp = segy_config.get_byte_pos()
    log.info("Preset: %s  |  byte_pos keys: %s", segy_config.get_active_name(), sorted(bp.keys()))

    # Determine best available source for identity and coordinates
    has_line_stake = all(f in bp for f in SHOT_LINE_STAKE + RECV_LINE_STAKE)
    has_grid = all(f in bp for f in SHOT_GRID_FIELDS + RECV_GRID_FIELDS)
    has_float = all(f in bp for f in SHOT_FLOAT_FIELDS + RECV_FLOAT_FIELDS)

    if has_grid:
        coord_fields_shot, coord_fields_recv = SHOT_GRID_FIELDS, RECV_GRID_FIELDS
        coord_label = "grid int32"
    elif has_float:
        coord_fields_shot, coord_fields_recv = SHOT_FLOAT_FIELDS, RECV_FLOAT_FIELDS
        coord_label = "float (scalar-corrected)"
    else:
        log.error("No coordinate fields found in preset. Need shot_x/shot_y or shot_x_grid/shot_y_grid")
        return

    log.info("Identity: %s  |  Coords: %s",
             "line+stake" if has_line_stake else coord_label, coord_label)

    # ---- Read all trace headers directly (no seisio) ----
    headall, ntraces, nsamples = _read_headers_direct(args.input_sgy)
    log.info("Traces=%d  Samples=%d", ntraces, nsamples)

    # ---- Extract coordinates ----
    def _get_coords(fields):
        return np.column_stack([headall[f] for f in fields])

    shot_coords = _get_coords(coord_fields_shot)
    recv_coords = _get_coords(coord_fields_recv)

    # Scalar correction for float coords (only a002 preset has the scalar field)
    if not has_grid and has_float and "scalar" in headall:
        scalar = headall["scalar"].copy()
        scalar[scalar == 0] = 1
        shot_coords = np.round(shot_coords / np.abs(scalar[:, None])).astype(np.int32)
        recv_coords = np.round(recv_coords / np.abs(scalar[:, None])).astype(np.int32)

    # ---- Identity (dedup) ----
    if has_line_stake:
        shot_ids = _get_coords(SHOT_LINE_STAKE)
        recv_ids = _get_coords(RECV_LINE_STAKE)

        def _unique_with_median(id_arr, coord_arr):
            uniq_ids, inv = np.unique(id_arr, axis=0, return_inverse=True)
            median = np.zeros((len(uniq_ids), 2), dtype=np.int32)
            for i in range(len(uniq_ids)):
                median[i] = np.median(coord_arr[inv == i], axis=0).astype(np.int32)
            return uniq_ids, median

        shot_id_unique, shot_xy = _unique_with_median(shot_ids, shot_coords)
        recv_id_unique, recv_xy = _unique_with_median(recv_ids, recv_coords)
        shot_stats = _describe("SHOT POINTS", shot_xy, shot_id_unique)
        recv_stats = _describe("RECEIVER POINTS", recv_xy, recv_id_unique)
    else:
        shot_xy = np.unique(shot_coords, axis=0)
        recv_xy = np.unique(recv_coords, axis=0)
        shot_stats = _describe("SHOT POINTS", shot_xy)
        recv_stats = _describe("RECEIVER POINTS", recv_xy)

    # ---- Report ----
    _print_stats(shot_stats)
    _print_stats(recv_stats)

    # Fold estimate
    n_shots = shot_stats["count"]
    n_recvs = recv_stats["count"]
    if n_shots > 0 and n_recvs > 0:
        avg_fold = ntraces / (n_shots + n_recvs)
        print(f"\n  {'─' * 55}")
        print(f"  Estimated avg. fold  : {avg_fold:.1f}  traces / (shots+recvs)")
        print(f"  Nominal fold (CMP)   : {ntraces / (n_shots * n_recvs):.1f}  (if full grid)")
        print(f"  {'─' * 55}")

    # ---- Plot ----
    out_dir = args.outdir or os.path.dirname(os.path.abspath(args.input_sgy))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input_sgy))[0]
    _plot_obs(shot_xy, recv_xy, stem, out_dir)

    print(f"\nDone.  Plot → {out_dir}/{stem}_obs_system.png\n")


if __name__ == "__main__":
    main()
