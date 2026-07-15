#!/usr/bin/env python3
"""
Dataset harvester for NN image-restoration training (denoise / AWB / sharpness).

Design contract (agreed):
  - Produces CLEAN, CONSISTENTLY-ENCODED TARGET PATCHES only.
  - Reference white balance, two tiers:
      * Default:      camera as-shot WB ("silver" tier -- consistent, unverified).
      * --card mode:  WB measured from an 18% gray card ("gold" tier).
        The card is measured in CAMERA-NATIVE space (unity WB, no color
        matrix) and the derived multipliers are applied by libraw on the
        RAW channels via user_wb -- i.e. BEFORE the camera->sRGB matrix,
        the physically correct stage. Measure pre-matrix, apply pre-matrix.
  - NO gray-world or any other *estimated* WB is ever baked into targets.
  - Output: linear sRGB (D65), float16, 0.0-1.0, unclipped. Stored as .npy.
  - Sampling: stratified (detail / mid / flat) with noise gating and
    non-overlap NMS. Card regions are excluded from patch selection.

Card mode usage:
  Single frame, inline region (x,y,w,h in pixels, x = horizontal):
      python dataset_harvester.py IMG_0001.CR3 --card --region 1024,3456,350,348

  Session directory with a region file (one card frame per lighting change):
      python dataset_harvester.py E:/cr3/session1 --card --region E:/cr3/data_file.txt

  Region file format, one line per CARD frame (fill manually):
      <CR3 image name>, <x>, <y>, <width>, <height>
  e.g.:
      IMG_0001.CR3, 1024, 3456, 350, 348
      IMG_0057.CR3, 2200, 1800, 400, 400

  Session inheritance convention: files are processed in SORTED NAME ORDER.
  A frame not listed in the region file inherits the multipliers of the most
  recent card frame preceding it in that order. Shoot the card FIRST in each
  lighting condition and let file numbering do the rest. Frames before any
  card frame fall back to as-shot WB (with a console warning).

  The script writes <output>/data_file_out.txt: the region file content with
  an addendum per line -- measured mean R, G, B of the card (linear camera
  space, 0-1), the derived gains, and validity flags.

Manual slice mode (automatic, no option needed):
  The script ALWAYS looks for a file named  data_harvest.txt  in the input
  folder (the folder given on the command line, or the parent folder of a
  single given CR3 file) and reports in the terminal whether it was found.
  If found, its slices are harvested as an ADDENDUM to the automatic
  harvesting -- manual slices never replace or reduce the automatic
  selection; the automatic picks merely avoid overlapping them.
  File format, one line per manual slice (a frame may appear on several
  lines):
      <CR3 image name>, <x>, <y>
  where (x, y) is the TOP-LEFT corner of the slice, x = horizontal, in
  full-frame pixels; slice size = --size. Manual slices bypass the quality
  gates (the user has decided), but their quality metrics (sharpness, std,
  noise, brightness) are measured and written into the same JSON report as
  automatic patches, with stratum "manual" plus gate flags if a metric is
  out of the automatic-selection limits; -png previews are written for
  them exactly like for automatic patches. If a point is out of frame (too
  low / too far right so a full slice cannot be clipped), the point is
  IGNORED with a terminal warning.

Platform contract: Windows 10/11 x64, Python 3.12 (64-bit).
Requires: numpy, rawpy, Pillow (Pillow only for -png previews).
All available as prebuilt win_amd64 wheels: pip install numpy rawpy pillow
No OpenCV / scipy / torch or other heavy frameworks are used.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import rawpy

# --- CONFIGURATION -----------------------------------------------------------
PATCHES_PER_STRATUM = {"detail": 3, "mid": 2, "flat": 2}

# Reject patches darker than this mean linear value (~6% of full scale).
MIN_BRIGHTNESS = 0.06

# Reject patches whose estimated noise sigma (linear, 0-1) exceeds this.
# Tune against a known-clean base-ISO frame of your camera.
MAX_NOISE_SIGMA = 0.004

# --- Absolute stratum eligibility gates (linear 0-1 units) -------------------
# Strata terciles are RELATIVE to each frame; these gates are ABSOLUTE, so a
# stratum may legitimately come up EMPTY on a frame lacking suitable content
# (e.g. a shallow-DOF close-up has no true flat surfaces). That is intended.
#
# 'flat' must be STATISTICALLY flat: low overall std rejects defocused texture
# (bokeh fur/skin), which is edge-free but full of shading gradients. Blur also
# suppresses noise, so without this gate the lowest-noise ranking would
# actively PREFER missed-focus regions.
FLAT_MAX_STD = 0.015
# 'detail' must be genuinely sharp in absolute terms, so a fully defocused
# frame cannot promote blur into the detail stratum via relative terciles.
# Calibrate: run on one known in-focus frame and one missed-focus frame,
# compare reported sharpness values, set the floor between them.
DETAIL_MIN_SHARPNESS = 1e-5

# Card-measurement validity limits (linear 0-1, camera space):
CARD_CLIP_LEVEL = 0.98    # any channel mean above this -> clipped, untrusted
CARD_FLOOR_LEVEL = 0.02   # any channel mean below this -> noise floor, untrusted
CARD_EDGE_TRIM = 0.20     # fraction of region trimmed on each side before averaging

# Manual-slice file, auto-discovered in the input folder (no CLI option):
MANUAL_FILE_NAME = "data_harvest.txt"

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)  # Rec.709, linear

IMMERKAER_KERNEL = np.array([[ 1, -2,  1],
                             [-2,  4, -2],
                             [ 1, -2,  1]], dtype=np.float32)

LAPLACE_KERNEL = np.array([[0,  1, 0],
                           [1, -4, 1],
                           [0,  1, 0]], dtype=np.float32)


# --- METRICS -----------------------------------------------------------------

def conv3x3(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    3x3 convolution in pure NumPy, replacing scipy.ndimage.convolve to keep
    dependencies minimal. np.pad mode="symmetric" == scipy's "reflect"
    boundary (edge pixel duplicated). Both kernels used here are 180-degree
    symmetric, so convolution and correlation coincide.
    """
    p = np.pad(img, 1, mode="symmetric")
    out = np.zeros(img.shape, dtype=np.float32)
    hh, ww = img.shape
    for i in range(3):
        for j in range(3):
            k = kernel[i, j]
            if k != 0.0:
                out += k * p[i:i + hh, j:j + ww]
    return out


def estimate_noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer fast noise sigma estimate on a grayscale patch (linear)."""
    h, w = gray.shape
    resp = conv3x3(gray, IMMERKAER_KERNEL)
    return float(math.sqrt(math.pi / 2.0) * np.abs(resp).sum()
                 / (6.0 * (w - 2) * (h - 2)))


def sharpness_metric(gray: np.ndarray) -> float:
    """Laplacian variance. Only meaningful AFTER the noise gate has passed."""
    return float(conv3x3(gray, LAPLACE_KERNEL).var())


# --- GRAY-CARD MEASUREMENT ---------------------------------------------------

def measure_card(cr3_path: Path, region: tuple) -> dict:
    """
    Measure the illuminant from an 18% gray card region.

    Decodes the frame in CAMERA-NATIVE space: unity WB multipliers, no
    camera->sRGB matrix (output_color=raw), linear, no auto-brightening.
    Whatever channel imbalance the card shows there is purely the
    illuminant's fingerprint on this sensor.

    region: (x, y, w, h) in full-frame pixel coordinates.
    Returns dict with means, gains, user_wb multipliers, validity flags.
    """
    x, y, w, h = region
    with rawpy.imread(str(cr3_path)) as raw:
        cam = raw.postprocess(
            gamma=(1, 1),
            no_auto_bright=True,
            use_camera_wb=False,
            user_wb=[1.0, 1.0, 1.0, 1.0],          # unity: measure, don't correct
            output_color=rawpy.ColorSpace.raw,     # NO color matrix
            output_bps=16,
        ).astype(np.float32) / 65535.0

    if not (0 <= y and y + h <= cam.shape[0] and 0 <= x and x + w <= cam.shape[1]):
        raise ValueError(f"card region {region} outside frame {cam.shape[1]}x{cam.shape[0]}")

    # Trim the region border to dodge edge effects / glare gradients.
    ty, tx = int(h * CARD_EDGE_TRIM), int(w * CARD_EDGE_TRIM)
    core = cam[y + ty : y + h - ty, x + tx : x + w - tx]
    if core.size == 0:
        raise ValueError(f"card region {region} too small after edge trim")

    mean_r = float(core[:, :, 0].mean())
    mean_g = float(core[:, :, 1].mean())
    mean_b = float(core[:, :, 2].mean())

    flags = []
    if max(mean_r, mean_g, mean_b) > CARD_CLIP_LEVEL:
        flags.append("CLIPPED")
    if min(mean_r, mean_g, mean_b) < CARD_FLOOR_LEVEL:
        flags.append("NEAR_NOISE_FLOOR")
    if min(mean_r, mean_b) <= 1e-6:
        raise ValueError("card region has a near-zero channel; cannot derive gains")

    gain_r = mean_g / mean_r
    gain_b = mean_g / mean_b

    return {
        "card_frame": cr3_path.name,
        "region_xywh": [x, y, w, h],
        "mean_rgb_camera_space": [round(mean_r, 6), round(mean_g, 6), round(mean_b, 6)],
        "gain_r": round(gain_r, 6),
        "gain_b": round(gain_b, 6),
        # libraw user_wb order: (R, G1, B, G2)
        "user_wb": [gain_r, 1.0, gain_b, 1.0],
        "flags": flags,
    }


# --- CORE --------------------------------------------------------------------

def load_linear_rgb(cr3_path: Path, user_wb=None) -> np.ndarray:
    """
    RAW -> linear sRGB float32 in 0..1.

    user_wb=None : camera as-shot WB (consistent 'silver' reference).
    user_wb=[...] : measured card multipliers, applied by libraw on the RAW
                    channels BEFORE the color matrix ('gold' reference).
    gamma=(1,1), no_auto_bright: light stays linear, no hidden tone curve.
    """
    kwargs = dict(gamma=(1, 1), no_auto_bright=True, output_bps=16)
    if user_wb is not None:
        kwargs.update(use_camera_wb=False, user_wb=list(user_wb))
    else:
        kwargs.update(use_camera_wb=True)
    with rawpy.imread(str(cr3_path)) as raw:
        rgb16 = raw.postprocess(**kwargs)
    return rgb16.astype(np.float32) / 65535.0


def rects_overlap(y, x, size, region_xywh) -> bool:
    """True if patch (y, x, size x size) intersects card region (x, y, w, h)."""
    rx, ry, rw, rh = region_xywh
    return not (x + size <= rx or rx + rw <= x or y + size <= ry or ry + rh <= y)


def collect_candidates(rgb, gray, patch_size, exclude_region=None):
    """Scan on a half-patch stride; gate by brightness and noise;
    skip anything touching the card region."""
    stride = patch_size // 2
    candidates = []
    for y in range(0, rgb.shape[0] - patch_size + 1, stride):
        for x in range(0, rgb.shape[1] - patch_size + 1, stride):
            if exclude_region and rects_overlap(y, x, patch_size, exclude_region):
                continue
            p_gray = gray[y:y + patch_size, x:x + patch_size]

            brightness = float(p_gray.mean())
            if brightness < MIN_BRIGHTNESS:
                continue
            sigma = estimate_noise_sigma(p_gray)
            if sigma > MAX_NOISE_SIGMA:
                continue  # too noisy to serve as a CLEAN target
            candidates.append({
                "y": y, "x": x,
                "brightness": brightness,
                "std": float(p_gray.std()),
                "noise_sigma": sigma,
                "sharpness": sharpness_metric(p_gray),
            })
    return candidates


def nms_select(sorted_cands, k, min_dist, taken):
    """Greedy pick up to k candidates at least min_dist apart from ALL taken."""
    picked = []
    for c in sorted_cands:
        if len(picked) == k:
            break
        if all(max(abs(c["y"] - t["y"]), abs(c["x"] - t["x"])) >= min_dist
               for t in taken):
            picked.append(c)
            taken.append(c)
    return picked


def stratified_select(candidates, patch_size, initial_taken=None):
    """detail / mid / flat by sharpness terciles + ABSOLUTE eligibility gates
    + non-overlap NMS. Strata may return fewer patches than their quota (or
    none) when the frame lacks suitable content -- intended behavior.
    Flat stratum ranked by LOWEST noise among statistically flat patches.
    initial_taken: positions (e.g. manual slices) automatic picks must avoid."""
    if not candidates:
        return []
    sharp = np.array([c["sharpness"] for c in candidates])
    t_low, t_high = np.percentile(sharp, [33.3, 66.6])
    strata = {
        "detail": sorted((c for c in candidates
                          if c["sharpness"] >= max(t_high, DETAIL_MIN_SHARPNESS)),
                         key=lambda c: c["sharpness"], reverse=True),
        "mid":    sorted((c for c in candidates
                          if t_low <= c["sharpness"] < t_high),
                         key=lambda c: c["sharpness"], reverse=True),
        "flat":   sorted((c for c in candidates
                          if c["sharpness"] < t_low and c["std"] <= FLAT_MAX_STD),
                         key=lambda c: c["noise_sigma"]),
    }
    taken = list(initial_taken) if initial_taken else []
    selection = []
    for name, cands in strata.items():
        picked = nms_select(cands, PATCHES_PER_STRATUM[name], patch_size, taken)
        if len(picked) < PATCHES_PER_STRATUM[name]:
            print(f"  [INFO] stratum '{name}': {len(picked)}/"
                  f"{PATCHES_PER_STRATUM[name]} -- frame lacks eligible content")
        for p in picked:
            p["stratum"] = name
            selection.append(p)
    return selection


def save_preview_png(patch_lin: np.ndarray, path: Path):
    """8-bit sRGB-encoded PNG for human eyeballing only (exact piecewise OETF)."""
    from PIL import Image
    a = np.clip(patch_lin, 0.0, 1.0)
    srgb = np.where(a <= 0.0031308, 12.92 * a, 1.055 * np.power(a, 1 / 2.4) - 0.055)
    Image.fromarray((srgb * 255.0 + 0.5).astype(np.uint8)).save(path)


def measure_manual_slices(rgb, gray, points, patch_size, frame_name):
    """
    Validate and measure user-chosen slices. Out-of-frame points are ignored
    with a terminal warning. Quality gates are NOT applied (user's decision),
    but out-of-limit metrics are flagged in the report.
    """
    H, W = gray.shape
    manual = []
    for (x, y) in points:
        if not (0 <= x and x + patch_size <= W and 0 <= y and y + patch_size <= H):
            print(f"  [WARN] {frame_name}: manual point (x={x}, y={y}) ignored -- "
                  f"a {patch_size}x{patch_size} slice does not fit the "
                  f"{W}x{H} frame")
            continue
        p_gray = gray[y:y + patch_size, x:x + patch_size]
        entry = {
            "y": y, "x": x,
            "brightness": float(p_gray.mean()),
            "std": float(p_gray.std()),
            "noise_sigma": estimate_noise_sigma(p_gray),
            "sharpness": sharpness_metric(p_gray),
            "stratum": "manual",
        }
        flags = []
        if entry["brightness"] < MIN_BRIGHTNESS:
            flags.append("DARK")
        if entry["noise_sigma"] > MAX_NOISE_SIGMA:
            flags.append("NOISY")
        entry["gate_flags"] = flags
        if flags:
            print(f"  [WARN] {frame_name}: manual slice (x={x}, y={y}) kept, "
                  f"but out of automatic limits: {'|'.join(flags)}")
        manual.append(entry)
    return manual


def analyze_and_harvest(cr3_path: Path, output_dir: Path, save_png: bool,
                        patch_size: int, wb: dict, manual_points=None):
    """
    wb: {"source": "as_shot" | "card_measured" | "card_inherited",
         "measurement": card dict or None}
    """
    meas = wb["measurement"]
    print(f"Harvesting: {cr3_path.name} ({patch_size}x{patch_size}) "
          f"[WB: {wb['source']}]...")

    gt_dir = output_dir / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    try:
        user_wb = meas["user_wb"] if meas else None
        rgb = load_linear_rgb(cr3_path, user_wb=user_wb)
    except Exception as e:
        print(f"  [ERROR] {e}")
        return

    gray = rgb @ LUMA  # linear luma

    exclude = None
    if wb["source"] == "card_measured" and meas["card_frame"] == cr3_path.name:
        exclude = meas["region_xywh"]  # the card itself must not become a patch

    frame_noise = estimate_noise_sigma(gray)

    manual = measure_manual_slices(rgb, gray, manual_points or [],
                                   patch_size, cr3_path.name)

    candidates = collect_candidates(rgb, gray, patch_size, exclude_region=exclude)
    selection = manual + stratified_select(candidates, patch_size,
                                           initial_taken=manual)

    report = {
        "source": cr3_path.name,
        "reference_wb": {
            "tier": "gold" if meas else "silver",
            "source": wb["source"],
            "card": meas,   # full measurement incl. R,G,B means, gains, flags
        },
        "encoding": "linear sRGB, float16, 0-1",
        "frame_noise_sigma": round(frame_noise, 6),
        "candidates_passed_gates": len(candidates),
        "patches": [],
    }

    for i, p in enumerate(selection):
        pid = f"{cr3_path.stem}_{patch_size}_{p['stratum']}_{i+1:02d}"
        patch = rgb[p["y"]:p["y"] + patch_size, p["x"]:p["x"] + patch_size]
        np.save(gt_dir / f"{pid}.npy", patch.astype(np.float16))
        if save_png:
            save_preview_png(patch, gt_dir / f"{pid}_preview.png")
        entry = {
            "id": pid, "y": p["y"], "x": p["x"], "stratum": p["stratum"],
            "sharpness": round(p["sharpness"], 6),
            "std": round(p["std"], 6),
            "noise_sigma": round(p["noise_sigma"], 6),
            "brightness": round(p["brightness"], 4),
        }
        if p.get("gate_flags"):
            entry["gate_flags"] = p["gate_flags"]
        report["patches"].append(entry)

    with open(output_dir / f"{cr3_path.stem}_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"  [OK] {len(selection)} patches "
          f"({', '.join(p['stratum'] for p in selection)})")


# --- REGION ARGUMENT HANDLING -------------------------------------------------

def parse_region_arg(region_arg: str):
    """
    Returns ("inline", (x, y, w, h))  for  --region 1024,3456,350,348
         or ("file", {name_lower: {"line": str, "region": (x,y,w,h)}}, order)
            for  --region path/to/data_file.txt
    """
    p = Path(region_arg)
    if p.is_file():
        mapping, order = {}, []
        with open(p, "r") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [s.strip() for s in line.split(",")]
                if len(parts) != 5:
                    raise ValueError(f"{p.name} line {ln}: expected "
                                     f"'<name>, <x>, <y>, <w>, <h>', got: {line}")
                name = parts[0]
                x, y, w, h = (int(v) for v in parts[1:])
                mapping[name.lower()] = {"line": line, "region": (x, y, w, h)}
                order.append(name.lower())
        if not mapping:
            raise ValueError(f"{p.name}: no region lines found")
        return "file", mapping, order

    parts = [s.strip() for s in region_arg.split(",")]
    if len(parts) != 4:
        raise ValueError("--region must be 'x,y,w,h' or a path to a region file")
    x, y, w, h = (int(v) for v in parts)
    return "inline", (x, y, w, h), None


def parse_manual_file(path_str: str):
    """
    Manual-slice file: lines '<CR3 name>, <x>, <y>'.
    Returns dict name_lower -> list of (x, y). A frame may appear on
    several lines (several manual slices).
    """
    p = Path(path_str)
    if not p.is_file():
        raise ValueError(f"manual slice file not found: {path_str}")
    mapping = {}
    with open(p, "r") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [s.strip() for s in line.split(",")]
            if len(parts) != 3:
                raise ValueError(f"{p.name} line {ln}: expected "
                                 f"'<name>, <x>, <y>', got: {line}")
            name = parts[0].lower()
            x, y = int(parts[1]), int(parts[2])
            mapping.setdefault(name, []).append((x, y))
    if not mapping:
        raise ValueError(f"{p.name}: no manual slice lines found")
    return mapping


def write_data_file_out(output_dir: Path, entries: list):
    """
    entries: list of dicts {line, measurement or error}.
    Writes data_file_out.txt: original line + addendum
      , R=<r>, G=<g>, B=<b>, gain_R=<gr>, gain_B=<gb>[, FLAGS=...]
    """
    out_path = output_dir / "data_file_out.txt"
    with open(out_path, "w") as f:
        f.write("# <CR3 image name>, <x>, <y>, <width>, <height>, "
                "R, G, B (linear camera space 0-1), gain_R, gain_B, flags\n")
        for e in entries:
            if "error" in e:
                f.write(f"{e['line']}, ERROR: {e['error']}\n")
                continue
            m = e["measurement"]
            r, g, b = m["mean_rgb_camera_space"]
            flags = (", FLAGS=" + "|".join(m["flags"])) if m["flags"] else ""
            f.write(f"{e['line']}, R={r:.6f}, G={g:.6f}, B={b:.6f}, "
                    f"gain_R={m['gain_r']:.6f}, gain_B={m['gain_b']:.6f}{flags}\n")
    print(f"Card measurements written to {out_path}")


# --- MAIN ---------------------------------------------------------------------

# Tokens that trigger help. argparse natively handles -h / --help;
# these extras (and a bare invocation with no arguments) are intercepted
# BEFORE parsing so nothing is ever processed in help mode.
HELP_TOKENS = {"?", "/?", "-?", "-help", "help"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="a .CR3 file or a directory of them")
    parser.add_argument("-png", action="store_true",
                        help="also write 8-bit sRGB preview PNGs")
    parser.add_argument("-s", "--size", type=int, choices=[512, 1024],
                        default=1024,
                        help="stored patch size; train-time random-crop smaller")
    parser.add_argument("--card", action="store_true",
                        help="gray-card WB mode; requires --region")
    parser.add_argument("--region", type=str, default=None,
                        help="'x,y,w,h' (pixels) or path to a region file "
                             "with lines '<CR3 name>, <x>, <y>, <w>, <h>'")
    return parser


def main():
    parser = build_parser()

    # Help mode: process nothing, print usage, exit.
    argv = sys.argv[1:]
    if not argv or any(a.lower() in HELP_TOKENS for a in argv):
        parser.print_help()
        return

    args = parser.parse_args(argv)

    if args.card and not args.region:
        parser.error("--card requires --region")
    if args.region and not args.card:
        parser.error("--region is only meaningful together with --card")

    target = Path(args.input)
    out_dir = Path("Harvested_Dataset")
    out_dir.mkdir(exist_ok=True)

    files = [target] if target.is_file() else sorted(target.rglob("*.[cC][rR]3"))
    if not files:
        print("No CR3 files found.")
        return

    # --- card mode setup -------------------------------------------------
    region_mode, region_data, region_order = (None, None, None)
    if args.card:
        region_mode, region_data, region_order = parse_region_arg(args.region)

    # --- manual slices: auto-discover data_harvest.txt in the input folder --
    manual_map = {}
    manual_dir = target if target.is_dir() else target.parent
    manual_path = manual_dir / MANUAL_FILE_NAME
    if manual_path.is_file():
        print(f"Manual slice file FOUND: {manual_path} -- its slices will be "
              f"added to the automatic harvesting")
        try:
            manual_map = parse_manual_file(str(manual_path))
        except ValueError as e:
            print(f"  [ERROR] {e} -- manual slices skipped, "
                  f"automatic harvesting continues")
            manual_map = {}
    else:
        print(f"Manual slice file not found ({manual_path}) -- "
              f"automatic harvesting only")

    current = None          # most recent successful card measurement
    out_entries = []        # rows for data_file_out.txt
    seen_names = set()
    seen_manual = set()

    for f in files:
        wb = {"source": "as_shot", "measurement": None}

        if args.card:
            key = f.name.lower()
            region = None
            line = None
            if region_mode == "inline":
                region, line = region_data, \
                    f"{f.name}, {region_data[0]}, {region_data[1]}, " \
                    f"{region_data[2]}, {region_data[3]}"
            elif key in region_data:
                region = region_data[key]["region"]
                line = region_data[key]["line"]
                seen_names.add(key)

            if region is not None:
                try:
                    meas = measure_card(f, region)
                    if meas["flags"]:
                        print(f"  [WARN] card in {f.name}: "
                              f"{'|'.join(meas['flags'])} -- measurement kept, "
                              f"verify it")
                    current = meas
                    wb = {"source": "card_measured", "measurement": meas}
                    out_entries.append({"line": line, "measurement": meas})
                except Exception as e:
                    print(f"  [ERROR] card measurement failed for {f.name}: {e}")
                    out_entries.append({"line": line, "error": str(e)})
                    # fall through: inherit previous card or as-shot
                    if current is not None:
                        wb = {"source": "card_inherited", "measurement": current}
            elif current is not None:
                wb = {"source": "card_inherited", "measurement": current}
            else:
                print(f"  [WARN] {f.name}: no card measured yet in sorted order; "
                      f"falling back to as-shot WB (silver tier)")

        key_m = f.name.lower()
        points = manual_map.get(key_m)
        if points:
            seen_manual.add(key_m)
        analyze_and_harvest(f, out_dir, args.png, args.size, wb,
                            manual_points=points)

    if manual_map:
        for name in manual_map:
            if name not in seen_manual:
                print(f"  [WARN] {MANUAL_FILE_NAME} lists '{name}' "
                      f"but no such CR3 was processed")

    if args.card:
        if region_mode == "file":
            for key in region_order:
                if key not in seen_names:
                    print(f"  [WARN] region file lists '{region_data[key]['line']}' "
                          f"but no such CR3 was processed")
        write_data_file_out(out_dir, out_entries)


if __name__ == "__main__":
    main()
