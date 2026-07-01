#!/usr/bin/env python3
"""Probe the input shapes of a Kinetics clip directory to size Qwen3-VL.

Answers the practical question: "can we caption with QWEN3_VL_MAX_NFRAMES=4,
or do we need to extend the frame budget (and recompile)?"

It reads metadata from a sample of .mp4 clips and reports the distribution of
resolution / fps / duration, then computes — for a range of candidate frame
counts — the resulting Qwen3-VL vision-token count and whether it fits within
MAX_SEQ_LEN. Motion-sampling density (seconds between sampled frames) is also
reported, since that is what limits the reliability of the motion fields
(motion_type / motion_intensity / camera_motion) in the captions.

Uses PyAV (`import av`) for metadata — the same library the Qwen3-VL server
image already installs (`uv pip install av`) — so no system ffmpeg/ffprobe is
required. Falls back to the `ffprobe` binary if PyAV is unavailable.

Qwen3-VL / Qwen2-VL vision tokenization (used for the estimate):
    - patch size 14, spatial merge 2  -> one token per 28x28 pixel block
    - temporal merge 2                -> frames are consumed in pairs
    tokens ≈ ceil(nframes / 2) * (round(H/28) * round(W/28))
This matches qwen_vl_utils' smart-resize behavior closely enough to size the
sequence budget; treat it as an estimate, not the exact processor output.

Usage:
    python probe_shapes.py --videos-dir /path/to/k400/videos/train \
        --sample 300 --max-seq-len 4096 --frame-candidates 4,8,12,16
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

PATCH_MERGE = 28   # patch(14) * spatial_merge(2)
TEMPORAL_MERGE = 2


def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)


def stable_fraction_hash(key: str) -> float:
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


try:
    import av  # PyAV — present in the Qwen3-VL server image
    _HAVE_AV = True
except ImportError:
    _HAVE_AV = False


def _probe_av(path: Path):
    """Return (width, height, fps, duration_s) via PyAV, or None on failure."""
    try:
        with av.open(str(path)) as container:
            vstreams = [s for s in container.streams if s.type == "video"]
            if not vstreams:
                return None
            st = vstreams[0]
            w = int(st.codec_context.width)
            h = int(st.codec_context.height)
            fps = float(st.average_rate) if st.average_rate else 0.0
            # Duration: prefer stream, fall back to container (both in their
            # own time_base / microseconds respectively).
            dur = 0.0
            if st.duration is not None and st.time_base:
                dur = float(st.duration * st.time_base)
            elif container.duration:
                dur = float(container.duration) / 1_000_000.0
            return w, h, fps, dur
    except Exception:
        return None


def _probe_ffprobe(path: Path):
    """Fallback: return (width, height, fps, duration_s) via the ffprobe CLI."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
        "-of", "json", str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if out.returncode != 0:
        return None
    try:
        data = json.loads(out.stdout)
        st = data["streams"][0]
        w, h = int(st["width"]), int(st["height"])
        num, den = st.get("avg_frame_rate", "0/1").split("/")
        fps = (float(num) / float(den)) if float(den) else 0.0
        dur = float(data.get("format", {}).get("duration", 0.0) or 0.0)
        return w, h, fps, dur
    except (KeyError, IndexError, ValueError, ZeroDivisionError):
        return None


def probe_video(path: Path):
    """Return (width, height, fps, duration_s) or None on failure."""
    if _HAVE_AV:
        return _probe_av(path)
    return _probe_ffprobe(path)


def tokens_per_frameset(w: int, h: int, nframes: int) -> int:
    """Estimated Qwen vision tokens for nframes at resolution WxH."""
    gw = max(1, round(w / PATCH_MERGE))
    gh = max(1, round(h / PATCH_MERGE))
    temporal = max(1, -(-nframes // TEMPORAL_MERGE))  # ceil
    return temporal * gw * gh


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0
    i = min(len(sorted_vals) - 1, int(round((p / 100.0) * (len(sorted_vals) - 1))))
    return sorted_vals[i]


def parse_args():
    p = argparse.ArgumentParser(description="Probe Kinetics clip shapes for Qwen3-VL sizing.")
    p.add_argument("--videos-dir", required=True, type=Path)
    p.add_argument("--sample", type=int, default=300,
                   help="Max clips to ffprobe (deterministic sample).")
    p.add_argument("--max-seq-len", type=int, default=4096,
                   help="Server MAX_SEQ_LEN; text+vision tokens must fit under this.")
    p.add_argument("--text-budget", type=int, default=320,
                   help="Approx tokens reserved for the text prompt + generation headroom.")
    p.add_argument("--frame-candidates", default="4,8,12,16",
                   help="Comma-separated frame counts to evaluate.")
    return p.parse_args()


def main():
    args = parse_args()
    if not _HAVE_AV and not shutil.which("ffprobe"):
        eprint("Neither PyAV (import av) nor ffprobe is available. "
               "Install one: `uv pip install av`.")
        sys.exit(1)
    eprint(f"Metadata backend: {'PyAV' if _HAVE_AV else 'ffprobe CLI'}")
    if not args.videos_dir.exists():
        eprint(f"videos-dir does not exist: {args.videos_dir}")
        sys.exit(1)

    candidates = [int(x) for x in args.frame_candidates.split(",") if x.strip()]

    clips = sorted(args.videos_dir.rglob("*.mp4"))
    eprint(f"Found {len(clips)} clips; sampling up to {args.sample} for probing")
    if not clips:
        sys.exit(1)

    # Deterministic sample spread across the tree.
    clips.sort(key=lambda p: stable_fraction_hash(p.stem))
    sample = clips[:args.sample]

    res_counter = Counter()
    fpss, durs, widths, heights = [], [], [], []
    failed = 0
    for path in sample:
        info = probe_video(path)
        if info is None:
            failed += 1
            continue
        w, h, fps, dur = info
        res_counter[(w, h)] += 1
        widths.append(w); heights.append(h)
        if fps > 0:
            fpss.append(fps)
        if dur > 0:
            durs.append(dur)

    n = len(sample) - failed
    if n == 0:
        eprint("All ffprobe calls failed; cannot report.")
        sys.exit(1)

    durs_s = sorted(durs)
    fpss_s = sorted(fpss)
    med_dur = pct(durs_s, 50) or 10.0

    print("=" * 68)
    print(f"KINETICS SHAPE PROBE  ({n} clips probed, {failed} failed)")
    print("=" * 68)

    print("\nTop resolutions (WxH : count):")
    for (w, h), c in res_counter.most_common(8):
        print(f"  {w}x{h:<5}: {c}")

    print("\nDuration (s):  "
          f"p05={pct(durs_s,5):.1f}  p50={pct(durs_s,50):.1f}  "
          f"p95={pct(durs_s,95):.1f}  max={durs_s[-1] if durs_s else 0:.1f}")
    if fpss_s:
        print("fps:           "
              f"p05={pct(fpss_s,5):.1f}  p50={pct(fpss_s,50):.1f}  "
              f"p95={pct(fpss_s,95):.1f}")

    # Use the p95 resolution as the sizing worst-case so the chosen frame count
    # fits for nearly all clips, not just the median one.
    ws, hs = sorted(widths), sorted(heights)
    w95, h95 = pct(ws, 95), pct(hs, 95)
    wmed, hmed = pct(ws, 50), pct(hs, 50)
    print(f"\nSizing resolution: median={wmed}x{hmed}  p95={w95}x{h95}")

    avail = args.max_seq_len - args.text_budget
    print(f"\nVision-token budget: MAX_SEQ_LEN={args.max_seq_len} "
          f"- text_budget={args.text_budget} = {avail} tokens for video\n")

    print(f"{'frames':>6} | {'tok@median':>10} | {'tok@p95':>8} | "
          f"{'fits p95?':>9} | {'sec/frame@p50dur':>16}")
    print("-" * 68)
    for nf in candidates:
        tok_med = tokens_per_frameset(wmed, hmed, nf)
        tok_p95 = tokens_per_frameset(w95, h95, nf)
        fits = "yes" if tok_p95 <= avail else "NO"
        sec_per = med_dur / nf if nf else 0
        print(f"{nf:>6} | {tok_med:>10} | {tok_p95:>8} | {fits:>9} | "
              f"{sec_per:>14.2f}s")

    print("\nHow to read this:")
    print("  * 'fits p95?' = does the vision-token count fit under the seq budget")
    print("    for ~95% of clips. 'NO' means raising nframes needs a bigger")
    print("    MAX_SEQ_LEN (and a NEFF recompile), or lower input resolution.")
    print("  * 'sec/frame' = spacing between sampled frames. Above ~1.5-2s the")
    print("    motion fields (motion_type/intensity/camera_motion) get")
    print("    unreliable due to temporal aliasing. Lower is better for motion.")
    print("  * Changing QWEN3_VL_MAX_NFRAMES forces a one-time recompile of the")
    print("    video prefill NEFF on the server.")


if __name__ == "__main__":
    main()
