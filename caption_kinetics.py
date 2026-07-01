#!/usr/bin/env python3
"""Caption Kinetics clips with a Qwen3-VL OpenAI-compatible endpoint.

Stage 1 of the Kinetics -> prompt-corpus pipeline. Walks a directory of
Kinetics .mp4 clips, samples a deterministic fraction of them, and asks a
Qwen3-VL server to produce a motion-focused text-to-video prompt plus
structured motion metadata for each clip. Results are appended to a JSONL
file and the run is resumable (already-captioned clip ids are skipped).

The target server is the one defined in ~/qwen3-vl/qwen3-vl-deploy.yaml:
  - OpenAI-compatible  POST /v1/chat/completions  on :8000
  - accepts a `video_url` content part; the SERVER samples frames itself
    (QWEN3_VL_MAX_NFRAMES=4 by default) via qwen_vl_utils.process_vision_info
  - one request at a time (global inference lock) -> single-threaded client
    is the right match; scale by running N replicas + a load balancer, not
    by threading this script.

Usage (probe 10% of an arranged K400 tree, deterministic sample):
    kubectl port-forward svc/qwen3-vl 8000:8000 &
    python caption_kinetics.py \
        --videos-dir /path/to/k400/videos/train \
        --out captions.jsonl \
        --sample-frac 0.10

Layouts supported for deriving the action label:
    arranged  videos/<split>/<label>/<id>.mp4   (arrange_by_classes.py output)
              -> label = parent directory name  (default: --label-from parent)
    flat      a directory of <id>.mp4 with labels in an annotations CSV
              -> pass --labels-csv annotations/train.csv (--label-from csv)
    none      no labels available -> --label-from none (more hallucination)
"""

import argparse
import base64
import hashlib
import io
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


# ─── Prompt sent to Qwen3-VL ────────────────────────────────────────────────
# Motion is the signal the downstream student model must learn to keep stable,
# so the instruction pushes the captioner toward motion/camera dynamics rather
# than static appearance. The known Kinetics action label is injected as a hint
# to anchor the description and reduce hallucination, but the model is told to
# trust the video over the label if they disagree (Kinetics labels are noisy).
SYSTEM_PROMPT = (
    "You are a precise video annotator that writes prompts for training a "
    "text-to-video diffusion model. You describe what is actually visible in "
    "the clip, focusing on motion and camera dynamics. You never invent "
    "fantasy, sci-fi, CGI, or impossible physics. Output only valid JSON."
)

USER_INSTRUCTION = """\
This short clip is labeled with the human action: "{label}".
Watch the motion across the frames and describe THIS clip.

Return a single JSON object with exactly these fields:
- "prompt": one realistic text-to-video prompt (<= 80 words) that could be
  recorded by a real camera. Describe the primary subject and appearance, the
  background/scene, the subject's motion, the camera motion, and the lighting.
  Prefer describing motion over static detail. No fantasy, no CGI.
- "camera_motion": one of "static", "slow_pan", "fast_pan", "zoom", "orbit",
  "handheld", "tracking".
- "subject_motion": a short phrase describing how the subject moves.
- "motion_intensity": integer 1-5 (1 = nearly still, 5 = fast/chaotic motion).
- "motion_type": one of "periodic", "chaotic", "smooth", "mixed".

If the video clearly contradicts the label, describe the video, not the label.
Output only the JSON object, nothing else."""

# When --label-from none, we cannot anchor on a label.
USER_INSTRUCTION_NO_LABEL = USER_INSTRUCTION.replace(
    'This short clip is labeled with the human action: "{label}".\n', ""
).replace("If the video clearly contradicts the label, describe the video, not the label.\n", "")


def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)


def clip_id(path: Path) -> str:
    """Kinetics clip id is the first 11 chars of the youtube id stem, matching
    arrange_by_classes.py (`str(p.stem)[:11]`). Falls back to full stem."""
    stem = path.stem
    return stem[:11] if len(stem) >= 11 else stem


def stable_fraction_hash(key: str) -> float:
    """Deterministic value in [0,1) from a key, stable across runs/machines.

    Used so --sample-frac 0.10 picks the SAME ~10% every run and spreads the
    sample across all labels, instead of taking the first 10% of the file list
    (which would over-sample a few alphabetically-early classes)."""
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def load_labels_csv(csv_path: Path) -> dict:
    """Kinetics annotation CSV -> {clip_id: label}.

    Mirrors arrange_by_classes.load_label: label is column 0, youtube id is
    column 1 (first 11 chars). Uses only the stdlib csv module."""
    import csv

    mapping = {}
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 2:
                continue
            label = row[0].replace('"', "").strip()
            yid = row[1].strip()[:11]
            mapping[yid] = label
    return mapping


def derive_label(path: Path, mode: str, labels_csv: dict) -> str:
    if mode == "parent":
        # arranged layout: .../<label>/<id>.mp4  ->  label with _ normalized
        return path.parent.name.replace("_", " ")
    if mode == "csv":
        return labels_csv.get(clip_id(path), "")
    return ""  # none


def video_to_data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "video/mp4"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def reencode_fixed(path: Path, size: int) -> bytes:
    """Re-encode a clip to a FIXED size x size resolution and return mp4 bytes.

    Why fixed (not aspect-preserving): the Neuron vision graph is a static-shape
    compiled NEFF. It emits a constant number of visual tokens for the ONE grid
    it was compiled for. If clips arrive at varying resolutions, the per-clip
    <video> placeholder count (computed on CPU) drifts away from that constant,
    causing the [1152] vs [683] mismatch crash. Forcing every clip to the same
    square resolution makes the grid constant -> the server compiles ONE video
    NEFF (first request) and reuses it for all others. Aspect ratio is not
    preserved (mild stretch); acceptable for motion-focused captioning.

    Uses PyAV so no system ffmpeg binary is needed (the server image ships av).
    """
    import av  # local import: only needed when --resize is set

    inp = av.open(str(path))
    try:
        istream = next(s for s in inp.streams if s.type == "video")
        buf = io.BytesIO()
        out = av.open(buf, mode="w", format="mp4")
        try:
            # Cap fps modestly; the server subsamples frames anyway.
            rate = istream.average_rate or 25
            ostream = out.add_stream("libx264", rate=rate)
            ostream.width = size
            ostream.height = size
            ostream.pix_fmt = "yuv420p"
            for frame in inp.decode(istream):
                frame = frame.reformat(width=size, height=size, format="yuv420p")
                for pkt in ostream.encode(frame):
                    out.mux(pkt)
            for pkt in ostream.encode():  # flush
                out.mux(pkt)
        finally:
            out.close()
        return buf.getvalue()
    finally:
        inp.close()


def video_bytes_to_data_uri(data: bytes) -> str:
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:video/mp4;base64,{b64}"


def build_payload(path: Path, label: str, model: str, max_tokens: int,
                  use_path: bool, resize: int = 0) -> dict:
    """OpenAI-compatible request body. The server turns the video_url into a
    {"type": "video", ...} content part and samples MAX_NFRAMES frames.

    resize>0 re-encodes the clip to a fixed resize x resize resolution so every
    request hits the SAME static vision NEFF on the server (avoids the varying
    grid -> visual-token mismatch crash). resize overrides use_path (the server
    can't read our in-memory re-encoded bytes from disk)."""
    if resize > 0:
        video_ref = video_bytes_to_data_uri(reencode_fixed(path, resize))
    elif use_path:
        video_ref = str(path)  # server reads the path directly (same host/mount)
    else:
        video_ref = video_to_data_uri(path)  # portable: base64 over the wire

    instruction = (USER_INSTRUCTION.format(label=label) if label
                   else USER_INSTRUCTION_NO_LABEL)
    return {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": video_ref}},
                {"type": "text", "text": instruction},
            ]},
        ],
    }


def post_chat(endpoint: str, payload: dict, timeout: float) -> str:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=data, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # Surface the server's error body — for this server a 500 usually means
        # the clip's resolution produced a vision-token count that overflowed
        # MAX_SEQ_LEN or failed to compile a prefill NEFF. Include it so the
        # caller can log *why*, not just "500".
        try:
            detail = e.read().decode("utf-8", "replace")[:500]
        except Exception:
            detail = ""
        raise urllib.error.HTTPError(
            e.url, e.code, f"{e.reason}: {detail}", e.headers, None)
    return body["choices"][0]["message"]["content"]


def parse_model_json(text: str) -> dict:
    """Extract the JSON object from the model output. Tolerates ```json fences
    and leading/trailing prose by slicing the outermost braces."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if "```" in t[3:] else t
        t = t[4:] if t.lower().startswith("json") else t
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1 and end > start:
        t = t[start:end + 1]
    return json.loads(t)


def load_done_ids(out_path: Path) -> set:
    """Resume support: collect clip ids already present in the output JSONL."""
    done = set()
    if not out_path.exists():
        return done
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("clip_id"):
                done.add(rec["clip_id"])
    return done


def parse_args():
    p = argparse.ArgumentParser(description="Caption Kinetics clips with Qwen3-VL.")
    p.add_argument("--videos-dir", required=True, type=Path,
                   help="Directory of .mp4 clips (searched recursively).")
    p.add_argument("--out", required=True, type=Path,
                   help="Output JSONL path (appended, resumable).")
    p.add_argument("--endpoint", default="http://qwen3-vl.default.svc.cluster.local:8000",
                   help="Qwen3-VL OpenAI-compatible base URL. Defaults to the "
                        "in-cluster service DNS (svc qwen3-vl, namespace default, "
                        "port 8000). Use http://localhost:8000 if port-forwarding.")
    p.add_argument("--model", default="Qwen3-VL-8B-Instruct")
    p.add_argument("--sample-frac", type=float, default=0.10,
                   help="Deterministic fraction of clips to caption (0-1).")
    p.add_argument("--max-clips", type=int, default=0,
                   help="Hard cap on clips this run (0 = no cap).")
    p.add_argument("--label-from", choices=["parent", "csv", "none"],
                   default="parent", help="How to derive the action label.")
    p.add_argument("--labels-csv", type=Path, default=None,
                   help="Annotations CSV when --label-from csv.")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--timeout", type=float, default=300.0,
                   help="Per-request HTTP timeout (s).")
    p.add_argument("--send-path", action="store_true",
                   help="Send the file path instead of base64 (only when the "
                        "server shares this filesystem; avoids /tmp leak).")
    p.add_argument("--resize", type=int, default=448,
                   help="Re-encode each clip to a fixed NxN resolution before "
                        "sending, so every request hits the same static vision "
                        "NEFF on the server (prevents the visual-token mismatch "
                        "crash). Default 448. Set 0 to disable (send original).")
    p.add_argument("--retries", type=int, default=2,
                   help="Retries per clip on transient errors.")
    p.add_argument("--request-delay", type=float, default=1.0,
                   help="Seconds to sleep before each clip's request (throttle "
                        "the single-threaded server). Default 1.0.")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.videos_dir.exists():
        eprint(f"videos-dir does not exist: {args.videos_dir}")
        sys.exit(1)

    labels_csv = {}
    if args.label_from == "csv":
        if not args.labels_csv or not args.labels_csv.exists():
            eprint("--label-from csv requires an existing --labels-csv")
            sys.exit(1)
        labels_csv = load_labels_csv(args.labels_csv)
        eprint(f"Loaded {len(labels_csv)} labels from {args.labels_csv}")

    all_clips = sorted(args.videos_dir.rglob("*.mp4"))
    eprint(f"Found {len(all_clips)} .mp4 clips under {args.videos_dir}")

    # Deterministic hash-based sampling spread across all labels.
    frac = max(0.0, min(1.0, args.sample_frac))
    sampled = [p for p in all_clips if stable_fraction_hash(clip_id(p)) < frac]
    eprint(f"Sampled {len(sampled)} clips (frac={frac:.3f}, deterministic)")

    done = load_done_ids(args.out)
    if done:
        eprint(f"Resume: {len(done)} clips already captioned, will skip them")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    processed = ok = failed = 0
    latencies = []
    t_start = time.time()

    with args.out.open("a") as out_f:
        for path in sampled:
            cid = clip_id(path)
            if cid in done:
                continue
            if args.max_clips and processed >= args.max_clips:
                eprint(f"Reached --max-clips={args.max_clips}, stopping")
                break
            processed += 1

            label = derive_label(path, args.label_from, labels_csv)
            payload = build_payload(path, label, args.model, args.max_tokens,
                                    args.send_path, args.resize)

            # Throttle: the server processes one request at a time (global
            # inference lock); a small pre-call delay eases pressure on it.
            if args.request_delay > 0:
                time.sleep(args.request_delay)

            raw = None
            err = None
            for attempt in range(args.retries + 1):
                t0 = time.time()
                try:
                    raw = post_chat(args.endpoint, payload, args.timeout)
                    latencies.append(time.time() - t0)
                    err = None
                    break
                except (urllib.error.URLError, urllib.error.HTTPError,
                        TimeoutError, ConnectionError) as e:
                    err = str(e)
                    if attempt < args.retries:
                        time.sleep(2 * (attempt + 1))

            rec = {"clip_id": cid, "video": str(path), "label": label}
            if raw is None:
                failed += 1
                rec["error"] = err or "unknown"
                eprint(f"[{processed}] FAIL {cid}: {err}")
            else:
                try:
                    parsed = parse_model_json(raw)
                    rec.update({
                        "prompt": parsed.get("prompt", "").strip(),
                        "camera_motion": parsed.get("camera_motion"),
                        "subject_motion": parsed.get("subject_motion"),
                        "motion_intensity": parsed.get("motion_intensity"),
                        "motion_type": parsed.get("motion_type"),
                    })
                    ok += 1
                except (json.JSONDecodeError, ValueError) as e:
                    # Keep the raw text so a bad-JSON clip is inspectable, not lost.
                    failed += 1
                    rec["error"] = f"json_parse: {e}"
                    rec["raw"] = raw
                    eprint(f"[{processed}] BADJSON {cid}: {e}")

            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()

            if processed % 10 == 0:
                avg = sum(latencies) / len(latencies) if latencies else 0
                eprint(f"  progress: {processed} done "
                       f"(ok={ok} fail={failed}) avg={avg:.1f}s/clip")

    elapsed = time.time() - t_start
    avg = sum(latencies) / len(latencies) if latencies else 0
    eprint("─" * 60)
    eprint(f"Done. processed={processed} ok={ok} failed={failed}")
    eprint(f"Wall time: {elapsed:.0f}s, avg successful req: {avg:.1f}s/clip")
    if avg and len(all_clips):
        # Rough projection to the full (100%) dataset at this per-clip cost.
        full_est = avg * len(all_clips)
        eprint(f"Projected single-replica time for all {len(all_clips)} clips: "
               f"{full_est/3600:.1f}h (before parallel replicas)")
    eprint(f"Output: {args.out}")


if __name__ == "__main__":
    main()
