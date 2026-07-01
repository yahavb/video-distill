# video-distill

Tooling to turn the **Kinetics** human-action video dataset into a
**motion-focused prompt corpus** for adapting video diffusion models (WAN2.2)
to streaming inference ([StreamDiffusion v2](https://streamdiffusionv2.github.io)).

This is **Stage 1** of a larger pipeline:

```
Kinetics clips ──► Qwen3-VL captions ──► (LLM prompt variations) ──►
    WAN2.2 teacher videos ──► StreamDiffusion-v2 student training
   └──────────── this repo ────────────┘
```

The goal of the distillation is to transfer *streaming behavior* — temporal
consistency, recurrent conditioning, rolling-context handling, stable streaming
dynamics — not image quality. Motion is therefore the signal we most want the
captions to capture, which is why the captioner emphasizes motion/camera
dynamics and emits structured motion metadata.

## Contents

| File | Purpose |
|------|---------|
| `probe_shapes.py` | `ffprobe` a clip dir → resolution/fps/duration distribution + a `frames → vision-tokens → fits? → sec/frame` table to size `QWEN3_VL_MAX_NFRAMES` / `MAX_SEQ_LEN`. No model needed. |
| `caption_kinetics.py` | Single-threaded, resumable captioner. Deterministic hash-sampling, motion-focused prompt + structured metadata, per-clip latency + full-run projection. |
| `caption-job.yaml` | CPU-only k8s Job: downloads one Kinetics part, probes shapes, captions ~10% against the in-cluster Qwen3-VL server, persists results to a PVC. |

## Why the frame budget matters

Kinetics clips are ~10s. The Qwen3-VL server samples a fixed number of frames
(`QWEN3_VL_MAX_NFRAMES`, default 4). At 4 frames that's ~1 frame / 2.5s — enough
for *appearance* but marginal for the **motion fields** (`motion_type`,
`motion_intensity`, `camera_motion`) that this pipeline cares about. Whether you
can afford more frames depends on clip resolution (vision tokens ≈
`ceil(frames/2) · (H/28) · (W/28)`), which is exactly what `probe_shapes.py`
measures. Low-res Kinetics (~256 short side) fits 16 frames easily; HD does not.
Changing the frame count requires a one-time NEFF recompile of the server.

## Quick start (in-cluster)

```bash
# 1. Point the manifest at a CPU (m-family) node:
kubectl get no --show-labels          # pick a label, edit nodeSelector
# 2. Set GIT_REPO/GIT_BRANCH env in the manifest to this repo.
kubectl apply -f caption-job.yaml
kubectl logs -f job/kinetics-caption-probe
# Results land at /var/mdl/video-distill/captions-<stamp>.jsonl on the PVC.
```

The pod is a **pure HTTP client**: it base64-encodes clips and POSTs them to the
trn2 Qwen3-VL server over the cluster service; the server does frame sampling
and inference. No Neuron resources are claimed by this job.

## Output schema (JSONL, one clip per line)

```json
{"clip_id": "abc123XYZ00", "video": "...", "label": "playing violin",
 "prompt": "A violinist ... the camera slowly circles ...",
 "camera_motion": "orbit", "subject_motion": "bowing arm moves rhythmically",
 "motion_intensity": 3, "motion_type": "periodic"}
```

## Attribution

Kinetics is by DeepMind; hosting and the original downloader scripts are by the
[CVDF](https://github.com/cvdfoundation/kinetics-dataset). Qwen3-VL is by the
Qwen team. See [`NOTICE`](./NOTICE). Code here is MIT ([`LICENSE`](./LICENSE)).
