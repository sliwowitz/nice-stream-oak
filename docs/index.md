# nice-stream-oak

OAK-4 camera bridge for the Nice-Stream installation: depth, rgb and
multi-person pose into shared memory, consumed by Unity (VFX Graph point
cloud + skeletons) and an OSC bridge.

## Processes

| Script | Role |
| --- | --- |
| `stream_server.py` | Camera server: depth + rgb (+ optionally on-device pose) → SHM. `--interactive` for a startup profile/model picker. |
| `pose_server.py` | Optional host-side pose backend (RTMO on the GPU). Pair with `NICE_STREAM_POSE_SOURCE=none`. |
| `osc_bridge.py` | Pose segment → [`/nice/*` OSC movement signals](osc.md) (Sonic Pi etc.). |

The shared-memory wire contract (NSK1 frame segments, NSKP pose segment)
lives in `nsk.py` — the single source of truth, mirrored by the Unity
readers in `Assets/NiceStream/`.

## Licensing

The code is MIT (REUSE-compliant, see `LICENSES/`). Model weights are
fetched at runtime and are **not** part of this repository: the Luxonis zoo
YOLOv8-pose models are AGPL-3.0, the RTMO body7 checkpoints are Apache-2.0.
Nothing in this repo redistributes them, and they don't relicense
derivatives of this code — see the README's Licensing section for the full
picture.

## Development

```
.venv/Scripts/python.exe -m pytest         # tests
.venv/Scripts/python.exe -m reuse lint     # license compliance
```

These docs are built and published to GitHub Pages by CI on every push to
master (`.github/workflows/docs.yml`).
