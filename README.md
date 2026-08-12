OAK bridge for Point Clouds and Skeletons

## Processes

- `stream_server.py` — the camera server: depth + rgb (+ optionally pose)
  into shared memory. Run with `--interactive` (or
  `NICE_STREAM_INTERACTIVE=1` in the PyCharm run config) for a terminal
  profile/model picker on startup.
- `pose_server.py` — optional host-side pose backend (RTMO on the GPU,
  higher quality than the on-device YOLO). Run stream_server with
  `NICE_STREAM_POSE_SOURCE=none` (interactive profile `[3]`), then this
  alongside it. Writes the same pose segment; Unity can't tell the
  difference. One-time install into the venv:

      pip install "onnxruntime-gpu[cuda,cudnn]"
      pip install --no-deps rtmlib tqdm

  Model via `NICE_STREAM_RTMO_MODEL` = `s` | `m` (default) | `l`.
  Watch the startup log for `onnxruntime providers:` — if
  CUDAExecutionProvider is missing it fell back to CPU.
- `osc_bridge.py` — pose segment -> OSC movement signals (Sonic Pi etc.).

Pose backends write the identical NSKP contract, so they are freely
swappable per run; the on-device YOLO path stays the default. The wire
layout lives in `nsk.py` (single source of truth, mirrored by the Unity
readers).

## Development

```
.venv/Scripts/python.exe -m pytest         # tests
.venv/Scripts/python.exe -m reuse lint     # license compliance (REUSE 3.3)
```

Docs (`mkdocs-terok`) are built and published to GitHub Pages by CI
(`.github/workflows/docs.yml`) on every push to master — no local docs
environment needed. (For a local preview: any Python >= 3.12 with
`pip install mkdocs-terok mkdocs-material "mkdocstrings[python]"
mkdocs-literate-nav`, then `properdocs serve`.)

## Licensing

All code in this repository is © 2026 Jiří Vyskočil, licensed under the
**MIT License**, and [REUSE](https://reuse.software/) compliant: every file
carries SPDX tags (inline or via `REUSE.toml`), the license text lives in
`LICENSES/MIT.txt` (the REUSE-mandated location), and an identical
top-level `LICENSE` exists so GitHub's interface detects it. REUSE forbids
extra files in `LICENSES/`, so the third-party situation is documented
here instead:

The neural-network **model weights are runtime-downloaded components** —
fetched from public model zoos at startup, not part of this repository,
and never redistributed by it:

| Component | License | Fetched by |
| --- | --- | --- |
| Luxonis zoo YOLOv8-pose models (on-device path) | AGPL-3.0-only | `stream_server.py` via HubAI |
| RTMO body7 checkpoints (host path) | Apache-2.0 | `pose_server.py` via rtmlib |

Those licenses apply to the weights, **not** to this codebase, and they do
not force relicensing of derivatives of our code: the code neither
contains nor links the weights — it loads them at runtime as data, the way
a media player loads a film. Derivatives of *our code* stay under MIT
terms. What the AGPL *would* require, should you go there: conveying the
weights themselves (shipping an installer/image with the YOLO models baked
in) or offering network interaction with a modified AGPL work — a local
art installation does neither. If you must bundle, get an Ultralytics
enterprise license or bundle only the Apache-2.0 RTMO path.

Runtime code dependencies have permissive top-level licenses: depthai
(MIT), depthai-nodes (Apache-2.0), rtmlib (Apache-2.0), numpy
(BSD-3-Clause), OpenCV (Apache-2.0), python-osc (public domain). Two of
the installed wheels bundle more, which again only matters if you ship the
venv in an installer/image: opencv-python redistributes FFmpeg (LGPL-2.1,
weak copyleft), and the depthai wheel carries Luxonis device firmware
licensed to run only on Luxonis hardware.

*(This is the project's understanding, not legal advice.)*

## Hardware notes

For testing OAK4, download OAK Viewer.
If it doesn't show up even if connected and LED is up
static blue, try reset (use SIM tool on the back button, hold for 5s).
OAK Viewer should pick it up afterwards.
