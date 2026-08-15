# nice-stream-oak

OAK bridge for point clouds and skeletons: reads an OAK-4 camera and
publishes depth, RGB and pose frames into shared memory for the Unity side.

## Processes

- `stream_server.py` — the camera server: depth + rgb + pose into shared
  memory. Run with `--interactive` (or `NICE_STREAM_INTERACTIVE=1` in the
  PyCharm run config) for a terminal profile/model picker on startup.
- `pose_server.py` — optional host-side pose backend (RTMO on the GPU,
  higher quality than the on-device YOLO). **You do not start this
  yourself**: run stream_server with `NICE_STREAM_POSE_SOURCE=host`
  (interactive profile `[3]`) and it starts and stops this for you, in the
  same terminal. Writes the same pose segment; Unity can't tell the
  difference. Use `=none` instead when you want to run it by hand, on
  another machine, or under a debugger. One-time install into the venv:

      pip install "onnxruntime-gpu[cuda,cudnn]"
      pip install --no-deps rtmlib tqdm

  Model via `NICE_STREAM_RTMO_MODEL` = `s` | `m` (default) | `l` | a path
  or URL. Watch the startup log for `onnxruntime providers:` — if
  CUDAExecutionProvider is missing it fell back to CPU.
  `NICE_STREAM_POSE_ROI="x,y,w,h"` (source pixels, empty = whole frame)
  crops the frame before it reaches the network, so visitors arrive at more
  pixels: `320,80,640,640` fills the 640x640 input from a 1280x800 frame
  with no padding and no downscaling.
- `osc_bridge.py` — pose segment -> OSC movement signals under `/nice/...`
  (Sonic Pi etc.; see `sonicpi_example.rb`).
- `osc_monitor.py` — terminal OSC monitor; debugs the bridge without
  Sonic Pi.
- `osc_tape.py` — records the OSC stream to a file and replays it later
  (`osc_bridge.py --record` writes the same format). Standard library
  only: hand it and a recording to whoever writes the music and they need
  no camera, no servers and no venv (see `docs/osc.md`).
- `oscii_bot_example.txt` — OSC-to-MIDI recipe for OSCII-bot; turns the
  bridge's stream into CCs and notes for MIDI-only gear (see `docs/osc.md`).
- `calib_capture.py` — grabs rgb frames from the running stream into a zip
  that HubAI accepts as INT8 quantization-calibration data.

**`stream_server.py` is the single entry point.** It owns the camera, and
`NICE_STREAM_POSE_SOURCE` picks who produces the skeletons:

| value | pose runs on | second process |
| --- | --- | --- |
| `left` (default) | the camera, IR mono — works in the dark | none |
| `rgb` | the camera, colour — needs visible light | none |
| `host` | your GPU, via RTMO | started for you |
| `none` | nobody here | yours to start |

Depth and rgb come out at full resolution under all four: the pose source
decides who writes `nice_stream_pose`, never what the point cloud gets.
Pose backends write the identical NSKP contract, so they are freely
swappable per run; the on-device YOLO path stays the default. The wire
layout lives in `nsk.py` (single source of truth, mirrored by the Unity
readers).

## Development

```
.venv/Scripts/python.exe -m pytest         # tests
.venv/Scripts/python.exe -m ruff check .   # lint
.venv/Scripts/python.exe -m mypy           # type check
.venv/Scripts/python.exe -m reuse lint     # license compliance (REUSE 3.3)
```

All four also run as a pre-commit hook (once per clone:
`git config core.hooksPath .githooks`) and in CI. `git commit --no-verify`
bypasses the hook in an emergency; CI still catches it.

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

**The USB-C link is orientation-sensitive.** The same cable in the same
ports can negotiate USB3 one day and USB2 the next, purely on which way
round the connector went in. Nothing looks broken — the camera opens,
streams, and runs at roughly a quarter of the frame rate (~8 fps instead
of ~30). The fix is to unplug the connector and replug it flipped 180°;
either end will do.

Because that costs an afternoon if you miss it, `stream_server.py` shouts
three warning lines on a degraded start, repeats the reminder on every fps
line, and publishes the negotiated speed in the status word (`nsk.LINK_*`)
so Unity can put a red banner over the Game view.

To test the OAK-4 outside this project, use OAK Viewer. If the camera does
not show up even though it is connected and its LED is static blue, reset
it: hold the button on the back with a SIM tool for 5 s. OAK Viewer picks
it up afterwards.
