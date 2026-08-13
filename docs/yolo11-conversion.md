# YOLO11-pose on the OAK-4 (RVC4)

The Luxonis zoo has no YOLO11 pose models — you convert your own through
HubAI (the free tier covers conversion) and use either the resulting
private model slug or the downloaded NNArchive file. `stream_server`
accepts both in `NICE_STREAM_POSE_MODEL` and in the interactive picker's
`[c]` custom entry. All five scales (n/s/m/l/x) are supported by the
Luxonis exporter; YOLO11-pose decodes through the exact same
`YOLOExtendedParser` path as YOLOv8 (explicit `yolov11` subtype exists
since depthai-nodes 0.5.1).

## One-time setup

```
pip install hubai-sdk        # any Python >= 3.10; the camera venv works
hubai login                  # paste a key from hub.luxonis.com/team-settings/api-keys
```

Get the checkpoint from Ultralytics (AGPL-3.0 — see the README's
Licensing section; the weights are used at conversion/run time, never
committed here): e.g. `yolo11l-pose.pt` from the ultralytics/assets
GitHub releases.

## Convert — FP16 (simplest: no calibration data at all)

```
hubai convert RVC4 --path yolo11l-pose.pt --name nice-stream-pose --quantization-mode FP16_STANDARD --yolo-input-shape 640 384
```

## Convert — INT8 (fastest; calibrate on real venue frames)

With `stream_server` running and people moving in front of the camera:

```
.venv/Scripts/python.exe calib_capture.py --count 300 --interval 0.5
```

```
hubai convert RVC4 --path yolo11l-pose.pt --name nice-stream-pose --quantization-mode INT8_STANDARD --quantization-data ./calib_frames.zip --max-quantization-images 300 --yolo-input-shape 640 384
```

If keypoints jitter under INT8, escalate the mode:
`INT8_INT16_MIXED` → `INT8_INT16_MIXED_ACCURACY_FOCUSED` →
`FP16_STANDARD`.

## Use it

The convert call both downloads the NNArchive and registers a private
model on your Hub team. Either works:

```
NICE_STREAM_POSE_MODEL=path/to/nice-stream-pose.tar.xz       # local file
NICE_STREAM_POSE_MODEL=<team>/nice-stream-pose:<variant>     # Hub slug
```

The slug path needs `DEPTHAI_HUB_API_KEY` set for the private download
(note: some depthai docs say `DEPTHAI_ZOO_API_KEY`; the code reads
`DEPTHAI_HUB_API_KEY`). The pipeline reads the input size from the
archive and letterboxes into it automatically — nothing else to change.
After the first run, check the log line `pose model ... (input WxH)` and
that skeletons overlay people correctly.

## Why 640×384

The camera frame is 1280×800 (1.6:1). 640×384 (1.667:1) letterboxes with
only ~13 px of padding per side, versus ~76 px wasted at the zoo's
640×352 (1.818:1), and stays a stride-32 multiple. More content pixels
per person → better keypoints for the same compute.

## Expected speed

yolo11l-pose is within ~5 % of yolov8l-pose in GFLOPs, so expect roughly
the zoo v8-large's measured 168 inf/s for INT8 at this size. FP16 is
unmeasured on RVC4 — the 48/12 TOPS ratio suggests ~40–60 inf/s, still
above the ~37 fps depth rate. Add the winner to `KNOWN_POSE_MODELS` in
`stream_server.py` so it shows up in the interactive picker.
