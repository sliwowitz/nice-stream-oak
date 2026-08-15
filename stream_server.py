# SPDX-FileCopyrightText: 2026 Jiří Vyskočil <jiri@vyskocil.com>
# SPDX-License-Identifier: MIT
"""Streams depth + rgb + multipose from one OAK camera into shared memory.

Supersedes depth_server.py. One depthai pipeline feeds three segments:

  nice_stream_depth   u16 depth, millimetres, aligned to the RGB camera
  nice_stream_rgb     RGB888 interleaved, same resolution & intrinsics as depth
  nice_stream_pose    3D skeletons: 2D keypoints from a configurable HubAI
                      pose model (default YOLOv8-large) lifted to metres by
                      sampling the aligned depth map. Optionally disabled
                      (POSE_SOURCE=none) in favour of the host backend,
                      pose_server.py.

The wire layout of all three segments lives in nsk.py -- the single source
of truth shared with pose_server.py and osc_bridge.py, and mirrored by the
Unity readers.

Because depth is aligned to CAM_A, all three streams share one set of
intrinsics and one pixel grid: a depth pixel, its colour and a keypoint at the
same (x, y) refer to the same ray. Unity does the unprojection.

Designed to run unattended for weeks; the camera is assumed to be unreliable.
Segments are created once and never torn down while this process lives.
Consumers may attach before the camera exists and survive it vanishing.
"""

import logging
import os
import signal
import struct
import sys
import time
from collections.abc import Callable
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from types import FrameType
from typing import Any, cast

import depthai as dai
import numpy as np
import numpy.typing as npt
from depthai_nodes.node import ParsingNeuralNetwork

import nsk
from nsk import (
    HEADER_SIZE,
    LINK_ETHERNET,
    LINK_NAMES,
    LINK_UNKNOWN,
    MAX_PERSONS,
    NUM_KP,
    STATUS_RECONNECTING,
    STATUS_STARTING,
    STATUS_STREAMING,
)

# Shapes shared by the pose helpers below. Keypoint objects come from the
# untyped depthai_nodes parser, hence Any.
DepthMap = npt.NDArray[np.uint16]
KpMapper = Callable[[Any], tuple[float, float]]
Remap    = Callable[[float, float, float], tuple[float, float, float]]

# ---------------------------------------------------------------- config ----
SHM_DEPTH = os.environ.get("NICE_STREAM_SHM_DEPTH", "nice_stream_depth")
SHM_RGB   = "nice_stream_rgb"
SHM_POSE  = "nice_stream_pose"

W, H   = 1280, 800
# 60 is the sensor target; stereo tops out around ~38 fps at 800P (DEFAULT
# preset) while rgb and pose reach the full 60. Consumers just take the newest.
FPS    = float(os.environ.get("NICE_STREAM_FPS", "60"))

# IR emitters (off by default in depthai; devices without them ignore these).
# Dot projector gives stereo texture on blank surfaces; flood light lets the
# mono cameras see in a dark room. NOTE: with pose on the left mono (the
# default), the NN sees both emitters -- flood is what keeps pose alive in
# the dark, but the dot speckle lands on the very frames the NN reads. If
# pose quality matters more than depth on blank walls, try IR_DOT=0.
IR_DOT   = float(os.environ.get("NICE_STREAM_IR_DOT", "0.8"))
IR_FLOOD = float(os.environ.get("NICE_STREAM_IR_FLOOD", "0.5"))

# Where the pose NN looks. "left" = the IR-flood-lit mono camera, usable in a
# dark room; keypoints are remapped into the CAM_A frame server-side so
# consumers never notice. "rgb" = the colour camera (needs visible light).
# "none" = no on-device pose; a host backend (pose_server.py) writes the
# pose segment instead, and the RVC4 spends everything on depth.
POSE_SOURCE = os.environ.get("NICE_STREAM_POSE_SOURCE", "left")

# Stereo quality preset. On RVC4 only ACCURACY and DENSITY differ: every
# other name (HIGH_DETAIL, FAST_ACCURACY, FACE, ROBOTICS) resolves to the
# same ACCURACY branch inside depthai, so asking for HIGH_DETAIL here is
# ACCURACY with extra steps. DENSITY fills more of the frame and trusts
# weaker matches -- more points, more wobble.
STEREO_PRESET = os.environ.get("NICE_STREAM_STEREO_PRESET", "ACCURACY")

# Any HubAI slug depthai-nodes can parse (the YOLO-pose family is the safe
# bet), OR a path to a local NNArchive (.tar.xz) -- e.g. a YOLO11-pose
# converted through HubAI. Input size is read from the model archive;
# frames are letterboxed into it and keypoints mapped back to full-frame
# pixels.
POSE_MODEL = os.environ.get(
    "NICE_STREAM_POSE_MODEL",
    "luxonis/yolov8-large-pose-estimation:coco-640x352")

# Known-good options for the interactive picker (slug, blurb). The zoo has
# no stronger RVC4 pose model than yolov8-large; anything newer arrives via
# custom slug (or NICE_STREAM_POSE_MODEL) after a HubAI conversion.
KNOWN_POSE_MODELS = [
    ("luxonis/yolov8-large-pose-estimation:coco-640x352",
     "best quality, ~169 inf/s RVC4"),
    ("luxonis/yolov8-nano-pose-estimation:coco-512x288",
     "fastest, ~596 inf/s RVC4"),
]

# Dev convenience: --interactive (or NICE_STREAM_INTERACTIVE=1, e.g. in a
# PyCharm run configuration) asks for the camera profile in the terminal on
# startup. Production runs leave this off and configure via env vars.
INTERACTIVE = ("--interactive" in sys.argv
               or os.environ.get("NICE_STREAM_INTERACTIVE", "0") == "1")

NBUF     = 3                        # frame ring depth (depth & rgb segments)
DEPTH_SZ = W * H * 2
RGB_SZ   = W * H * 3

PREFER_USB      = True
RETRY_DELAY     = 3.0
RETRY_DELAY_MAX = 30.0
STALE_AFTER     = 5.0               # no depth frame for this long -> dead

# depthai's UsbSpeed -> our wire codes, by name: the enum's numbering belongs
# to Luxonis, the wire's belongs to nsk.py.
USB_SPEED_LINKS = {
    "LOW":        nsk.LINK_LOW,
    "FULL":       nsk.LINK_FULL,
    "HIGH":       nsk.LINK_HIGH,
    "SUPER":      nsk.LINK_SUPER,
    "SUPER_PLUS": nsk.LINK_SUPER_PLUS,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("stream_server.log", encoding="utf-8")],
)
log = logging.getLogger("nice-stream")


# ------------------------------------------------------------------ main ----
def main() -> None:
    """Creates the segments once, then runs camera sessions until stopped.

    Frame ids persist across sessions, so consumers see one monotonic
    stream. The retry delay backs off toward RETRY_DELAY_MAX.
    """
    global _link, _reconnects

    if INTERACTIVE:
        configure_interactively()
    log.info("config: pose source=%s  model=%s  ir dot=%.2f flood=%.2f",
             POSE_SOURCE, POSE_MODEL, IR_DOT, IR_FLOOD)

    create_segments()
    write_headers()
    set_status(STATUS_STARTING)
    nsk.commit_frame(buf_depth, 0)
    nsk.commit_frame(buf_rgb, 0)
    if POSE_SOURCE != "none":       # never stomp pose_server's commits
        nsk.commit_frame(buf_pose, 0)

    ids = {"depth": 0, "rgb": 0, "pose": 0}
    delay = RETRY_DELAY

    while _running:
        try:
            ids = stream_once(ids)
            if not _running:
                break
            log.warning("session ended cleanly -- reopening")
        # Deliberately broad: any device failure becomes a reconnect. The
        # process must survive weeks unattended.
        except Exception as exc:
            log.error("session failed: %s", exc)

        if not _running:
            break

        _reconnects += 1
        _link = LINK_UNKNOWN        # no device, no link -- clear Unity's banner
        set_status(STATUS_RECONNECTING)
        log.info("reconnect #%d in %.0fs", _reconnects, delay)
        deadline = time.time() + delay
        while _running and time.time() < deadline:
            time.sleep(0.2)
        delay = min(delay * 1.5, RETRY_DELAY_MAX)

    set_status(STATUS_STARTING)
    log.info("stopped after %d reconnects (depth %d, rgb %d, pose %d frames)",
             _reconnects, ids["depth"], ids["rgb"], ids["pose"])


# ------------------------------------------------------------ one session ---
def stream_once(ids: dict[str, int]) -> dict[str, int]:
    """Runs one camera session until the device misbehaves.

    Builds the pipeline, then pumps frames into the segments until a stall,
    a device error, or shutdown. Mutates and returns the frame ids.
    """
    info = pick_device()
    if info is None:
        raise RuntimeError("no OAK devices found")

    log.info("opening %s (%s)", info.name, info.protocol)

    with dai.Pipeline(dai.Device(info)) as pipeline:
        # Before the pipeline build, which costs seconds and buries anything
        # logged under it: a bad link is worth knowing about immediately, in
        # the terminal and in Unity.
        check_link(pipeline.getDefaultDevice(), info)
        set_status(STATUS_STARTING)

        cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        # Undistorted: the wide-lens RGB is heavily distorted, and ImageAlign
        # outputs undistorted depth -- both sides must agree or colours slide
        # off the points toward the frame edges.
        rgb_out = cam_rgb.requestOutput((W, H), dai.ImgFrame.Type.RGB888i,
                                        fps=FPS, enableUndistortion=True)

        left  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        stereo = pipeline.create(dai.node.StereoDepth).build(
            left=left.requestOutput((W, H), fps=FPS),
            right=right.requestOutput((W, H), fps=FPS))
        preset = getattr(dai.node.StereoDepth.PresetMode, STEREO_PRESET,
                         dai.node.StereoDepth.PresetMode.ACCURACY)
        stereo.setDefaultProfilePreset(preset)
        log.info("stereo preset: %s", preset.name)

        # One pixel grid for everything: depth warped into the RGB camera's
        # perspective. setDepthAlign(CAM_A) is silently ineffective on RVC4,
        # so alignment goes through an explicit ImageAlign node instead.
        align = pipeline.create(dai.node.ImageAlign)
        stereo.depth.link(align.input)
        rgb_out.link(align.inputAlignTo)

        configure_depth_filters(stereo)

        pose_enabled = POSE_SOURCE != "none"
        pose_on_left = POSE_SOURCE == "left"

        nn: Any = None
        kp_to_frame: KpMapper | None = None
        nn_w: int | None = 0
        nn_h: int | None = 0
        if pose_enabled:
            # Resolve the model up front to learn its input size. Handing the
            # Camera node straight to ParsingNeuralNetwork lets depthai pick
            # the resize, and its default is a centre CROP: at 1280x800 ->
            # 640x352 that cut ~50 px off the top and bottom -- heads and
            # ankles -- and skewed every keypoint scaled by full-frame height.
            # An explicit LETTERBOX output keeps the whole FOV; make_kp_mapper
            # undoes the pads.
            nn_archive = resolve_pose_archive(pipeline)
            nn_w, nn_h = nn_archive.getInputWidth(), nn_archive.getInputHeight()
            if not nn_w or not nn_h:
                raise RuntimeError(f"model '{POSE_MODEL}' lacks a static input size")
            log.info("pose model %s (input %dx%d)", POSE_MODEL, nn_w, nn_h)
            kp_to_frame = make_kp_mapper(nn_w, nn_h)

            # The RGB pose input is undistorted to match the aligned-depth
            # grid (same as rgb_out); the left path stays raw so keypoints
            # agree with the raw CAM_B intrinsics used by the B->A remap.
            pose_cam = left if pose_on_left else cam_rgb
            pose_out = pose_cam.requestOutput(
                (nn_w, nn_h), dai.ImgFrame.Type.BGR888i, fps=FPS,
                resizeMode=dai.ImgResizeMode.LETTERBOX,
                enableUndistortion=None if pose_on_left else True)
            nn = pipeline.create(ParsingNeuralNetwork).build(pose_out, nn_archive)
        else:
            log.info("on-device pose disabled -- pose_server.py owns the "
                     "pose segment")

        # Small queues, non-blocking: we always want the newest frame, and a
        # stalled consumer must never apply back-pressure to the device.
        q_depth = align.outputAligned.createOutputQueue(maxSize=2, blocking=False)
        q_rgb   = rgb_out.createOutputQueue(maxSize=2, blocking=False)
        q_pose  = nn.out.createOutputQueue(maxSize=2, blocking=False) \
                  if pose_enabled else None
        # Left-frame keypoints need left-frame depth: tap the raw (pre-align,
        # rectified-left) depth as well. Post-processing filters still apply.
        q_raw   = stereo.depth.createOutputQueue(maxSize=2, blocking=False) \
                  if pose_on_left else None

        pipeline.start()

        device = pipeline.getDefaultDevice()
        calib = device.readCalibration()
        K = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, W, H)
        write_headers(K[0][0], K[1][1], K[0][2], K[1][2])
        log.info("intrinsics (CAM_A-aligned) fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                 K[0][0], K[1][1], K[0][2], K[1][2])

        set_ir_emitters(device)

        remap: Remap | None = None
        if pose_on_left:
            remap = make_left_to_cam_a_remap(calib, K)
        elif pose_enabled:
            log.info("pose on RGB")

        set_status(STATUS_STREAMING)

        depth_frame = None          # newest CAM_A-aligned depth (the SHM stream)
        raw_depth   = None          # newest rectified-left depth (keypoint z)
        rgb_tx      = None          # newest rgb ImgTransformation (rgb pose path)
        tx_mapper_logged = False
        last_frame_at = time.time()
        fps_n, fps_t0 = 0, time.time()

        while _running and pipeline.isRunning():
            got_any = False

            pkt: Any = q_depth.tryGet()
            if pkt is not None:
                depth_frame = np.ascontiguousarray(pkt.getFrame(), dtype=np.uint16)
                ids["depth"] += 1
                off = HEADER_SIZE + (ids["depth"] % NBUF) * DEPTH_SZ
                buf_depth[off:off + DEPTH_SZ] = depth_frame.tobytes()
                nsk.commit_frame(buf_depth, ids["depth"])
                last_frame_at = time.time()
                got_any = True

                fps_n += 1
                if fps_n >= 120:
                    log.info("%.1f fps  (depth %d, rgb %d, pose %d)%s",
                             fps_n / (time.time() - fps_t0),
                             ids["depth"], ids["rgb"], ids["pose"],
                             link_suffix())
                    fps_n, fps_t0 = 0, time.time()

            pkt = q_rgb.tryGet()
            if pkt is not None:
                rgb_tx = pkt.getTransformation()
                ids["rgb"] += 1
                off = HEADER_SIZE + (ids["rgb"] % NBUF) * RGB_SZ
                buf_rgb[off:off + RGB_SZ] = np.ascontiguousarray(
                    pkt.getFrame(), dtype=np.uint8).tobytes()
                nsk.commit_frame(buf_rgb, ids["rgb"])
                got_any = True

            if q_raw is not None:
                pkt = q_raw.tryGet()
                if pkt is not None:
                    raw_depth = np.ascontiguousarray(pkt.getFrame(), dtype=np.uint16)
                    got_any = True

            pkt = q_pose.tryGet() if q_pose is not None else None
            pose_depth = raw_depth if pose_on_left else depth_frame
            if pkt is not None and pose_depth is not None:
                ids["pose"] += 1
                # RGB path: undo the letterbox through the real transformations
                # whenever both sides carry one (exact even when the NN input
                # and rgb_out cover different sensor FOVs); analytic fallback
                # otherwise. Left path: analytic is exact, and the B->A remap
                # expects raw-left pixels anyway.
                # pose_enabled built the mapper and resolved the NN input size
                to_frame = kp_to_frame
                assert to_frame is not None and nn_w is not None and nn_h is not None
                if not pose_on_left and rgb_tx is not None:
                    pose_tx = pkt.getTransformation()
                    if pose_tx is not None:
                        to_frame = make_tx_mapper(pose_tx, rgb_tx, nn_w, nn_h)
                        if not tx_mapper_logged:
                            log.info("rgb pose keypoints remapped via "
                                     "ImgTransformation (FOV-exact)")
                            tx_mapper_logged = True
                write_pose(pkt.detections, pose_depth, ids["pose"],
                           to_frame, remap)
                got_any = True

            if not got_any:
                if time.time() - last_frame_at > STALE_AFTER:
                    raise RuntimeError(
                        f"no depth frame for {STALE_AFTER:.0f}s -- device stalled")
                time.sleep(0.001)

    return ids


# --------------------------------------------------------- pipeline steps ---
def pick_device() -> dai.DeviceInfo | None:
    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        return None
    if PREFER_USB:
        for d in devices:
            if d.protocol == dai.XLinkProtocol.X_LINK_USB_EP:
                return d
    return devices[0]


_link = LINK_UNKNOWN                # negotiated link of the open device


def check_link(device: dai.Device, info: dai.DeviceInfo) -> None:
    """Publishes the negotiated link speed and shouts when it is degraded.

    A USB-C plug seated the wrong way round renegotiates as USB2: nothing
    looks wrong, everything runs at roughly a quarter speed, and the only
    hint is one firmware line that the first fps report buries seconds
    later. Hence the three-line banner here, the reminder on every fps line
    (see link_suffix), and the byte Unity reads out of the status word.
    """
    global _link

    if info.protocol != dai.XLinkProtocol.X_LINK_USB_EP:
        _link = LINK_ETHERNET       # not a USB link; speed is not ours to judge
        log.info("link: %s (%s)", LINK_NAMES[_link], info.protocol)
        return

    _link = USB_SPEED_LINKS.get(device.getUsbSpeed().name, LINK_UNKNOWN)
    if not nsk.link_is_degraded(_link):
        log.info("USB link: %s", LINK_NAMES[_link])
        return

    log.warning("!!!!!!!!!!!!!!!!!!!!!  DEGRADED USB LINK  !!!!!!!!!!!!!!!!!!!!!")
    log.warning("!!  negotiated %s, expected USB3 (SUPER) -- everything "
                "below runs at ~1/4 speed", LINK_NAMES[_link])
    log.warning("!!  FIX: unplug the USB-C connector and replug it FLIPPED "
                "180 degrees (either end)")


def link_suffix() -> str:
    """Tail for the fps line while the link is degraded. The startup banner
    scrolls away within a minute; the fps lines are what a puzzled human
    actually reads."""
    if not nsk.link_is_degraded(_link):
        return ""
    return f"   <<<< {LINK_NAMES[_link]} LINK -- REPLUG THE USB-C FLIPPED"


def configure_depth_filters(stereo: dai.node.StereoDepth) -> None:
    """Enables on-device depth post-processing. Temporal is the flicker
    killer: static scenes stop shimmering. Runs on the RVC4, costs the
    host nothing.

    The RVC4 presets ship every filter disabled, so this function is what
    turns them on at all.

    Wobble -- points oscillating in depth on a scene that is not moving --
    is what these settings fight, and the two ways to fight it are not
    equal. Averaging over time (alpha) steadies a static scene and smears a
    moving person by exactly the same factor. Rejecting untrustworthy
    pixels (delta, confidence) costs fill rate instead of lag. Prefer the
    second: a person walking through the room must not smear.
    """
    pp = stereo.initialConfig.postProcessing
    pp.temporalFilter.enable = True
    pp.temporalFilter.alpha = 0.35          # lower = smoother, more lag
    # DISPARITY units, not millimetres. The device's own "auto" is three
    # integer disparity levels, which at RVC4's fixed 4 subpixel bits is
    # 3 * 16 = 48. Below that the filter disengages more eagerly than the
    # default and does LESS than leaving it alone; above it, more pixels
    # keep averaging and moving edges start to smear.
    pp.temporalFilter.delta = 48
    pp.temporalFilter.persistencyMode = \
        dai.StereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_2_IN_LAST_4
    pp.speckleFilter.enable = True          # kills lone flying pixels
    pp.speckleFilter.speckleRange = 50
    pp.thresholdFilter.minRange = 300       # mm, matches ValidRange in Unity
    pp.thresholdFilter.maxRange = 12000

    # Spatial filtering is intra-frame: it cannot lower frame-to-frame
    # variance, so it buys no stability at all -- it only softens the
    # surfaces we want crisp. Left off deliberately.
    pp.spatialFilter.enable = False

    # Scores a pixel down when its neighbourhood moves about between frames,
    # and INVALIDATES rather than smooths: stability with no lag, paid for in
    # fill rate. The one anti-wobble knob free of smear, so reach for it
    # first when the cloud shimmers. Weight is 4 out of 32 by default;
    # raising it makes temporal instability count for more against a pixel.
    # motionVectorConfidenceThreshold (0..3, default 1) is the second lever:
    # higher rejects more variance and costs more fill.
    stereo.initialConfig.confidenceMetrics.motionVectorConfidenceWeight = 16


def resolve_pose_archive(pipeline: dai.Pipeline) -> dai.NNArchive:
    """Loads POSE_MODEL as an NNArchive: a local .tar.xz path if one
    exists, otherwise a HubAI slug fetched for this device's platform."""
    if os.path.exists(POSE_MODEL):
        return dai.NNArchive(Path(POSE_MODEL))
    desc = dai.NNModelDescription(POSE_MODEL)
    desc.platform = pipeline.getDefaultDevice().getPlatformAsString()
    return dai.NNArchive(dai.getModelFromZoo(desc))


def set_ir_emitters(device: dai.Device) -> None:
    """Applies IR_DOT / IR_FLOOD and logs what the device accepted."""
    if IR_DOT > 0.0 or IR_FLOOD > 0.0:
        try:
            ok = device.setIrLaserDotProjectorIntensity(IR_DOT)
            ok = device.setIrFloodLightIntensity(IR_FLOOD) and ok
            log.info("IR emitters: dot %.0f%%, flood %.0f%%%s",
                     IR_DOT * 100, IR_FLOOD * 100,
                     "" if ok else "  (device reports no emitters)")
        except RuntimeError as exc:
            log.warning("IR emitters unavailable on this device: %s", exc)
    else:
        log.info("IR emitters off")


def make_left_to_cam_a_remap(calib: dai.CalibrationHandler,
                             K: list[list[float]]) -> Remap:
    """Builds the (px, py, z) -> (px, py, z) remap from raw CAM_B pixels
    into CAM_A pixels + metres, so left-mono keypoints honour the same
    stream contract as rgb ones. K is the CAM_A intrinsic matrix."""
    KB = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, W, H)
    fxB, fyB, cxB, cyB = KB[0][0], KB[1][1], KB[0][2], KB[1][2]
    fxA, fyA, cxA, cyA = K[0][0], K[1][1], K[0][2], K[1][2]
    E = np.array(calib.getCameraExtrinsics(
        dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_A),
        dtype=np.float64)
    # depthai extrinsic translations are centimetres; metres if someone
    # fixes that upstream. The stereo baseline is a few cm, so scale
    # by magnitude.
    if np.linalg.norm(E[:3, 3]) > 1.0:
        E[:3, 3] /= 100.0
    R, t = E[:3, :3], E[:3, 3]
    log.info("pose on LEFT mono; B->A baseline %.1f mm",
             np.linalg.norm(t) * 1000)

    def remap(px: float, py: float, z: float) -> tuple[float, float, float]:
        p = R @ np.array([(px - cxB) * z / fxB,
                          (py - cyB) * z / fyB, z]) + t
        if p[2] <= 0.0:
            return 0.0, 0.0, 0.0
        return (fxA * p[0] / p[2] + cxA,
                fyA * p[1] / p[2] + cyA, p[2])

    return remap


# ---------------------------------------------------------- pose mapping ----
_kp_convention_logged = False


def make_kp_mapper(nn_w: int, nn_h: int) -> KpMapper:
    """Returns kp -> (x, y) in full-frame pixels, undoing the letterbox
    analytically -- assumes the NN input letterboxes exactly the (W, H)
    frame's FOV. True for the left mono (natively 1280x800), NOT guaranteed
    for the RGB path, where resize modes act on the sensor FOV per output
    (4:3-class sensor: rgb_out crops to 16:10, the NN letterbox pads the
    full sensor). The RGB path therefore prefers make_tx_mapper and only
    falls back here.

    Keypoints live in the NN's letterboxed input frame (the parser never
    remaps them). depthai-nodes has also flip-flopped between normalised and
    pixel coordinates; detect that once at runtime instead of trusting docs.
    """
    scale, pad_x, pad_y = nsk.letterbox_transform(W, H, nn_w, nn_h)

    def to_frame(kp: Any) -> tuple[float, float]:
        global _kp_convention_logged
        x, y = kp.imageCoordinates.x, kp.imageCoordinates.y
        norm = abs(x) <= 2.0 and abs(y) <= 2.0
        if not _kp_convention_logged:
            log.info("keypoints arrive %s; undoing letterbox %dx%d -> %dx%d "
                     "(scale %.3f, pad %.1f/%.1f)",
                     "normalised" if norm else "in pixels",
                     nn_w, nn_h, W, H, scale, pad_x, pad_y)
            _kp_convention_logged = True
        if norm:
            x, y = x * nn_w, y * nn_h
        return (x - pad_x) / scale, (y - pad_y) / scale

    return to_frame


def make_tx_mapper(pose_tx: dai.ImgTransformation, target_tx: dai.ImgTransformation,
                   nn_w: int, nn_h: int) -> KpMapper:
    """Returns kp -> full-frame pixels via depthai's own ImgTransformations.

    The parser forwards the true source->NN-input transformation on the
    detections message; the rgb frames carry theirs. Remapping through the
    pair is exact by construction -- sensor crops, scales and pads included --
    where the analytic undo has to guess the sensor geometry.
    """
    def to_frame(kp: Any) -> tuple[float, float]:
        x, y = kp.imageCoordinates.x, kp.imageCoordinates.y
        if abs(x) <= 2.0 and abs(y) <= 2.0:              # normalised
            x, y = x * nn_w, y * nn_h
        p = pose_tx.remapPointTo(target_tx, dai.Point2f(x, y))
        return p.x, p.y

    return to_frame


def write_pose(detections: Any, depth_frame: DepthMap, frame_id: int,
               to_frame: KpMapper, remap: Remap | None = None) -> None:
    """Fills the pose segment: top MAX_PERSONS detections by confidence.

    `to_frame` maps a parser keypoint to full-frame pixels (letterbox undo);
    depth_frame must be in that same frame. When the NN runs on the left mono
    camera, `remap` converts (px, py, z) from the left frame into CAM_A
    pixels + metres so the stream contract never changes. The depth lift
    (nsk.lift_joints) samples inboard along each bone; the edge gate and
    serialization live in nsk.write_pose_slots.
    """
    dets = sorted(detections, key=lambda d: d.confidence, reverse=True)

    persons: list[nsk.Person] = []
    for det in dets[:MAX_PERSONS]:
        kps = det.getKeypoints()
        points: list[tuple[float, float, float]] = []    # px, py, conf
        for i in range(NUM_KP):
            if i < len(kps):
                px, py = to_frame(kps[i])
                points.append((px, py, float(kps[i].confidence)))
            else:
                points.append((0.0, 0.0, 0.0))

        joints = nsk.lift_joints(depth_frame, points)
        if remap is not None:
            joints = [(*remap(px, py, z), conf) if z > 0.0 else (px, py, z, conf)
                      for px, py, z, conf in joints]
        persons.append((det.confidence, joints))

    nsk.write_pose_slots(buf_pose, persons, frame_id)


# ------------------------------------------------------------ shared mem ----
# Created by main(), not at import, so tests can import this module freely.
# The SharedMemory handles stay Optional for the __main__ cleanup; the
# buffers are bound once, non-Optionally, in create_segments() -- using one
# before then is a NameError, which is exactly as loud as it should be.
shm_depth: SharedMemory | None = None
shm_rgb:   SharedMemory | None = None
shm_pose:  SharedMemory | None = None
buf_depth: memoryview
buf_rgb:   memoryview
buf_pose:  memoryview

_reconnects = 0


def create_segments() -> None:
    global shm_depth, shm_rgb, shm_pose, buf_depth, buf_rgb, buf_pose
    shm_depth = _open_logged(SHM_DEPTH, HEADER_SIZE + NBUF * DEPTH_SZ)
    shm_rgb   = _open_logged(SHM_RGB,   HEADER_SIZE + NBUF * RGB_SZ)
    shm_pose  = _open_logged(SHM_POSE,  nsk.POSE_SZ)
    # typeshed types .buf as 'memoryview | None'; on an open segment it never is
    buf_depth = cast(memoryview, shm_depth.buf)
    buf_rgb   = cast(memoryview, shm_rgb.buf)
    buf_pose  = cast(memoryview, shm_pose.buf)


def _open_logged(name: str, size: int) -> SharedMemory:
    seg, created = nsk.open_segment(name, size)
    log.info("%s shared memory '%s' (%.1f MB)",
             "created" if created else "attached to", name, size / 1e6)
    return seg


def write_headers(fx: float = 0.0, fy: float = 0.0,
                  cx: float = 0.0, cy: float = 0.0) -> None:
    intr = (fx, fy, cx, cy)
    nsk.write_frame_header(buf_depth, W, H, 2, NBUF, intr)
    nsk.write_frame_header(buf_rgb,   W, H, 3, NBUF, intr)
    # In host-backend mode the pose segment belongs to pose_server: never
    # blank its intrinsics at startup (consumers divide by fx while
    # pose_server's committed frames are still readable). Real intrinsics
    # from a connected camera are still shared -- pose_server copies them.
    if POSE_SOURCE != "none" or fx != 0.0:
        nsk.write_pose_header(buf_pose, intr)


def set_status(status: int) -> None:
    word = nsk.pack_status(status, _link)
    struct.pack_into("<2I", buf_depth, 56, word, _reconnects)
    struct.pack_into("<2I", buf_rgb,   56, word, _reconnects)
    struct.pack_into("<I",  buf_pose,  44, word)


# ------------------------------------------------------ interactive setup ---
def configure_interactively() -> None:
    """Terminal profile picker for development. Overrides the env-derived
    config for this run only; a run without a terminal keeps env config."""
    global POSE_SOURCE, IR_DOT, IR_FLOOD

    print("\n=== nice-stream setup ===")
    print("  [1] non-IR camera (dev)      pose=rgb   dot=0    flood=0")
    print("  [2] IR camera (gallery)      pose=left  dot=0.8  flood=0.5")
    print("  [3] host pose backend        pose=none  (run pose_server.py; IR from env)")
    print(f"  [4] env config               pose={POSE_SOURCE}  "
          f"dot={IR_DOT:g}  flood={IR_FLOOD:g}")
    print("  [5] custom")
    choice = str(_ask("profile", "4")).strip()

    if choice == "1":
        POSE_SOURCE, IR_DOT, IR_FLOOD = "rgb", 0.0, 0.0
    elif choice == "2":
        POSE_SOURCE, IR_DOT, IR_FLOOD = "left", 0.8, 0.5
    elif choice == "3":
        POSE_SOURCE = "none"
    elif choice == "5":
        while True:
            POSE_SOURCE = _ask("pose source (left/rgb/none)", POSE_SOURCE)
            if POSE_SOURCE in ("left", "rgb", "none"):
                break
            print("  ! must be 'left', 'rgb' or 'none'")
        IR_DOT   = _ask("IR dot intensity 0..1", IR_DOT, float)
        IR_FLOOD = _ask("IR flood intensity 0..1", IR_FLOOD, float)

    if POSE_SOURCE != "none":                # model is moot without a device NN
        _choose_model()


def _choose_model() -> None:
    """Model menu, shown regardless of camera profile: pick a known-good
    slug by number, 'c' for a custom one, Enter keeps the current model."""
    global POSE_MODEL

    print("\n  pose model:")
    for i, (slug, blurb) in enumerate(KNOWN_POSE_MODELS, 1):
        mark = "   <- current" if slug == POSE_MODEL else ""
        print(f"    [{i}] {slug}   ({blurb}){mark}")
    print("    [c] custom slug")
    choice = str(_ask("model (Enter = keep current)", "keep")).strip().lower()

    if choice in ("", "keep"):
        return
    if choice == "c":
        POSE_MODEL = _ask("HubAI slug or NNArchive path", POSE_MODEL)
    elif choice.isdigit() and 1 <= int(choice) <= len(KNOWN_POSE_MODELS):
        POSE_MODEL = KNOWN_POSE_MODELS[int(choice) - 1][0]
    else:
        print(f"  ! unknown choice; keeping {POSE_MODEL}")


def _ask(label: str, current: Any, parse: type[Any] = str) -> Any:
    """One prompt; Enter (or EOF -- no terminal attached) keeps `current`."""
    try:
        raw = input(f"  {label} [{current}]: ").strip()
    except EOFError:
        return current
    if not raw:
        return current
    try:
        return parse(raw)
    except ValueError:
        print(f"  ! not a valid {parse.__name__}; keeping {current}")
        return current


# ------------------------------------------------------------- shutdown ----
_running = True


def _stop(signum: int, _frame: FrameType | None) -> None:
    global _running
    log.info("signal %s -- shutting down", signum)
    _running = False


signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)


if __name__ == "__main__":
    try:
        main()
    finally:
        for seg in (shm_depth, shm_rgb, shm_pose):
            if seg is None:
                continue
            seg.close()
            try:
                seg.unlink()
            except FileNotFoundError:
                pass
