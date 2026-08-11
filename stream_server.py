"""
nice-stream-oak: depth + rgb + multipose -> shared memory.

Supersedes depth_server.py. One pipeline, three segments:

  nice_stream_depth   u16 depth, millimetres, aligned to the RGB camera
  nice_stream_rgb     RGB888 interleaved, same resolution & intrinsics as depth
  nice_stream_pose    3D skeletons: 2D keypoints from a configurable HubAI
                      pose model (default YOLOv8-large) lifted to metres by
                      sampling the aligned depth map

Because depth is aligned to CAM_A, all three streams share one set of
intrinsics and one pixel grid: a depth pixel, its colour and a keypoint at the
same (x, y) refer to the same ray. Unity does the unprojection.

Designed to run unattended for weeks; the camera is assumed to be unreliable.
Segments are created once and never torn down while this process lives.
Consumers may attach before the camera exists and survive it vanishing.

Frame header (little endian, 64 bytes) -- depth & rgb segments:
   0  uint32  magic 'NSK1'
   4  uint32  version
   8  uint32  width
  12  uint32  height
  16  uint32  bytes_per_pixel
  20  uint32  buffer_count
  24  uint64  frame_id        <- written last, acts as the commit
  32  float64 timestamp (unix seconds)
  40  float32 fx, fy, cx, cy
  56  uint32  status          <- 0 starting, 1 streaming, 2 reconnecting
  60  uint32  reconnect_count

Pose header (little endian, 64 bytes), magic 'NSKP':
   0  uint32  magic
   4  uint32  version
   8  uint32  max_persons
  12  uint32  num_keypoints   (17, COCO order)
  16  uint32  floats_per_kp   (4: x_px, y_px, z_m, confidence)
  20  uint32  person_stride   (bytes)
  24  uint64  frame_id        <- written last, acts as the commit
  32  float64 timestamp
  40  uint32  num_persons     (valid persons this frame)
  44  uint32  status
  48  float32 fx, fy, cx, cy

Pose body: max_persons fixed slots, each person_stride bytes:
   0  float32 detection confidence (0 -> slot empty)
   4  float32 reserved x3
  16  17 x (float32 x_px, y_px, z_m, conf)   z_m = 0 -> no depth at that joint

COCO keypoint order: nose, l_eye, r_eye, l_ear, r_ear, l_shoulder, r_shoulder,
l_elbow, r_elbow, l_wrist, r_wrist, l_hip, r_hip, l_knee, r_knee, l_ankle,
r_ankle.
"""

import logging
import os
import signal
import struct
import sys
import time
from multiprocessing import shared_memory

import numpy as np
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork

# ---------------------------------------------------------------- config ----
SHM_DEPTH = os.environ.get("NICE_STREAM_SHM", "nice_stream_depth")
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
POSE_SOURCE = os.environ.get("NICE_STREAM_POSE_SOURCE", "left")

# Stereo quality preset. HIGH_DETAIL maximizes point-cloud quality (subpixel,
# heavy post-processing) at some fps cost; DEFAULT is the faster fallback.
STEREO_PRESET = os.environ.get("NICE_STREAM_STEREO_PRESET", "HIGH_DETAIL")

# 1 = skip the graceful device teardown on shutdown (dev iterations only:
# the clean close is what protects the OAK-4 from wedging into a state that
# needs a replug, so leave this off in production).
FAST_EXIT = os.environ.get("NICE_STREAM_FAST_EXIT", "0") == "1"

# Dev convenience: --interactive (or NICE_STREAM_INTERACTIVE=1, e.g. in a
# PyCharm run configuration) asks for the camera profile in the terminal on
# startup. Production runs leave this off and configure via env vars.
INTERACTIVE = ("--interactive" in sys.argv
               or os.environ.get("NICE_STREAM_INTERACTIVE", "0") == "1")
NBUF   = 3
HDR    = 64
MAGIC_FRAME = 0x314B534E            # 'NSK1'
MAGIC_POSE  = 0x504B534E            # 'NSKP'
VERSION     = 1

# Any HubAI slug depthai-nodes can parse (the YOLO-pose family is the safe
# bet). Input size is read from the model archive; frames are letterboxed
# into it and keypoints mapped back to full-frame pixels.
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
MAX_PERSONS  = 8
NUM_KP       = 17
FLOATS_PER_KP = 4
PERSON_STRIDE = 16 + NUM_KP * FLOATS_PER_KP * 4     # 288 bytes
DEPTH_PATCH   = 2                   # median over (2k+1)^2 px around a keypoint
JOINT_DEV_MAX = 1.5                 # m; joints further than this from the
                                    # person's median depth are depth-edge
                                    # glitches (hand sampled the wall behind)

DEPTH_SZ = W * H * 2
RGB_SZ   = W * H * 3
POSE_SZ  = HDR + MAX_PERSONS * PERSON_STRIDE

PREFER_USB      = True
RETRY_DELAY     = 3.0
RETRY_DELAY_MAX = 30.0
STALE_AFTER     = 5.0               # no depth frame for this long -> dead

STATUS_STARTING     = 0
STATUS_STREAMING    = 1
STATUS_RECONNECTING = 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("stream_server.log", encoding="utf-8")],
)
log = logging.getLogger("nice-stream")


# ------------------------------------------------------ interactive setup ---
def _ask(label, current, cast=str):
    """One prompt; Enter (or EOF -- no terminal attached) keeps `current`."""
    try:
        raw = input(f"  {label} [{current}]: ").strip()
    except EOFError:
        return current
    if not raw:
        return current
    try:
        return cast(raw)
    except ValueError:
        print(f"  ! not a valid {cast.__name__}; keeping {current}")
        return current


def _choose_model():
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
        POSE_MODEL = _ask("HubAI slug", POSE_MODEL)
    elif choice.isdigit() and 1 <= int(choice) <= len(KNOWN_POSE_MODELS):
        POSE_MODEL = KNOWN_POSE_MODELS[int(choice) - 1][0]
    else:
        print(f"  ! unknown choice; keeping {POSE_MODEL}")


def configure_interactively():
    """Terminal profile picker for development. Overrides the env-derived
    config for this run only; a run without a terminal keeps env config."""
    global POSE_SOURCE, IR_DOT, IR_FLOOD

    print("\n=== nice-stream setup ===")
    print("  [1] non-IR camera (dev)      pose=rgb   dot=0    flood=0")
    print("  [2] IR camera (gallery)      pose=left  dot=0.8  flood=0.5")
    print(f"  [3] env config               pose={POSE_SOURCE}  "
          f"dot={IR_DOT:g}  flood={IR_FLOOD:g}")
    print("  [4] custom")
    choice = str(_ask("profile", "3")).strip()

    if choice == "1":
        POSE_SOURCE, IR_DOT, IR_FLOOD = "rgb", 0.0, 0.0
    elif choice == "2":
        POSE_SOURCE, IR_DOT, IR_FLOOD = "left", 0.8, 0.5
    elif choice == "4":
        while True:
            POSE_SOURCE = _ask("pose source (left/rgb)", POSE_SOURCE)
            if POSE_SOURCE in ("left", "rgb"):
                break
            print("  ! must be 'left' or 'rgb'")
        IR_DOT   = _ask("IR dot intensity 0..1", IR_DOT, float)
        IR_FLOOD = _ask("IR flood intensity 0..1", IR_FLOOD, float)

    _choose_model()


# ------------------------------------------------------------ shared mem ----
def open_segment(name, size):
    try:
        seg = shared_memory.SharedMemory(name=name, create=True, size=size)
        log.info("created shared memory '%s' (%.1f MB)", name, size / 1e6)
    except FileExistsError:
        seg = shared_memory.SharedMemory(name=name)
        log.info("attached to existing shared memory '%s'", name)
    return seg


shm_depth = open_segment(SHM_DEPTH, HDR + NBUF * DEPTH_SZ)
shm_rgb   = open_segment(SHM_RGB,   HDR + NBUF * RGB_SZ)
shm_pose  = open_segment(SHM_POSE,  POSE_SZ)

_reconnects = 0


def write_frame_headers(fx=0.0, fy=0.0, cx=0.0, cy=0.0):
    for buf, bpp in ((shm_depth.buf, 2), (shm_rgb.buf, 3)):
        struct.pack_into("<6I", buf, 0, MAGIC_FRAME, VERSION, W, H, bpp, NBUF)
        struct.pack_into("<4f", buf, 40, fx, fy, cx, cy)
    struct.pack_into("<6I", shm_pose.buf, 0, MAGIC_POSE, VERSION,
                     MAX_PERSONS, NUM_KP, FLOATS_PER_KP, PERSON_STRIDE)
    struct.pack_into("<4f", shm_pose.buf, 48, fx, fy, cx, cy)


def set_status(status):
    struct.pack_into("<2I", shm_depth.buf, 56, status, _reconnects)
    struct.pack_into("<2I", shm_rgb.buf,   56, status, _reconnects)
    struct.pack_into("<I",  shm_pose.buf,  44, status)


def commit_frame(buf, frame_id):
    struct.pack_into("<Qd", buf, 24, frame_id, time.time())


# ------------------------------------------------------------- shutdown ----
_running = True


def _stop(signum, _frame):
    global _running
    log.info("signal %s -- shutting down", signum)
    _running = False
    if FAST_EXIT:
        log.warning("FAST_EXIT: skipping graceful device teardown "
                    "(dev only -- the camera may need a replug afterwards)")
        os._exit(130)


signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)


# --------------------------------------------------------- device picker ----
def pick_device():
    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        return None
    if PREFER_USB:
        for d in devices:
            if d.protocol == dai.XLinkProtocol.X_LINK_USB_EP:
                return d
    return devices[0]


# ------------------------------------------------------------------ pose ----
_kp_convention_logged = False


def make_kp_mapper(nn_w, nn_h):
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
    scale = min(nn_w / W, nn_h / H)
    pad_x = (nn_w - W * scale) * 0.5
    pad_y = (nn_h - H * scale) * 0.5

    def to_frame(kp):
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


def make_tx_mapper(pose_tx, target_tx, nn_w, nn_h):
    """kp -> full-frame pixels via depthai's own ImgTransformations.

    The parser forwards the true source->NN-input transformation on the
    detections message; the rgb frames carry theirs. Remapping through the
    pair is exact by construction -- sensor crops, scales and pads included --
    where the analytic undo has to guess the sensor geometry.
    """
    def to_frame(kp):
        x, y = kp.imageCoordinates.x, kp.imageCoordinates.y
        if abs(x) <= 2.0 and abs(y) <= 2.0:              # normalised
            x, y = x * nn_w, y * nn_h
        p = pose_tx.remapPointTo(target_tx, dai.Point2f(x, y))
        return p.x, p.y

    return to_frame


def depth_at(depth_frame, px, py):
    """Median depth (metres) in a small patch; 0.0 when nothing valid."""
    if not (np.isfinite(px) and np.isfinite(py)):
        return 0.0          # NaN keypoint must not recycle the session
    x, y = int(round(px)), int(round(py))
    if not (0 <= x < W and 0 <= y < H):
        return 0.0
    x0, x1 = max(0, x - DEPTH_PATCH), min(W, x + DEPTH_PATCH + 1)
    y0, y1 = max(0, y - DEPTH_PATCH), min(H, y + DEPTH_PATCH + 1)
    patch = depth_frame[y0:y1, x0:x1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid)) * 0.001


def write_pose(detections, depth_frame, frame_id, to_frame, remap=None):
    """Fill the pose segment: top MAX_PERSONS detections by confidence.

    `to_frame` maps a parser keypoint to full-frame pixels (letterbox undo);
    depth_frame must be in that same frame. When the NN runs on the left mono
    camera, `remap` converts (px, py, z) from the left frame into CAM_A
    pixels + metres so the stream contract never changes.
    """
    buf = shm_pose.buf
    dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    dets = dets[:MAX_PERSONS]

    for slot in range(MAX_PERSONS):
        base = HDR + slot * PERSON_STRIDE
        if slot >= len(dets):
            struct.pack_into("<f", buf, base, 0.0)      # empty slot
            continue
        det = dets[slot]
        struct.pack_into("<4f", buf, base, det.confidence, 0.0, 0.0, 0.0)
        kps = det.getKeypoints()

        joints = []                          # (px, py, z, conf) per keypoint
        for i in range(NUM_KP):
            if i < len(kps):
                px, py = to_frame(kps[i])
                z = depth_at(depth_frame, px, py) if kps[i].confidence > 0.3 else 0.0
                if remap is not None and z > 0.0:
                    px, py, z = remap(px, py, z)
                joints.append((px, py, z, kps[i].confidence))
            else:
                joints.append((0.0, 0.0, 0.0, 0.0))

        # Depth-edge gate: a keypoint on a silhouette boundary can sample the
        # background instead of the person. Anything implausibly far from the
        # person's own median depth loses its z (consumers hold last pose).
        zs = [j[2] for j in joints if j[2] > 0.0]
        if len(zs) >= 3:
            ref = float(np.median(zs))
            joints = [(px, py, z if (z == 0.0 or abs(z - ref) <= JOINT_DEV_MAX)
                       else 0.0, c)
                      for px, py, z, c in joints]

        for i, (px, py, z, c) in enumerate(joints):
            struct.pack_into("<4f", buf, base + 16 + i * 16, px, py, z, c)

    struct.pack_into("<I", buf, 40, len(dets))
    commit_frame(buf, frame_id)                          # commit last


# ------------------------------------------------------------ one session ---
def stream_once(ids):
    """Runs until the device misbehaves. Mutates and returns the frame ids."""
    info = pick_device()
    if info is None:
        raise RuntimeError("no OAK devices found")

    log.info("opening %s (%s)", info.name, info.protocol)

    with dai.Pipeline(dai.Device(info)) as pipeline:
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
                         dai.node.StereoDepth.PresetMode.HIGH_DETAIL)
        stereo.setDefaultProfilePreset(preset)
        log.info("stereo preset: %s", preset.name)

        # One pixel grid for everything: depth warped into the RGB camera's
        # perspective. setDepthAlign(CAM_A) is silently ineffective on RVC4,
        # so alignment goes through an explicit ImageAlign node instead.
        align = pipeline.create(dai.node.ImageAlign)
        stereo.depth.link(align.input)
        rgb_out.link(align.inputAlignTo)

        # On-device stabilisation. Temporal is the flicker killer: static
        # scenes stop shimmering. Runs on the RVC4, costs the host nothing.
        pp = stereo.initialConfig.postProcessing
        pp.temporalFilter.enable = True
        pp.temporalFilter.alpha = 0.35          # lower = smoother, more lag
        pp.temporalFilter.delta = 30            # mm; jumps larger than this pass through
        pp.temporalFilter.persistencyMode = \
            dai.StereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_2_IN_LAST_4
        pp.speckleFilter.enable = True          # kills lone flying pixels
        pp.speckleFilter.speckleRange = 50
        pp.thresholdFilter.minRange = 300       # mm, matches ValidRange in Unity
        pp.thresholdFilter.maxRange = 12000
        # Edge-preserving smoothing + small hole filling: visibly calmer
        # point-cloud surfaces; costs device compute we now have to spare.
        pp.spatialFilter.enable = True
        pp.spatialFilter.holeFillingRadius = 2
        pp.spatialFilter.numIterations = 1

        pose_on_left = POSE_SOURCE == "left"

        # Resolve the model up front to learn its input size. Handing the
        # Camera node straight to ParsingNeuralNetwork lets depthai pick the
        # resize, and its default is a centre CROP: at 1280x800 -> 640x352
        # that cut ~50 px off the top and bottom -- heads and ankles -- and
        # skewed every keypoint scaled by full-frame height. An explicit
        # LETTERBOX output keeps the whole FOV; make_kp_mapper undoes the pads.
        desc = dai.NNModelDescription(POSE_MODEL)
        desc.platform = pipeline.getDefaultDevice().getPlatformAsString()
        nn_archive = dai.NNArchive(dai.getModelFromZoo(desc))
        nn_w, nn_h = nn_archive.getInputWidth(), nn_archive.getInputHeight()
        if not nn_w or not nn_h:
            raise RuntimeError(f"model '{POSE_MODEL}' lacks a static input size")
        log.info("pose model %s (input %dx%d)", POSE_MODEL, nn_w, nn_h)
        kp_to_frame = make_kp_mapper(nn_w, nn_h)

        # The RGB pose input is undistorted to match the aligned-depth grid
        # (same as rgb_out); the left path stays raw so keypoints agree with
        # the raw CAM_B intrinsics used by the B->A remap below.
        pose_cam = left if pose_on_left else cam_rgb
        pose_out = pose_cam.requestOutput(
            (nn_w, nn_h), dai.ImgFrame.Type.BGR888i, fps=FPS,
            resizeMode=dai.ImgResizeMode.LETTERBOX,
            enableUndistortion=None if pose_on_left else True)
        nn = pipeline.create(ParsingNeuralNetwork).build(pose_out, nn_archive)

        # Small queues, non-blocking: we always want the newest frame, and a
        # stalled consumer must never apply back-pressure to the device.
        q_depth = align.outputAligned.createOutputQueue(maxSize=2, blocking=False)
        q_rgb   = rgb_out.createOutputQueue(maxSize=2, blocking=False)
        q_pose  = nn.out.createOutputQueue(maxSize=2, blocking=False)
        # Left-frame keypoints need left-frame depth: tap the raw (pre-align,
        # rectified-left) depth as well. Post-processing filters still apply.
        q_raw   = stereo.depth.createOutputQueue(maxSize=2, blocking=False) \
                  if pose_on_left else None

        pipeline.start()

        device = pipeline.getDefaultDevice()
        calib = device.readCalibration()
        K = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, W, H)
        write_frame_headers(K[0][0], K[1][1], K[0][2], K[1][2])
        log.info("intrinsics (CAM_A-aligned) fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                 K[0][0], K[1][1], K[0][2], K[1][2])

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

        remap = None
        if pose_on_left:
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

            def remap(px, py, z):
                p = R @ np.array([(px - cxB) * z / fxB,
                                  (py - cyB) * z / fyB, z]) + t
                if p[2] <= 0.0:
                    return 0.0, 0.0, 0.0
                return (fxA * p[0] / p[2] + cxA,
                        fyA * p[1] / p[2] + cyA, p[2])
        else:
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

            pkt = q_depth.tryGet()
            if pkt is not None:
                depth_frame = np.ascontiguousarray(pkt.getFrame(), dtype=np.uint16)
                ids["depth"] += 1
                off = HDR + (ids["depth"] % NBUF) * DEPTH_SZ
                shm_depth.buf[off:off + DEPTH_SZ] = depth_frame.tobytes()
                commit_frame(shm_depth.buf, ids["depth"])
                last_frame_at = time.time()
                got_any = True

                fps_n += 1
                if fps_n >= 120:
                    log.info("%.1f fps  (depth %d, rgb %d, pose %d)",
                             fps_n / (time.time() - fps_t0),
                             ids["depth"], ids["rgb"], ids["pose"])
                    fps_n, fps_t0 = 0, time.time()

            pkt = q_rgb.tryGet()
            if pkt is not None:
                rgb_tx = pkt.getTransformation()
                ids["rgb"] += 1
                off = HDR + (ids["rgb"] % NBUF) * RGB_SZ
                shm_rgb.buf[off:off + RGB_SZ] = np.ascontiguousarray(
                    pkt.getFrame(), dtype=np.uint8).tobytes()
                commit_frame(shm_rgb.buf, ids["rgb"])
                got_any = True

            if q_raw is not None:
                pkt = q_raw.tryGet()
                if pkt is not None:
                    raw_depth = np.ascontiguousarray(pkt.getFrame(), dtype=np.uint16)
                    got_any = True

            pkt = q_pose.tryGet()
            pose_depth = raw_depth if pose_on_left else depth_frame
            if pkt is not None and pose_depth is not None:
                ids["pose"] += 1
                # RGB path: undo the letterbox through the real transformations
                # whenever both sides carry one (exact even when the NN input
                # and rgb_out cover different sensor FOVs); analytic fallback
                # otherwise. Left path: analytic is exact, and the B->A remap
                # expects raw-left pixels anyway.
                to_frame = kp_to_frame
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


# ------------------------------------------------------------------ main ----
def main():
    global _reconnects

    if INTERACTIVE:
        configure_interactively()
    log.info("config: pose source=%s  model=%s  ir dot=%.2f flood=%.2f",
             POSE_SOURCE, POSE_MODEL, IR_DOT, IR_FLOOD)

    write_frame_headers()
    set_status(STATUS_STARTING)
    commit_frame(shm_depth.buf, 0)
    commit_frame(shm_rgb.buf, 0)
    commit_frame(shm_pose.buf, 0)

    ids = {"depth": 0, "rgb": 0, "pose": 0}
    delay = RETRY_DELAY

    while _running:
        try:
            ids = stream_once(ids)
            if not _running:
                break
            log.warning("session ended cleanly -- reopening")
        except Exception as exc:
            log.error("session failed: %s", exc)

        if not _running:
            break

        _reconnects += 1
        set_status(STATUS_RECONNECTING)
        log.info("reconnect #%d in %.0fs", _reconnects, delay)
        deadline = time.time() + delay
        while _running and time.time() < deadline:
            time.sleep(0.2)
        delay = min(delay * 1.5, RETRY_DELAY_MAX)

    set_status(STATUS_STARTING)
    log.info("stopped after %d reconnects (depth %d, rgb %d, pose %d frames)",
             _reconnects, ids["depth"], ids["rgb"], ids["pose"])


if __name__ == "__main__":
    try:
        main()
    finally:
        for seg in (shm_depth, shm_rgb, shm_pose):
            seg.close()
            try:
                seg.unlink()
            except FileNotFoundError:
                pass
