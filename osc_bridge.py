"""nice-stream pose -> OSC movement signals.

Reads the 'nice_stream_pose' shared-memory segment (the only segment this
bridge touches), derives movement metrics, and broadcasts them as one OSC
bundle per emitted frame. Meaning, not capture: skeletons in, gestures out.

The pose stream is identity-free (slots are confidence-ranked per frame), so
this bridge runs its own frame-to-frame tracker: greedy nearest-neighbour
association on smoothed centroids, persistent integer IDs, and a sticky
ID -> OSC-slot map so receiver addresses never reshuffle while a person stays
in the space. Brief dropouts (occlusion, crossing) are bridged by a grace
window.

OSC namespace (floats unless noted):
  /nice/group/count        int   tracked people
  /nice/group/centroid     x y z metres, camera frame (y up, z away from cam)
  /nice/group/spread       0..1  dispersion around centroid / room scale
  /nice/group/energy       m/s   summed smoothed per-person speed
  /nice/slot/<i>/present   int   0/1
  /nice/slot/<i>/id        int   persistent person id (while present)
  /nice/slot/<i>/position  x y z metres
  /nice/slot/<i>/speed     m/s   smoothed centroid speed
  /nice/slot/<i>/conf      0..1  detection confidence
  /nice/event/entered      slot id     (fires once)
  /nice/event/left         slot id     (fires once)

Sonic Pi: run with --port 4560; cues arrive as /osc*/nice/... (see
sonicpi_example.rb). Debug without Sonic Pi: osc_monitor.py.

Run with the venv (numpy not required, python-osc is):
  .venv/Scripts/python.exe osc_bridge.py --host 127.0.0.1 --port 9000 --verbose
"""

import argparse
import math
import struct
import time
from multiprocessing import shared_memory

from pythonosc import osc_bundle_builder, osc_message_builder, udp_client

HDR = 64
MAGIC = 0x504B534E  # 'NSKP'
MIN_JOINT_CONF = 0.3
MAX_SLOTS = 8


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1", help="OSC receiver address")
    p.add_argument("--port", type=int, default=9000,
                   help="receiver port (Sonic Pi: 4560)")
    p.add_argument("--rate", type=float, default=30.0, help="max emit Hz")
    p.add_argument("--room-scale", type=float, default=5.0,
                   help="metres; normalizes spread to 0..1")
    p.add_argument("--smooth", type=float, default=0.15,
                   help="seconds; EMA time constant for positions")
    p.add_argument("--gate", type=float, default=0.5,
                   help="metres; base association distance gate")
    p.add_argument("--gate-speed", type=float, default=1.5,
                   help="m/s; gate growth for tracks unseen for a while")
    p.add_argument("--grace", type=float, default=2.0,
                   help="seconds a lost track survives before 'left'")
    p.add_argument("--segment", default="nice_stream_pose")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def attach(name):
    while True:
        try:
            shm = shared_memory.SharedMemory(name=name)
            magic, _, max_p, n_kp, _, stride = struct.unpack_from("<6I", shm.buf, 0)
            if magic != MAGIC:
                raise RuntimeError(f"bad magic 0x{magic:08X}")
            K = struct.unpack_from("<4f", shm.buf, 48)
            print(f"attached '{name}': {max_p} slots, {n_kp} kp, fx={K[0]:.1f}")
            return shm, max_p, n_kp, stride, K
        except FileNotFoundError:
            print(f"waiting for '{name}' ...")
            time.sleep(1.0)


def read_frame(buf, max_p, n_kp, stride):
    """Seqlock read: pose is single-buffered; frame_id is the commit."""
    for _ in range(4):
        fid_before, = struct.unpack_from("<Q", buf, 24)
        if fid_before == 0:
            return 0, []
        raw = bytes(buf[HDR:HDR + max_p * stride])
        fid_after, = struct.unpack_from("<Q", buf, 24)
        if fid_before == fid_after:
            break
    else:
        return 0, []

    persons = []
    for slot in range(max_p):
        base = slot * stride
        conf, = struct.unpack_from("<f", raw, base)
        if conf <= 0.0:
            continue
        joints = [struct.unpack_from("<4f", raw, base + 16 + k * 16)
                  for k in range(n_kp)]
        persons.append((conf, joints))
    return fid_after, persons


def centroid_of(joints, K):
    """Confidence-weighted metric centroid of joints with valid depth."""
    fx, fy, cx, cy = K
    sw = sx = sy = sz = 0.0
    for px, py, z, c in joints:
        if z <= 0.0 or c < MIN_JOINT_CONF:
            continue
        sw += c
        sx += c * (px - cx) * z / fx
        sy += c * -(py - cy) * z / fy
        sz += c * z
    if sw == 0.0:
        return None
    return (sx / sw, sy / sw, sz / sw)


class Track:
    """One persistent person: smoothed position, speed, identity, OSC slot."""

    def __init__(self, tid, slot, pos, now):
        self.id = tid
        self.slot = slot
        self.pos = pos          # EMA-smoothed
        self.speed = 0.0        # EMA-smoothed, m/s
        self.conf = 0.0
        self.last_seen = now

    def update(self, pos, conf, now, tau):
        dt = max(1e-3, now - self.last_seen)
        k = min(1.0, dt / tau)
        prev = self.pos
        self.pos = tuple(p + (q - p) * k for p, q in zip(self.pos, pos))
        inst = math.dist(self.pos, prev) / dt
        self.speed += (inst - self.speed) * k
        self.conf = conf
        self.last_seen = now


class Tracker:
    """Greedy nearest-neighbour association with a grace window.

    Distances are measured on the floor plane (x, z): people's heights do not
    change, and vertical centroid noise (occluded legs) should not break
    identity.
    """

    def __init__(self, args):
        self.args = args
        self.tracks = []
        self.next_id = 0
        self.events = []        # ('entered'|'left', slot, id)

    def _free_slot(self):
        used = {t.slot for t in self.tracks}
        for s in range(MAX_SLOTS):
            if s not in used:
                return s
        return None

    def step(self, detections, now):
        """detections: list of (conf, centroid). Returns active tracks."""
        self.events.clear()

        # pair (distance, track, detection) greedily on the floor plane
        pairs = []
        for t in self.tracks:
            gate = self.args.gate + self.args.gate_speed * (now - t.last_seen)
            for di, (conf, c) in enumerate(detections):
                d = math.hypot(c[0] - t.pos[0], c[2] - t.pos[2])
                if d <= gate:
                    pairs.append((d, t, di))
        pairs.sort(key=lambda p: p[0])

        matched_tracks, matched_dets = set(), set()
        for d, t, di in pairs:
            if id(t) in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(id(t))
            matched_dets.add(di)
            conf, c = detections[di]
            t.update(c, conf, now, self.args.smooth)

        # unmatched detections -> new tracks (if a slot is free)
        for di, (conf, c) in enumerate(detections):
            if di in matched_dets:
                continue
            slot = self._free_slot()
            if slot is None:
                continue
            t = Track(self.next_id, slot, c, now)
            t.conf = conf
            self.next_id += 1
            self.tracks.append(t)
            self.events.append(("entered", t.slot, t.id))

        # expire tracks beyond the grace window
        survivors = []
        for t in self.tracks:
            if now - t.last_seen > self.args.grace:
                self.events.append(("left", t.slot, t.id))
            else:
                survivors.append(t)
        self.tracks = survivors
        return self.tracks


def main():
    args = parse_args()
    client = udp_client.SimpleUDPClient(args.host, args.port)
    shm, max_p, n_kp, stride, K = attach(args.segment)
    tracker = Tracker(args)

    last_fid = 0
    last_emit = 0.0

    print(f"emitting to {args.host}:{args.port} at <= {args.rate:.0f} Hz")
    while True:
        fid, persons = read_frame(shm.buf, max_p, n_kp, stride)
        now = time.time()
        if fid == last_fid or now - last_emit < 1.0 / args.rate:
            time.sleep(0.003)
            continue
        last_fid, last_emit = fid, now

        detections = []
        for conf, joints in persons:
            c = centroid_of(joints, K)
            if c is not None:
                detections.append((conf, c))

        tracks = tracker.step(detections, now)
        count = len(tracks)

        gx = gy = gz = spread = energy = 0.0
        if count:
            gx = sum(t.pos[0] for t in tracks) / count
            gy = sum(t.pos[1] for t in tracks) / count
            gz = sum(t.pos[2] for t in tracks) / count
            energy = sum(t.speed for t in tracks)
            if count > 1:
                var = sum((t.pos[0] - gx) ** 2 + (t.pos[2] - gz) ** 2
                          for t in tracks) / count
                spread = min(1.0, math.sqrt(var) / (args.room_scale * 0.5))

        bundle = osc_bundle_builder.OscBundleBuilder(
            osc_bundle_builder.IMMEDIATELY)

        def add(address, *values):
            msg = osc_message_builder.OscMessageBuilder(address=address)
            for v in values:
                msg.add_arg(v)
            bundle.add_content(msg.build())

        add("/nice/group/count", count)
        add("/nice/group/centroid", gx, gy, gz)
        add("/nice/group/spread", spread)
        add("/nice/group/energy", energy)

        by_slot = {t.slot: t for t in tracks}
        for s in range(MAX_SLOTS):
            t = by_slot.get(s)
            if t is not None:
                add(f"/nice/slot/{s}/present", 1)
                add(f"/nice/slot/{s}/id", t.id)
                add(f"/nice/slot/{s}/position", t.pos[0], t.pos[1], t.pos[2])
                add(f"/nice/slot/{s}/speed", t.speed)
                add(f"/nice/slot/{s}/conf", t.conf)
            else:
                add(f"/nice/slot/{s}/present", 0)

        for kind, slot, tid in tracker.events:
            add(f"/nice/event/{kind}", slot, tid)

        client.send(bundle.build())

        if args.verbose:
            slots = " ".join(f"{t.slot}:id{t.id}" for t in tracks)
            print(f"\rfid {fid}  n={count}  energy={energy:.2f} "
                  f"spread={spread:.2f}  [{slots}]        ",
                  end="", flush=True)
            for kind, slot, tid in tracker.events:
                print(f"\n{kind}: slot {slot} id {tid}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")
