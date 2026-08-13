# SPDX-FileCopyrightText: 2026 Jiří Vyskočil <jiri@vyskocil.com>
# SPDX-License-Identifier: MIT
"""Derives movement signals from nice-stream pose frames and emits them as OSC.

Reads the 'nice_stream_pose' shared-memory segment (the only segment this
bridge touches), derives movement metrics, and packs them into one OSC
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

Run with the venv (python-osc required):
  .venv/Scripts/python.exe osc_bridge.py --host 127.0.0.1 --port 9000 --verbose
A broadcast --host (e.g. 192.168.1.255) reaches every listening machine on
the LAN at once; receivers need no configuration beyond an open port.
"""

import argparse
import functools
import math
import time
from multiprocessing import shared_memory
from typing import cast

from pythonosc import osc_bundle_builder, osc_message_builder, udp_client

import nsk
from nsk import KP_CONF_MIN as MIN_JOINT_CONF
from nsk import MAX_PERSONS as MAX_SLOTS

# A 3-D point in metres, camera frame (y up, z away from the camera).
Vec3 = tuple[float, float, float]


def main() -> None:
    args = parse_args()
    # SO_BROADCAST costs nothing for unicast and lets --host point at a
    # subnet broadcast address so every listener on the LAN hears the stream.
    client = udp_client.SimpleUDPClient(args.host, args.port, allow_broadcast=True)
    # pythonosc leaves the socket non-blocking; on Windows a busy send
    # buffer (e.g. pending ARP for a fresh remote host) then raises
    # WinError 10035 and kills the bridge. A blocking UDP send just waits
    # the microseconds the buffer needs.
    client._sock.setblocking(True)  # private, but pythonosc has no public knob
    shm, max_persons, num_kp, stride, K = attach_pose_segment(args.segment)
    tracker = Tracker(args)

    last_fid = 0
    last_emit = 0.0

    print(f"emitting to {args.host}:{args.port} at <= {args.rate:.0f} Hz")
    while True:
        fid, persons = nsk.read_pose_frame(cast(nsk.Buf, shm.buf),
                                           max_persons, num_kp, stride)
        now = time.time()
        if fid == last_fid or now - last_emit < 1.0 / args.rate:
            time.sleep(0.003)
            continue
        last_fid, last_emit = fid, now

        detections: list[tuple[float, Vec3]] = []
        for conf, joints in persons:
            centroid = centroid_of(joints, K)
            if centroid is not None:
                detections.append((conf, centroid))

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
        add = functools.partial(_add_message, bundle)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1",
                        help="OSC receiver address; a subnet broadcast address "
                             "(e.g. 192.168.1.255) reaches every LAN listener")
    parser.add_argument("--port", type=int, default=9000,
                        help="receiver port (Sonic Pi: 4560)")
    parser.add_argument("--rate", type=float, default=30.0, help="max emit Hz")
    parser.add_argument("--room-scale", type=float, default=5.0,
                        help="metres; normalizes spread to 0..1")
    parser.add_argument("--smooth", type=float, default=0.15,
                        help="seconds; EMA time constant for positions")
    parser.add_argument("--gate", type=float, default=0.5,
                        help="metres; base association distance gate")
    parser.add_argument("--gate-speed", type=float, default=1.5,
                        help="m/s; gate growth for tracks unseen for a while")
    parser.add_argument("--grace", type=float, default=2.0,
                        help="seconds a lost track survives before 'left'")
    parser.add_argument("--segment", default="nice_stream_pose")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def attach_pose_segment(name: str) -> tuple[shared_memory.SharedMemory,
                                            int, int, int, nsk.Intrinsics]:
    """Block until the pose segment exists and its header parses.

    A producer (pose_server in particular) may create the segment well
    before it writes the header -- an unparseable header means "not yet",
    never "give up".
    """
    while True:
        try:
            shm = shared_memory.SharedMemory(name=name)
        except FileNotFoundError:
            print(f"waiting for '{name}' ...")
            time.sleep(1.0)
            continue
        # buf is memoryview | None in typeshed; never None on an open segment
        hdr = nsk.parse_pose_header(cast(nsk.Buf, shm.buf))
        if hdr is None:
            shm.close()
            print(f"waiting for '{name}' header ...")
            time.sleep(1.0)
            continue
        K = hdr["intrinsics"]
        print(f"attached '{name}': {hdr['max_persons']} slots, "
              f"{hdr['num_kp']} kp, fx={K[0]:.1f}")
        return shm, hdr["max_persons"], hdr["num_kp"], hdr["stride"], K


def centroid_of(joints: list[nsk.Joint], K: nsk.Intrinsics) -> Vec3 | None:
    """Compute the confidence-weighted centroid, in metres, of usable joints.

    A joint is usable with depth > 0 and conf >= MIN_JOINT_CONF; returns
    None when no joint qualifies.
    """
    fx, fy, cx, cy = K
    sw = sx = sy = sz = 0.0
    for px, py, z, conf in joints:
        if z <= 0.0 or conf < MIN_JOINT_CONF:
            continue
        sw += conf
        sx += conf * (px - cx) * z / fx
        sy += conf * -(py - cy) * z / fy
        sz += conf * z
    if sw == 0.0:
        return None
    return (sx / sw, sy / sw, sz / sw)


class Track:
    """One persistent person: smoothed position, speed, identity, OSC slot."""

    def __init__(self, track_id: int, slot: int, pos: Vec3, now: float) -> None:
        self.id = track_id
        self.slot = slot
        self.pos: tuple[float, ...] = pos   # EMA-smoothed; always length 3
        self.speed = 0.0        # EMA-smoothed, m/s
        self.conf = 0.0
        self.last_seen = now

    def update(self, pos: Vec3, conf: float, now: float, tau: float) -> None:
        dt = max(1e-3, now - self.last_seen)
        # EMA blended by dt so gaps do not lag: k reaches 1 at dt >= tau.
        k = min(1.0, dt / tau)
        prev = self.pos
        self.pos = tuple(p + (q - p) * k
                         for p, q in zip(self.pos, pos, strict=True))
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

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.tracks: list[Track] = []
        self.next_id = 0
        self.events: list[tuple[str, int, int]] = []    # ('entered'|'left', slot, id)

    def step(self, detections: list[tuple[float, Vec3]],
             now: float) -> list[Track]:
        """Match detections to tracks; spawn and expire as needed.

        Each detection is (confidence, centroid). Entered/left events from
        this step replace self.events. Returns the active tracks.
        """
        self.events.clear()

        # pair (distance, track, detection) greedily on the floor plane
        pairs: list[tuple[float, Track, int]] = []
        for t in self.tracks:
            gate = self.args.gate + self.args.gate_speed * (now - t.last_seen)
            for di, (_conf, centroid) in enumerate(detections):
                d = math.hypot(centroid[0] - t.pos[0], centroid[2] - t.pos[2])
                if d <= gate:
                    pairs.append((d, t, di))
        pairs.sort(key=lambda pair: pair[0])

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for _d, t, di in pairs:
            if id(t) in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(id(t))
            matched_dets.add(di)
            conf, centroid = detections[di]
            t.update(centroid, conf, now, self.args.smooth)

        # unmatched detections -> new tracks (if a slot is free)
        for di, (conf, centroid) in enumerate(detections):
            if di in matched_dets:
                continue
            slot = self._free_slot()
            if slot is None:
                continue
            t = Track(self.next_id, slot, centroid, now)
            t.conf = conf
            self.next_id += 1
            self.tracks.append(t)
            self.events.append(("entered", t.slot, t.id))

        # expire tracks beyond the grace window
        survivors: list[Track] = []
        for t in self.tracks:
            if now - t.last_seen > self.args.grace:
                self.events.append(("left", t.slot, t.id))
            else:
                survivors.append(t)
        self.tracks = survivors
        return self.tracks

    def _free_slot(self) -> int | None:
        used = {t.slot for t in self.tracks}
        for s in range(MAX_SLOTS):
            if s not in used:
                return s
        return None


def _add_message(bundle: osc_bundle_builder.OscBundleBuilder, address: str,
                 *values: float) -> None:
    msg = osc_message_builder.OscMessageBuilder(address=address)
    for v in values:
        msg.add_arg(v)
    bundle.add_content(msg.build())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")
