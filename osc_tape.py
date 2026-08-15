# SPDX-FileCopyrightText: 2026 Jiří Vyskočil <jiri@vyskocil.com>
# SPDX-License-Identifier: MIT
"""Records the OSC stream to a file and replays it later, off-site.

Composing against this installation should not require the camera, the
servers, or the room. Record once while people are actually moving in the
space, hand the file to whoever is writing the music, and their Sonic Pi
at home receives the same stream -- same addresses, same values, same
timing -- with nothing else running.

Deliberately standard-library only, because the recording travels to
machines that have no venv, no python-osc, and no reason to acquire them:

  python osc_tape.py play gallery.osctape --port 4560     # Sonic Pi
  python osc_tape.py play gallery.osctape --loop          # rehearse
  python osc_tape.py record gallery.osctape --port 9001   # capture

Recordings are made and replayed whole -- everything the room produced,
so nothing is lost to a decision made on the night. --only narrows a
replay for a receiver that wants less, which is a debugging convenience
rather than the normal way to listen.

Recording live is usually simpler from the other end -- osc_bridge.py
--record writes this same format straight from the sender, needing no
second process and losing nothing to the network. Use `record` here to
capture a stream you can only reach over the wire.

Format: b'NSOSC1\\n', then <f64 seconds since first packet, u32 length,
payload> per packet. Whole UDP payloads, byte for byte, so a replay is
indistinguishable from the live bridge: bundles stay bundles, and no
decoder sits between the recording and the receiver to reinterpret it.
"""

import argparse
import fnmatch
import socket
import struct
import time
from collections.abc import Callable
from typing import BinaryIO

MAGIC = b"NSOSC1\n"
HEADER = struct.Struct("<dI")   # seconds since first packet, payload length

BUNDLE = b"#bundle\x00"         # an OSC bundle's first 8 bytes
ELEMENT = struct.Struct(">i")   # each bundle element is preceded by its size

# One packet, timestamped: (seconds since the recording's first packet, bytes).
Packet = tuple[float, bytes]


def main() -> None:
    args = parse_args()
    if args.mode == "record":
        record(args.file, args.bind, args.port)
    else:
        play(args.file, args.host, args.port, args.speed, args.loop,
             args.only)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_subparsers(dest="mode", required=True)

    rec = modes.add_parser("record", help="listen on a port, save what arrives")
    rec.add_argument("file")
    rec.add_argument("--port", type=int, default=9001)
    rec.add_argument("--bind", default="0.0.0.0")

    playback = modes.add_parser("play", help="send a saved recording")
    playback.add_argument("file")
    playback.add_argument("--host", default="127.0.0.1",
                          help="receiver address (Sonic Pi at home: leave it)")
    playback.add_argument("--port", type=int, default=4560,
                          help="receiver port (Sonic Pi: 4560)")
    playback.add_argument("--speed", type=float, default=1.0,
                          help="playback rate; 0.5 is half speed")
    playback.add_argument("--loop", action="store_true",
                          help="repeat forever, for rehearsing")
    playback.add_argument("--only", default=None, metavar="GLOB[,GLOB...]",
                          help="send only these addresses, e.g. "
                               "'/nice/group/*'; default sends everything")
    return parser.parse_args()


def record(path: str, bind: str, port: int) -> None:
    """Save every datagram arriving on a port until interrupted."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((bind, port))
    print(f"recording {bind}:{port} -> {path}   (Ctrl+C to stop)")

    start: float | None = None
    count = 0
    with open(path, "wb") as tape:
        tape.write(MAGIC)
        try:
            while True:
                payload, _sender = sock.recvfrom(65535)
                now = time.monotonic()
                if start is None:
                    start = now         # the clock starts at the first packet,
                count += 1              # so waiting to press play costs nothing
                write_packet(tape, now - start, payload)
                print(f"\r{count} packets, {now - start:6.1f}s ",
                      end="", flush=True)
        except KeyboardInterrupt:
            pass
    print(f"\nsaved {count} packets to {path}")


def play(path: str, host: str, port: int, speed: float, loop: bool,
         only: str | None = None) -> None:
    """Send a recording at its original pace, optionally narrowed and repeated."""
    packets = load(path)
    if only is not None:
        packets = narrow(packets, address_filter(only))
    if not packets:
        raise SystemExit(f"{path} holds no packets"
                         + (f" matching {only}" if only else ""))
    duration = packets[-1][0] / speed

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Costs nothing for unicast, and lets --host be a subnet broadcast
    # address, exactly as the live bridge allows.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    print(f"{len(packets)} packets, {duration:.1f}s -> {host}:{port}"
          f" at {speed}x" + (", looping" if loop else "")
          + "   (Ctrl+C to stop)")

    try:
        while True:
            start = time.monotonic()
            for index, (offset, payload) in enumerate(packets):
                # Sleep against the recording's own clock rather than
                # accumulating per-packet drift.
                behind = offset / speed - (time.monotonic() - start)
                if behind > 0:
                    time.sleep(behind)
                sock.sendto(payload, (host, port))
                if index % 20 == 0:
                    print(f"\r{offset / speed:6.1f}/{duration:.1f}s ",
                          end="", flush=True)
            if not loop:
                break
    except KeyboardInterrupt:
        pass
    print("\nstopped")


def address_filter(patterns: str | None) -> Callable[[str], bool]:
    """Build the predicate deciding which addresses reach the wire.

    `patterns` is a comma-separated list of globs matched against the whole
    address -- '*' spans '/', so both '/nice/group/*' and '/nice/*energy'
    work -- and None keeps everything. osc_bridge.py shares this so a live
    stream and a replayed one narrow by exactly the same rule.
    """
    if not patterns:
        return lambda _address: True
    globs = [p.strip() for p in patterns.split(",") if p.strip()]
    return lambda address: any(fnmatch.fnmatchcase(address, g) for g in globs)


def narrow(packets: list[Packet], keep: Callable[[str], bool]) -> list[Packet]:
    """Drop unwanted addresses, and any packet left with nothing in it."""
    kept = [(offset, trimmed)
            for offset, payload in packets
            if (trimmed := trim(payload, keep)) is not None]
    return kept


def trim(payload: bytes, keep: Callable[[str], bool]) -> bytes | None:
    """Rebuild one packet with only the wanted addresses, or None if empty.

    Bundle elements are copied byte for byte rather than decoded and
    re-encoded: the addresses are all this needs to read, and arguments
    that never pass through a parser cannot be altered by one. Anything
    unrecognised (a nested bundle, a shape this bridge never sends) is
    kept whole rather than guessed at.
    """
    if not payload.startswith(BUNDLE):
        return payload if keep(address_of(payload)) else None

    out = [payload[:16]]                        # '#bundle\0' and the timetag
    at, empty = 16, True
    while at + ELEMENT.size <= len(payload):
        size = ELEMENT.unpack_from(payload, at)[0]
        at += ELEMENT.size
        element = payload[at:at + size]
        at += size
        if len(element) < size:
            break                               # truncated tail: stop here
        if element.startswith(BUNDLE) or keep(address_of(element)):
            out.append(ELEMENT.pack(size))
            out.append(element)
            empty = False
    return None if empty else b"".join(out)


def address_of(message: bytes) -> str:
    """The address of one OSC message: its leading null-terminated string."""
    end = message.find(b"\x00")
    return message[:end if end >= 0 else len(message)].decode("ascii", "replace")


def load(path: str) -> list[Packet]:
    """Read a whole recording into memory (minutes of frames are megabytes).

    A short read at the end means the recorder was killed mid-write; keep
    what is intact rather than refusing the file.
    """
    packets: list[Packet] = []
    with open(path, "rb") as tape:
        if tape.read(len(MAGIC)) != MAGIC:
            raise SystemExit(f"{path} is not an OSC recording")
        while True:
            head = tape.read(HEADER.size)
            if len(head) < HEADER.size:
                return packets
            offset, length = HEADER.unpack(head)
            payload = tape.read(length)
            if len(payload) < length:
                return packets
            packets.append((offset, payload))


def write_packet(tape: BinaryIO, offset: float, payload: bytes) -> None:
    """Append one timestamped packet, flushed.

    Flushing per packet because recordings end by Ctrl+C or by a killed
    terminal, and a buffered tail lost to either would be a session nobody
    can repeat -- the people who made it have gone home.
    """
    tape.write(HEADER.pack(offset, len(payload)))
    tape.write(payload)
    tape.flush()


if __name__ == "__main__":
    main()
