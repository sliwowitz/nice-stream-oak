# SPDX-FileCopyrightText: 2026 Jiří Vyskočil <jiri@vyskocil.com>
# SPDX-License-Identifier: MIT
"""Capture INT8-calibration frames from the running rgb SHM stream.

Quantization is only as good as its calibration set: the zoo's RVC4 pose
models were calibrated on 40 generic COCO images, which is part of why
they underperform. This captures evenly spaced frames from the actual
camera under the actual venue lighting into a flat zip that HubAI accepts
as `--quantization-data` (see docs/yolo11-conversion.md).

Run stream_server first, have people move around in front of the camera,
then:

  .venv/Scripts/python.exe calib_capture.py --count 300 --interval 0.5
"""

import argparse
import time
import zipfile
from multiprocessing import shared_memory
from typing import cast

import cv2
import numpy as np

import nsk


def main() -> None:
    args = parse_args()

    print(f"waiting for '{args.segment}' (is stream_server running?) ...")
    while True:
        try:
            shm = shared_memory.SharedMemory(name=args.segment)
        except FileNotFoundError:
            time.sleep(1.0)
            continue
        hdr = nsk.parse_frame_header(cast(nsk.Buf, shm.buf))
        if hdr is not None:
            break
        time.sleep(1.0)
    print(f"attached: {hdr['w']}x{hdr['h']}, capturing {args.count} frames "
          f"every {args.interval:g}s -> {args.out}")

    last_id = 0
    saved = 0
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as zf:
        while saved < args.count:
            frame, fid = nsk.read_newest_frame(cast(nsk.Buf, shm.buf), hdr,
                                               np.uint8, 3, skip_id=last_id)
            if frame is None:
                time.sleep(0.05)
                continue
            last_id = fid
            # Segment holds RGB; jpeg encoders expect BGR.
            ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                continue
            zf.writestr(f"calib_{saved:04d}.jpg", jpg.tobytes())
            saved += 1
            if saved % 25 == 0:
                print(f"  {saved}/{args.count}")
            time.sleep(args.interval)

    shm.close()
    print(f"done: {args.out} ({saved} frames). Feed it to hubai convert "
          "--quantization-data (docs/yolo11-conversion.md).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=300,
                        help="frames to capture")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="seconds between frames (spread over varied poses)")
    parser.add_argument("--out", default="calib_frames.zip")
    parser.add_argument("--segment", default="nice_stream_rgb")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\naborted")
