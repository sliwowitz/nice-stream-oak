# SPDX-FileCopyrightText: 2026 Jiří Vyskočil <jiri@vyskocil.com>
# SPDX-License-Identifier: MIT
"""Debug viewer for all three nice-stream segments. Esc quits.

Shows RGB with skeleton overlay, and depth as a colormap. Reads the same
shared memory Unity will read -- if it looks right here, the server is fine
and any remaining problem is on the Unity side.
"""
import struct
from multiprocessing import shared_memory

import cv2
import numpy as np

HDR = 64
COCO_EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
              (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
              (0, 5), (0, 6)]

depth_shm = shared_memory.SharedMemory(name="nice_stream_depth")
rgb_shm   = shared_memory.SharedMemory(name="nice_stream_rgb")
pose_shm  = shared_memory.SharedMemory(name="nice_stream_pose")

_, _, W, H, _, NBUF = struct.unpack_from("<6I", depth_shm.buf, 0)
magic_p, _, MAXP, NKP, FPKP, STRIDE = struct.unpack_from("<6I", pose_shm.buf, 0)
assert magic_p == 0x504B534E, "bad pose magic"
print(f"frames {W}x{H} nbuf={NBUF}; pose max_persons={MAXP} kp={NKP}")

last_d = last_r = -1
while True:
    fid, = struct.unpack_from("<Q", rgb_shm.buf, 24)
    if fid != last_r and fid > 0:
        last_r = fid
        off = HDR + (fid % NBUF) * W * H * 3
        rgb = np.frombuffer(rgb_shm.buf, np.uint8, W * H * 3, off) \
                .reshape(H, W, 3).copy()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        n_persons, = struct.unpack_from("<I", pose_shm.buf, 40)
        for p in range(n_persons):
            base = HDR + p * STRIDE
            conf, = struct.unpack_from("<f", pose_shm.buf, base)
            if conf <= 0:
                continue
            pts = []
            for i in range(NKP):
                x, y, z, c = struct.unpack_from("<4f", pose_shm.buf,
                                                base + 16 + i * 16)
                pts.append((int(x), int(y), z, c))
                if c > 0.3:
                    col = (0, 255, 0) if z > 0 else (0, 165, 255)
                    cv2.circle(bgr, (int(x), int(y)), 4, col, -1)
                    if z > 0:
                        cv2.putText(bgr, f"{z:.2f}", (int(x) + 5, int(y)),
                                    cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 0))
            for a, b in COCO_EDGES:
                if pts[a][3] > 0.3 and pts[b][3] > 0.3:
                    cv2.line(bgr, pts[a][:2], pts[b][:2], (0, 200, 255), 2)
        cv2.putText(bgr, f"rgb {fid}  persons {n_persons}", (10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        cv2.imshow("rgb + pose", bgr)

    fid, = struct.unpack_from("<Q", depth_shm.buf, 24)
    if fid != last_d and fid > 0:
        last_d = fid
        off = HDR + (fid % NBUF) * W * H * 2
        d = np.frombuffer(depth_shm.buf, np.uint16, W * H, off).reshape(H, W)
        cv2.imshow("depth", cv2.applyColorMap(
            cv2.convertScaleAbs(d, alpha=0.03), cv2.COLORMAP_JET))

    if cv2.waitKey(1) == 27:
        break
