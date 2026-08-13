# SPDX-FileCopyrightText: 2026 Jiří Vyskočil <jiri@vyskocil.com>
# SPDX-License-Identifier: MIT
"""Debug viewer for all three nice-stream segments. Esc quits.

Shows RGB with skeleton overlay, and depth as a colormap. Reads the same
shared memory Unity will read, through the same nsk.py wire-contract code the
servers write with -- if it looks right here, the server is fine and any
remaining problem is on the Unity side.
"""
import sys
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nsk  # noqa: E402

COCO_EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
              (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
              (0, 5), (0, 6)]

depth_shm = shared_memory.SharedMemory(name="nice_stream_depth")
rgb_shm   = shared_memory.SharedMemory(name="nice_stream_rgb")
pose_shm  = shared_memory.SharedMemory(name="nice_stream_pose")

frame_hdr = nsk.parse_frame_header(depth_shm.buf)
pose_hdr  = nsk.parse_pose_header(pose_shm.buf)
assert frame_hdr is not None and pose_hdr is not None, "segments not ready"
print(f"frames {frame_hdr['w']}x{frame_hdr['h']} nbuf={frame_hdr['nbuf']}; "
      f"pose max_persons={pose_hdr['max_persons']} kp={pose_hdr['num_kp']}")

last_d = last_r = 0
while True:
    rgb, fid = nsk.read_newest_frame(rgb_shm.buf, frame_hdr, np.uint8, 3,
                                     skip_id=last_r)
    if rgb is not None:
        last_r = fid
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        _, persons = nsk.read_pose_frame(
            pose_shm.buf, pose_hdr["max_persons"], pose_hdr["num_kp"],
            pose_hdr["stride"])
        for _conf, joints in persons:
            pts = [(int(x), int(y), z, c) for x, y, z, c in joints]
            for x, y, z, c in pts:
                if c > nsk.KP_CONF_MIN:
                    col = (0, 255, 0) if z > 0 else (0, 165, 255)
                    cv2.circle(bgr, (x, y), 4, col, -1)
                    if z > 0:
                        cv2.putText(bgr, f"{z:.2f}", (x + 5, y),
                                    cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 0))
            for a, b in COCO_EDGES:
                if pts[a][3] > nsk.KP_CONF_MIN and pts[b][3] > nsk.KP_CONF_MIN:
                    cv2.line(bgr, pts[a][:2], pts[b][:2], (0, 200, 255), 2)
        cv2.putText(bgr, f"rgb {fid}  persons {len(persons)}", (10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        cv2.imshow("rgb + pose", bgr)

    depth, fid = nsk.read_newest_frame(depth_shm.buf, frame_hdr, np.uint16, 1,
                                       skip_id=last_d)
    if depth is not None:
        last_d = fid
        cv2.imshow("depth", cv2.applyColorMap(
            cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET))

    if cv2.waitKey(1) == 27:
        break
