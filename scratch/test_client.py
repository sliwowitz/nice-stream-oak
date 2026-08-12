# SPDX-FileCopyrightText: 2026 Jiří Vyskočil <jiri@vyskocil.com>
# SPDX-License-Identifier: MIT
import struct, time
from multiprocessing import shared_memory
import numpy as np, cv2

shm = shared_memory.SharedMemory(name="nice_stream_depth")
magic, ver, W, H, BPP, NBUF = struct.unpack_from("<6I", shm.buf, 0)
assert magic == 0x314B534E
last = -1
while True:
    fid, = struct.unpack_from("<Q", shm.buf, 24)
    if fid != last:
        off = 64 + (fid % NBUF) * W * H * BPP
        d = np.frombuffer(shm.buf, np.uint16, W*H, off).reshape(H, W)
        cv2.imshow("depth", cv2.applyColorMap(
            cv2.convertScaleAbs(d, alpha=0.03), cv2.COLORMAP_JET))
        last = fid
    if cv2.waitKey(1) == 27: break