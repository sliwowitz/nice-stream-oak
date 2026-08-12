# SPDX-FileCopyrightText: 2026 Jiří Vyskočil <jiri@vyskocil.com>
# SPDX-License-Identifier: MIT
import depthai as dai
print("depthai", dai.__version__)
print(dai.Device.getAllAvailableDevices())