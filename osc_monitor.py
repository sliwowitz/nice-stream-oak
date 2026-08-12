# SPDX-FileCopyrightText: 2026 Jiří Vyskočil <jiri@vyskocil.com>
# SPDX-License-Identifier: MIT
"""Rudimentary OSC monitor: shows the latest value per address, ~4x/second.

Debugs the bridge without Sonic Pi:
  .venv/Scripts/python.exe osc_monitor.py            # listens on 0.0.0.0:9000
  .venv/Scripts/python.exe osc_bridge.py --verbose   # in another terminal
"""

import argparse
import threading
import time

from pythonosc import dispatcher, osc_server


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--bind", default="0.0.0.0")
    args = p.parse_args()

    latest = {}
    counts = {"n": 0}

    def handler(addr, *values):
        latest[addr] = values
        counts["n"] += 1

    d = dispatcher.Dispatcher()
    d.set_default_handler(handler)
    srv = osc_server.ThreadingOSCUDPServer((args.bind, args.port), d)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"listening on {args.bind}:{args.port}  (Ctrl+C quits)")

    try:
        while True:
            time.sleep(0.25)
            lines = [f"-- {counts['n']} messages --"]
            for addr in sorted(latest):
                vals = ", ".join(f"{v:.3f}" if isinstance(v, float) else str(v)
                                 for v in latest[addr])
                lines.append(f"{addr:28s} {vals}")
            print("\x1b[2J\x1b[H" + "\n".join(lines), flush=True)
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
