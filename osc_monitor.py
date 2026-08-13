# SPDX-FileCopyrightText: 2026 Jiří Vyskočil <jiri@vyskocil.com>
# SPDX-License-Identifier: MIT
"""Shows the latest value per OSC address, redrawn about four times a second.

Debugs the bridge without Sonic Pi:
  .venv/Scripts/python.exe osc_monitor.py            # listens on 0.0.0.0:9000
  .venv/Scripts/python.exe osc_bridge.py --verbose   # in another terminal
"""

import argparse
import threading
import time
from typing import Any

from pythonosc import dispatcher, osc_server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--bind", default="0.0.0.0")
    args = parser.parse_args()

    latest: dict[str, tuple[Any, ...]] = {}
    message_count = 0

    def handler(addr: str, *values: Any) -> None:
        nonlocal message_count
        latest[addr] = values
        message_count += 1

    dispatch = dispatcher.Dispatcher()
    dispatch.set_default_handler(handler)
    server = osc_server.ThreadingOSCUDPServer((args.bind, args.port), dispatch)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"listening on {args.bind}:{args.port}  (Ctrl+C quits)")

    try:
        while True:
            time.sleep(0.25)
            lines = [f"-- {message_count} messages --"]
            for addr in sorted(latest):
                vals = ", ".join(f"{v:.3f}" if isinstance(v, float) else str(v)
                                 for v in latest[addr])
                lines.append(f"{addr:28s} {vals}")
            # ANSI: clear screen, home the cursor, repaint in one write.
            print("\x1b[2J\x1b[H" + "\n".join(lines), flush=True)
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
