# OSC output

What a client must implement to consume the bridge's stream.

**Transport:** plain UDP push — the bridge sends to the client, the
client only listens (no handshake, no subscription). Start the bridge
with `--host <client-ip> --port <port>`. Pointing `--host` at the
subnet broadcast address (e.g. `192.168.1.255`) reaches every machine
on the LAN that has the port open — no per-client IP needed, any
number of simultaneous listeners. One OSC bundle per frame,
rate-capped (default 30 Hz, `--rate`). Coordinates are metres in the
camera frame: y up, z away from the camera.

**Messages** (arguments are floats unless noted):

| Address | Arguments | Meaning |
| --- | --- | --- |
| `/nice/group/count` | int | people currently tracked |
| `/nice/group/centroid` | x y z | group centroid |
| `/nice/group/spread` | 0..1 | dispersion around centroid / room scale |
| `/nice/group/energy` | m/s | summed smoothed per-person speed |
| `/nice/slot/<i>/present` | int 0/1 | slot `<i>` occupied |
| `/nice/slot/<i>/id` | int | persistent person id (while present) |
| `/nice/slot/<i>/position` | x y z | person centroid, metres |
| `/nice/slot/<i>/speed` | m/s | smoothed centroid speed |
| `/nice/slot/<i>/conf` | 0..1 | detection confidence |
| `/nice/event/entered` | int slot, int id | fires once on appearance |
| `/nice/event/left` | int slot, int id | fires once after departure |

**Slot semantics:** 8 slots, `<i>` = 0–7. A slot sticks to its person
for their whole stay, so a client can bind state to a slot address
without it reshuffling mid-presence. `id` is unique per appearance.
Brief dropouts (occlusion, people crossing) are bridged by a grace
window (default 2 s, `--grace`) — no `left`/`entered` pair fires for
them.

**Sonic Pi:** run the bridge with `--port 4560`; cues arrive as
`/osc*/nice/...` (see `sonicpi_example.rb`). When the bridge runs on
another machine, enable *Receive remote OSC messages* in Preferences →
IO.

The authoritative definition is the module docstring of
`osc_bridge.py`; if this page disagrees, the docstring wins.
