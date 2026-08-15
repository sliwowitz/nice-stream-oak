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
| `/nice/pair/<i>-<j>/distance` | m | how far apart slots `<i>` and `<j>` are |
| `/nice/event/entered` | int slot, int id | fires once on appearance |
| `/nice/event/left` | int slot, int id | fires once after departure |

**Slot semantics:** 8 slots, `<i>` = 0–7. A slot sticks to its person
for their whole stay, so a client can bind state to a slot address
without it reshuffling mid-presence. `id` is unique per appearance.
Brief dropouts (occlusion, people crossing) are bridged by a grace
window (default 2 s, `--grace`) — no `left`/`entered` pair fires for
them.

**Pair semantics:** one address per slot pair, `<i>` < `<j>` in a single
`<i>-<j>` element (`/nice/pair/2-5/distance`), so a pair address is as
sticky as the two slots it joins — up to 28 pairs for 8 slots. Distance
is measured on the floor plane (x, z), ignoring height, the same measure
`spread` and the tracker use. A pair is sent only while both its slots
are present; the rest are simply absent from the bundle rather than sent
as zeroes, so read `/nice/slot/<i>/present` from the same bundle to tell
"far apart" from "not there". Pairs touching one slot take two patterns,
`/nice/pair/2-*/distance` and `/nice/pair/*-2/distance`.

**Sonic Pi:** run the bridge with `--port 4560`; cues arrive as
`/osc*/nice/...` (see `sonicpi_example.rb`). When the bridge runs on
another machine, enable *Receive remote OSC messages* in Preferences →
IO.

Sonic Pi's interface degrades to unusable a few minutes into a full
stream. It renders every arriving message into its cue log, and its
maintainers describe those logs as costing "a huge amount of CPU" at
high message rates. Count messages, not frames: a bundle is 21 addresses
with two people and 72 with eight, so even `--rate 10` delivers hundreds
a second. Lowering the rate alone will not save it.

Give Sonic Pi its own narrowed instance instead:

```
osc_bridge.py --port 4560 --rate 10 \
    --only "/nice/group/energy,/nice/slot/0/position,/nice/event/*"
```

`--only` takes comma-separated globs matched against the whole address
(`*` spans `/`). The instance above sends 2 messages a frame — 20 a
second — regardless of how many people are in the room. Other receivers
keep their own unfiltered instance; the bridge only sends, so run as many
as you need. Then also:

- `use_cue_logging false`, and clear *Log cues* and *Log synths* in
  Preferences (under the editor/logging settings; the exact wording moves
  between versions). Note that hiding the log panels is itself broken in
  4.5.1 — narrowing the stream is the fix that does not depend on the GUI.
- `sleep` after each `sync` in loops following a continuous signal, so the
  music's rate is yours rather than the camera's. `sync` waits for the
  *next* cue, so frames arriving during the sleep are skipped, not queued.
  Rare events (`entered`, `left`) can stay unthrottled.

## Recording a session, composing away from it

The camera, the servers and the room are needed to *make* the stream, not
to work with it. Record while people are actually in the space, then hand
over two files — the recording and `osc_tape.py` — and whoever writes the
music has the installation on their own machine.

Record straight from the bridge, which cannot miss a bundle:

```
osc_bridge.py --host <receiver> --port 4560 --rate 10 --record gallery.osctape
```

It records exactly what it sends, `--only` filtering included, and flushes
every packet, so an interrupted terminal costs nothing. `osc_tape.py
record gallery.osctape --port 9001` does the same from the listening end
when the sender is out of reach.

Replay needs nothing but a Python install — no venv, no python-osc, no
camera:

```
python osc_tape.py play gallery.osctape --port 4560          # Sonic Pi
python osc_tape.py play gallery.osctape --loop --speed 0.5   # rehearse
```

Recordings are made and replayed whole; `--only` narrows a replay the same
way it narrows the live bridge, for a receiver that wants less. Hand the
recipient [Playing a recording](playback.md) and they need nothing else.

Whole UDP payloads are stored byte for byte, so a replay is
indistinguishable from the live bridge: same addresses, same bundles, same
timing. Sender-dependent cue names differ (`/osc:127.0.0.1:...` rather than
the camera machine's address), which is precisely why the examples match
on `/osc*/nice/...`.

**MIDI:** the stream converts to MIDI client-side, without touching the
wire format — `oscii_bot_example.txt` is a ready recipe for
[OSCII-bot](https://www.cockos.com/oscii-bot/) (group signals as CCs on
channel 1, one channel per person slot, entered/left as notes; ranges
tunable at the top of the script). Point a dedicated bridge instance at
it (`--port 9001`); running several bridges to different targets is
fine — the bridge only sends.

The authoritative definition is the module docstring of
`osc_bridge.py`; if this page disagrees, the docstring wins.
