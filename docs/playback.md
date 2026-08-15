# Playing a recording (Windows)

A recording is a few minutes of an actual room: people walking, arriving,
standing near each other, in front of a depth camera that tracked them.
Playing it back sends Sonic Pi exactly what the installation sends live —
same messages, same values, same timing — with no camera, no server and
no network involved. Everything below happens on your own machine.

You should have received two things: `osc_tape.py`, and one or more
`.osctape` recordings.

## 1. Install Python (once)

In PowerShell:

```bash
winget install Python.Python.3.13
```

Or download the installer from [python.org](https://www.python.org/downloads/windows/)
and **tick "Add python.exe to PATH"** on the first screen — the one box
people miss, and skipping it is what makes every later command fail.

Close and reopen PowerShell, then check:

```bash
python --version
```

You want a version number. If instead the Microsoft Store opens, Windows'
placeholder is shadowing the real thing: use `py` in place of `python`
everywhere below, and it will work.

Nothing else needs installing. No packages, no virtual environment.

## 2. Open a terminal where the files are

Put `osc_tape.py` and the recordings in one folder. In Explorer,
shift-right-click the folder background and choose **Open PowerShell
window here**. Or open PowerShell and `cd` to it:

```bash
cd C:\Users\you\Documents\nice-stream
```

## 3. Start Sonic Pi, then play

Sonic Pi listens on port 4560 and accepts local messages out of the box —
nothing to configure. Start it first, then:

```bash
python osc_tape.py play gallery-full-2026-08-15.osctape
```

It prints how many packets and how long, then counts through. `Ctrl+C`
stops it. That is the whole stream — everything the room produced, all 74
addresses a frame — which is what you want to compose against.

## 4. One habit in Sonic Pi

`sleep` after each `sync` in any loop following a continuous signal:

```ruby
live_loop :energy do
  use_real_time
  energy = sync "/osc*/nice/group/energy"
  synth :dsaw, note: :e2, sustain: 0.3, release: 0.2,
        cutoff: 50 + [energy[0] * 40, 80].min
  sleep 0.25
end
```

`sync` waits for the *next* matching message, so the `sleep` sets your
musical rate rather than the camera's — ten frames a second is more often
than most music wants to change. Frames arriving during the sleep are
skipped, not queued. Loops waiting on rare events (`entered`, `left`) need
no sleep; they should catch every one.

Cues arrive prefixed with the sender, so match on `/osc*/nice/...` rather
than a literal address — that pattern works for both a replay and the live
installation.

## What is in the stream

Positions are metres as seen from the camera: x sideways, y up, z away
from the lens.

| Address | Arguments | Meaning |
| --- | --- | --- |
| `/nice/group/count` | int | people currently tracked |
| `/nice/group/centroid` | x y z | where the group is, as one point |
| `/nice/group/spread` | 0..1 | how scattered they are |
| `/nice/group/energy` | m/s | summed movement of everyone |
| `/nice/slot/<i>/present` | int 0/1 | slot `<i>` occupied |
| `/nice/slot/<i>/id` | int | which person, while they stay |
| `/nice/slot/<i>/position` | x y z | that person, in metres |
| `/nice/slot/<i>/speed` | m/s | how fast they are moving |
| `/nice/slot/<i>/conf` | 0..1 | how sure the camera is |
| `/nice/pair/<i>-<j>/distance` | m | how far apart two people are |
| `/nice/event/entered` | int slot, int id | someone arrived |
| `/nice/event/left` | int slot, int id | someone left |

**Slots.** Eight of them, `0`–`7`, handed out as people arrive. A slot
belongs to one person for their whole visit, so state you attach to slot 3
stays with the same body. Brief disappearances — someone walks behind
someone else — are bridged silently; no `left`/`entered` fires for those.

**Pairs.** One address per pair of slots, up to 28 of them, measured along
the floor so height differences and raised arms do not register. A pair is
sent only while both of its people are present; when one leaves, that
address simply stops arriving, and `/nice/slot/<i>/present` tells you which
slots are live.

## The recordings

Captured 15 August 2026, all at about 10 frames a second.

| File | Length | Most people at once |
| --- | --- | --- |
| `gallery-full-2026-08-15.osctape` | 2:00 | 8 |
| `gallery-full-2026-08-15-2.osctape` | 3:04 | 8 |
| `gallery-full-2026-08-15-3.osctape` | 3:50 | 8 |
| `gallery-full-2026-08-15-4.osctape` | 5:06 | 8 |
| `gallery-full-2026-08-15-5.osctape` | 0:37 | 7 |
| `gallery-full-2026-08-15-6.osctape` | 1:15 | 7 |
| `gallery-full-2026-08-15-after-7.osctape` | 3:06 | 8 |
| `gallery-full-2026-08-15-after-8.osctape` | 0:56 | 8 |

The longer ones have the most coming and going — the 5-minute recording
alone has 88 arrivals and 83 departures, and every one of the 28 possible
pairs occurs in it.

## Options

| Option | Does |
| --- | --- |
| `--loop` | start over at the end, forever |
| `--speed 0.5` | slower; `--speed 4` to skim a recording quickly |
| `--port 4560` | where to send; 4560 is Sonic Pi's, and the default |
| `--host` | another machine on the network, if you ever need it |
| `--only "<globs>"` | send only some addresses, e.g. `"/nice/group/*"` |

Rehearsing against one stretch on repeat is what `--loop` is for:

```bash
python osc_tape.py play gallery-full-2026-08-15-5.osctape --loop
```

`--only` is there if you ever want a quieter stream to debug against —
`/nice/group/*` for the whole group family, `/nice/pair/0-1/distance` for
one exact address, commas between them, quotes because PowerShell wants
them. You do not need it.

## When nothing happens

| Symptom | Cause |
| --- | --- |
| `python` opens the Microsoft Store | Windows' placeholder — use `py` instead |
| `'python' is not recognized` | PATH box unticked at install; reinstall or use `py` |
| `No such file or directory` | wrong folder, or the filename is mistyped — `ls` lists it |
| Player runs, Sonic Pi silent | Sonic Pi not started, or it is not on port 4560 |
| Still silent, and `sync` never returns | match on `/osc*/nice/...`, not `/nice/...` |
| Sonic Pi's interface slows down | turn off *Log cues* and *Log synths* in Preferences |
| `not an OSC recording` | the file is not a `.osctape`, or arrived damaged |

To see whether anything is landing at all, turn the cue log on briefly and
watch the messages scroll — then turn it back off.
