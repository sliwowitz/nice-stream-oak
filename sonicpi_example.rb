# SPDX-FileCopyrightText: 2026 Jiří Vyskočil <jiri@vyskocil.com>
# SPDX-License-Identifier: MIT
# nice-stream OSC -> sound, minimal wiring proof.
#
# 1. Sonic Pi 4 listens for OSC on port 4560 (Prefs > IO > "Receive remote
#    OSC messages" must be ON for messages from another machine; local is on
#    by default).
# 2. Run the bridge against it:
#      .venv/Scripts/python.exe osc_bridge.py --host <this machine> --port 4560
# 3. Incoming addresses appear as cues named /osc:<sender-ip>:<port>/nice/...
#    The pattern /osc*/nice/... matches regardless of sender.
#
# Surviving the data rate. Sonic Pi renders every arriving message into
# its cue log, and that rendering is what grinds the GUI to a halt minutes
# into a stream. What matters is messages per second, not frames: a full
# bundle is 21 addresses with two people and 72 with eight, so --rate alone
# never gets the count low enough. Send this instance only what the piece
# below actually plays -- 3 messages a frame, whoever walks in:
#
#   osc_bridge.py --port 4560 --rate 10 --only
#     "/nice/group/energy,/nice/slot/0/position,/nice/pair/0-1/distance,/nice/event/*"
#
# Then two habits, both used below:
#   * use_cue_logging false, and turn "Log cues" and "Log synths" off in
#     Preferences.
#   * sleep after each sync in the continuous loops. sync waits for the
#     *next* matching cue, so a sleep sets the musical rate and the frames
#     arriving meanwhile are simply skipped -- no queue, no backlog. Only
#     the rare entered/left loops should react to every cue.
use_cue_logging false

# Drone whose brightness follows the group's collective energy
live_loop :energy_drone do
  use_real_time
  energy = sync "/osc*/nice/group/energy"
  # energy[0] is the float; ~0 still .. ~3 lively
  synth :dsaw, note: :e2, sustain: 0.3, release: 0.2,
        cutoff: 50 + [energy[0] * 40, 80].min, amp: 0.6
  sleep 0.25
end

# A soft chime whenever somebody enters; a low thud when they leave
live_loop :arrivals do
  use_real_time
  slot, id = sync "/osc*/nice/event/entered"
  play :e5, amp: 0.8, release: 1.5, pan: (slot - 4) / 4.0
end

live_loop :departures do
  use_real_time
  slot, id = sync "/osc*/nice/event/left"
  play :e2, amp: 0.6, release: 2.0
end

# Per-person pitch from position: person in slot 0, x position -> pan,
# distance from camera -> pitch. Copy this loop for more slots.
live_loop :person0 do
  use_real_time
  x, y, z = sync "/osc*/nice/slot/0/position"
  play scale(:e3, :minor_pentatonic)[(z * 2).to_i % 5],
       amp: 0.4, release: 0.4, pan: [[x / 2.5, -1].max, 1].min
  sleep 0.5
end

# Two people, one interval: the closer slots 0 and 1 stand, the tighter the
# harmony. Only arrives while both are present. One loop per pair of interest.
live_loop :pair01 do
  use_real_time
  d = sync("/osc*/nice/pair/0-1/distance")[0]
  play :e4, amp: 0.3, release: 0.3
  play :e4 + [d * 3, 12].min, amp: 0.3, release: 0.3
  sleep 0.5
end
