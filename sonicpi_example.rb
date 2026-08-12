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

# Drone whose brightness follows the group's collective energy
live_loop :energy_drone do
  use_real_time
  energy = sync "/osc*/nice/group/energy"
  # energy[0] is the float; ~0 still .. ~3 lively
  synth :dsaw, note: :e2, sustain: 0.3, release: 0.2,
        cutoff: 50 + [energy[0] * 40, 80].min, amp: 0.6
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
end
