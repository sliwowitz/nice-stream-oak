# SPDX-FileCopyrightText: 2026 Jiří Vyskočil <jiri@vyskocil.com>
# SPDX-License-Identifier: MIT
"""Recording-format tests: a session nobody can repeat must survive reading.

The people who made a recording have gone home, so the reader forgives a
tail lost to a killed terminal and keeps whatever is intact."""

import sys
from pathlib import Path

import pytest
from pythonosc import osc_bundle_builder, osc_message_builder, osc_packet

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import osc_tape

PACKETS = [(0.0, b"first bundle"), (0.1, b"second"), (0.25, b"third one")]

# A frame shaped like the bridge's: mixed argument counts and types.
FRAME = {"/nice/group/count": (3,),
         "/nice/group/energy": (1.25,),
         "/nice/slot/0/position": (0.5, 1.0, 2.5),
         "/nice/pair/0-1/distance": (1.75,)}


def bundle(frame: dict[str, tuple] = FRAME) -> bytes:
    builder = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
    for address, args in frame.items():
        message = osc_message_builder.OscMessageBuilder(address=address)
        for arg in args:
            message.add_arg(arg)
        builder.add_content(message.build())
    return builder.build().dgram


def decode(payload: bytes) -> dict[str, tuple]:
    return {m.message.address: tuple(m.message.params)
            for m in osc_packet.OscPacket(payload).messages}


def write_tape(path: Path, packets: list[osc_tape.Packet]) -> Path:
    with open(path, "wb") as tape:
        tape.write(osc_tape.MAGIC)
        for offset, payload in packets:
            osc_tape.write_packet(tape, offset, payload)
    return path


def test_a_recording_reads_back_byte_for_byte(tmp_path):
    tape = write_tape(tmp_path / "session.osctape", PACKETS)
    assert osc_tape.load(str(tape)) == PACKETS


def test_a_tail_lost_mid_write_keeps_the_rest(tmp_path):
    """A killed recorder leaves a half-written packet; the session still plays."""
    tape = write_tape(tmp_path / "killed.osctape", PACKETS)
    whole = tape.read_bytes()
    tape.write_bytes(whole[:-4])
    assert osc_tape.load(str(tape)) == PACKETS[:-1]


def test_a_truncated_length_word_keeps_the_rest(tmp_path):
    tape = write_tape(tmp_path / "shorn.osctape", PACKETS)
    whole = tape.read_bytes()
    tape.write_bytes(whole + osc_tape.HEADER.pack(0.3, 99)[:6])
    assert osc_tape.load(str(tape)) == PACKETS


def test_another_file_is_refused_rather_than_replayed(tmp_path):
    other = tmp_path / "notes.txt"
    other.write_bytes(b"these are not packets")
    with pytest.raises(SystemExit):
        osc_tape.load(str(other))


def test_an_empty_recording_is_empty_not_broken(tmp_path):
    tape = write_tape(tmp_path / "nobody-came.osctape", [])
    assert osc_tape.load(str(tape)) == []


# --------------------------------------------------------- address filter ---
def test_no_filter_keeps_every_address():
    keep = osc_tape.address_filter(None)
    assert keep("/nice/group/energy") and keep("/nice/slot/7/conf")


def test_globs_span_slashes_and_select_whole_families():
    keep = osc_tape.address_filter("/nice/group/*,/nice/pair/*")
    assert keep("/nice/group/energy")
    assert keep("/nice/pair/0-1/distance")
    assert not keep("/nice/slot/0/position")
    assert not keep("/nice/event/entered")


def test_a_single_address_narrows_to_exactly_it():
    keep = osc_tape.address_filter(" /nice/slot/0/position ")
    assert keep("/nice/slot/0/position")
    assert not keep("/nice/slot/1/position")


# --------------------------------------------------------------- trimming ---
def test_trimming_keeps_the_wanted_addresses_and_their_arguments():
    """Arguments must survive untouched: they are copied, never re-encoded."""
    keep = osc_tape.address_filter("/nice/group/*,/nice/pair/*")
    trimmed = osc_tape.trim(bundle(), keep)
    assert decode(trimmed) == {"/nice/group/count": (3,),
                               "/nice/group/energy": (1.25,),
                               "/nice/pair/0-1/distance": (1.75,)}


def test_trimming_a_multi_argument_address_keeps_every_argument():
    keep = osc_tape.address_filter("/nice/slot/0/position")
    assert decode(osc_tape.trim(bundle(), keep)) == {
        "/nice/slot/0/position": (0.5, 1.0, 2.5)}


def test_a_bundle_nothing_matches_is_dropped_rather_than_sent_empty():
    keep = osc_tape.address_filter("/nice/nothing/*")
    assert osc_tape.trim(bundle(), keep) is None


def test_no_filter_leaves_the_bundle_exactly_as_recorded():
    whole = bundle()
    assert osc_tape.trim(whole, osc_tape.address_filter(None)) == whole


def test_a_bare_message_is_judged_by_its_own_address():
    message = osc_message_builder.OscMessageBuilder(address="/nice/group/count")
    message.add_arg(3)
    payload = message.build().dgram
    assert osc_tape.trim(payload, osc_tape.address_filter("/nice/group/*"))
    assert osc_tape.trim(payload, osc_tape.address_filter("/nice/slot/*")) is None


def test_narrowing_drops_packets_left_with_nothing():
    packets = [(0.0, bundle()), (0.1, bundle({"/nice/slot/1/conf": (0.9,)}))]
    narrowed = osc_tape.narrow(packets,
                               osc_tape.address_filter("/nice/group/*"))
    assert [offset for offset, _ in narrowed] == [0.0]
