import pytest

from censor.align import Interval
from censor.subs import Cue
from censor.timeline import TimeMap


def approx(value, expected, tol=1e-3):
    return abs(value - expected) < tol


def test_no_cuts_is_identity():
    tm = TimeMap([], duration=100.0)
    assert not tm
    assert tm.map(42.0) == 42.0
    assert tm.output_duration == 100.0


def test_time_after_a_cut_shifts_earlier():
    tm = TimeMap([(10.0, 20.0)], duration=100.0)
    assert tm.map(5.0) == 5.0
    assert tm.map(15.0) is None
    assert approx(tm.map(25.0), 15.0)
    assert approx(tm.output_duration, 90.0)


def test_two_cuts_accumulate():
    tm = TimeMap([(10.0, 20.0), (50.0, 55.0)], duration=100.0)
    assert approx(tm.map(60.0), 45.0)
    assert approx(tm.total_removed, 15.0)


def test_adjacent_cuts_merge():
    tm = TimeMap([(10.0, 20.0), (20.0, 30.0)], duration=100.0)
    assert tm.cuts == [(10.0, 30.0)]
    assert approx(tm.map(35.0), 15.0)


def test_cut_at_start_and_end():
    tm = TimeMap([(0.0, 10.0), (90.0, 100.0)], duration=100.0)
    assert tm.map(5.0) is None
    assert approx(tm.map(10.0), 0.0)
    assert tm.map(95.0) is None
    assert approx(tm.output_duration, 80.0)


def test_unmap_is_the_inverse_of_map():
    tm = TimeMap([(10.0, 20.0), (50.0, 55.0)], duration=100.0)
    for source in (5.0, 25.0, 49.0, 60.0, 99.0):
        mapped = tm.map(source)
        assert mapped is not None
        assert approx(tm.unmap(mapped), source)


def test_cue_wholly_inside_a_cut_is_dropped():
    tm = TimeMap([(10.0, 20.0)], duration=100.0)
    assert tm.map_cues([Cue(1, 12.0, 14.0, "gone")]) == []


def test_cue_straddling_a_boundary_is_clipped():
    tm = TimeMap([(10.0, 20.0)], duration=100.0)
    out = tm.map_cues([Cue(1, 8.0, 14.0, "clipped")])
    assert len(out) == 1
    assert approx(out[0].start, 8.0) and approx(out[0].end, 10.0)


def test_cue_spanning_a_whole_cut_is_rejoined():
    tm = TimeMap([(10.0, 20.0)], duration=100.0)
    out = tm.map_cues([Cue(1, 8.0, 24.0, "spans the seam")])
    assert len(out) == 1
    assert approx(out[0].start, 8.0)
    assert approx(out[0].end, 14.0, tol=0.01)  # 2s before + 4s after, seam removed


def test_cue_after_a_cut_keeps_its_duration():
    tm = TimeMap([(10.0, 20.0)], duration=100.0)
    out = tm.map_cues([Cue(1, 30.0, 33.0, "later")])
    assert approx(out[0].start, 20.0)
    assert approx(out[0].end - out[0].start, 3.0)


def test_mute_interval_overlapping_a_cut_is_trimmed():
    tm = TimeMap([(10.0, 20.0)], duration=100.0)
    interval = Interval(start=9.0, end=21.0, level="L1", token="x", tier="t", confidence=1.0)
    mapped = tm.map_intervals([interval])
    assert len(mapped) == 1
    assert approx(mapped[0].start, 9.0)
    assert approx(mapped[0].end, 11.0)


def test_segments_lists_surviving_ranges():
    tm = TimeMap([(10.0, 20.0), (50.0, 55.0)], duration=100.0)
    segments = tm.segments()
    assert len(segments) == 3
    assert approx(segments[0][1], 10.0)
    assert approx(segments[1][0], 20.0)
    assert approx(segments[2][1], 100.0)


def test_chapters_inside_a_cut_are_removed():
    tm = TimeMap([(10.0, 40.0)], duration=100.0)
    chapters = [
        {"start_time": "0", "end_time": "10", "tags": {"title": "one"}},
        {"start_time": "12", "end_time": "38", "tags": {"title": "gone"}},
        {"start_time": "40", "end_time": "100", "tags": {"title": "three"}},
    ]
    mapped = tm.map_chapters(chapters)
    assert [c["title"] for c in mapped] == ["one", "three"]
    assert approx(mapped[1]["start_time"], 10.0)


@pytest.mark.parametrize("duration", [0.0, 100.0])
def test_map_is_monotonic(duration):
    tm = TimeMap([(10.0, 20.0), (50.0, 55.0)], duration=duration)
    previous = -1.0
    for source in [x / 2 for x in range(0, 200)]:
        mapped = tm.map(source)
        if mapped is None:
            continue
        assert mapped >= previous - 1e-9
        previous = mapped
