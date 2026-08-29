import pytest

from censor.align import (
    AlignStats,
    Interval,
    TimingModel,
    build_timing_model,
    merge_intervals,
    resolve,
)
from censor.asr import Word
from censor.config import Settings, load_wordlist
from censor.matcher import Matcher
from censor.subs import Cue


def matcher() -> Matcher:
    return Matcher(load_wordlist(tiers=["profanity", "religious"]))


def settings(**kwargs) -> Settings:
    return Settings(extras_policy="ignore", **kwargs)


def words(*specs) -> list[Word]:
    return [Word(text=t, start=s, end=e, score=sc) for t, s, e, sc in specs]


# --------------------------------------------------------------------------- #
# Ladder
# --------------------------------------------------------------------------- #


def test_l1_uses_the_exact_asr_word_timing():
    cues = [Cue(1, 10.0, 12.0, "Oh fuck this.")]
    asr = words(("Oh", 10.1, 10.3, 0.9), ("fuck", 10.4, 10.8, 0.9), ("this", 10.9, 11.2, 0.9))
    intervals, stats = resolve(cues, asr, matcher(), settings())
    assert stats.levels["L1"] == 1
    assert intervals[0].start == pytest.approx(10.34, abs=0.01)
    assert intervals[0].end == pytest.approx(10.92, abs=0.01)


def test_l2_catches_a_misheard_word():
    cues = [Cue(1, 10.0, 12.0, "Oh fuck this.")]
    asr = words(("Oh", 10.1, 10.3, 0.9), ("fock", 10.4, 10.8, 0.4), ("this", 10.9, 11.2, 0.9))
    _, stats = resolve(cues, asr, matcher(), settings())
    assert stats.levels["L2"] == 1


def test_l3_interpolates_between_surrounding_anchors():
    cues = [Cue(1, 10.0, 13.0, "Listen here you fucking idiot okay")]
    asr = words(
        ("Listen", 10.0, 10.4, 0.9),
        ("here", 10.4, 10.7, 0.9),
        ("you", 10.7, 10.9, 0.9),
        ("[music]", 11.0, 11.6, 0.2),
        ("idiot", 11.7, 12.1, 0.9),
        ("okay", 12.2, 12.5, 0.9),
    )
    intervals, stats = resolve(cues, asr, matcher(), settings())
    assert stats.levels["L3"] == 1
    # The swear sits between "you" (ends 10.9) and "idiot" (starts 11.7).
    assert 10.5 < intervals[0].start < 11.6
    assert 11.0 < intervals[0].end < 12.2


def test_l4_falls_back_to_a_proportional_slice_of_the_cue():
    cues = [Cue(1, 10.0, 14.0, "one two three fucking five six seven")]
    asr = words(("unrelated", 40.0, 40.5, 0.9))
    intervals, stats = resolve(cues, asr, matcher(), settings())
    assert stats.levels["L4"] == 1
    # Slice must land inside the cue plus the blind pad, not the whole file.
    assert intervals[0].start > 9.0
    assert intervals[0].end < 15.0


def test_paranoid_mode_mutes_the_whole_cue():
    cues = [Cue(1, 10.0, 14.0, "one two three fucking five")]
    intervals, stats = resolve(cues, [], matcher(), settings(paranoid=True))
    assert stats.levels["L5"] == 1
    assert intervals[0].start <= 10.0
    assert intervals[0].end >= 14.0


def test_a_missed_swear_is_never_silently_dropped():
    """Whatever the evidence, every subtitle hit must yield an interval."""
    cues = [Cue(1, 10.0, 12.0, "Oh fuck."), Cue(2, 30.0, 31.0, "Bloody hell.")]
    for asr in ([], words(("noise", 99.0, 99.2, 0.1))):
        intervals, stats = resolve(cues, asr, matcher(), settings())
        assert stats.subtitle_hits == len(intervals) >= 2


# --------------------------------------------------------------------------- #
# ASR-only extras
# --------------------------------------------------------------------------- #


def test_asr_only_hit_is_muted_when_policy_allows():
    cues = [Cue(1, 10.0, 12.0, "Nothing rude here at all.")]
    asr = words(("fuck", 50.0, 50.4, 0.9))
    intervals, stats = resolve(cues, asr, matcher(), Settings(extras_policy="mute"))
    assert stats.asr_only_hits == 1
    assert any(i.level == "ASR" for i in intervals)


def test_low_confidence_asr_only_hit_is_not_muted():
    asr = words(("fuck", 50.0, 50.4, 0.05))
    intervals, stats = resolve([], asr, matcher(), Settings(extras_policy="mute"))
    assert stats.extras_dropped == 1
    assert intervals == []


def test_report_policy_records_but_does_not_mute():
    asr = words(("fuck", 50.0, 50.4, 0.99))
    intervals, stats = resolve([], asr, matcher(), Settings(extras_policy="report"))
    assert stats.asr_only_hits == 1
    assert intervals == []


# --------------------------------------------------------------------------- #
# Sync
# --------------------------------------------------------------------------- #


def test_offset_is_recovered_from_a_shifted_subtitle_file():
    shift = 2.4
    vocabulary = [
        "hospital", "detective", "corridor", "envelope", "sergeant", "kitchen",
        "witness", "briefcase", "helicopter", "november", "sandwich", "portrait",
        "manager", "curtain", "trumpet", "granite", "sailor", "orchard",
        "printer", "compass", "meadow", "juniper", "lantern", "quarry",
    ]
    cues = []
    asr = []
    for index, word in enumerate(vocabulary):
        start = 10.0 + index * 20.0
        cues.append(Cue(index + 1, start, start + 1.5, f"{word} again"))
        asr.append(Word(text=word, start=start + shift, end=start + shift + 0.4))
        asr.append(Word(text="again", start=start + shift + 0.5, end=start + shift + 0.9))

    stats = AlignStats()
    model = build_timing_model(cues, asr, stats=stats, log=lambda *a: None)
    assert model.to_asr(100.0) == pytest.approx(100.0 + shift, abs=0.5)
    assert stats.offset_confidence > 0.5


def test_forced_offset_is_respected():
    model = build_timing_model([], [], forced_offset=-1.25)
    assert model.to_asr(50.0) == pytest.approx(48.75)


def test_timing_model_interpolates_drift_between_anchors():
    # Anchors are (subtitle time, audio time) pairs: no shift at the start,
    # ten seconds of accumulated drift by the end.
    model = TimingModel(anchors=[(0.0, 0.0), (1000.0, 1010.0)])
    assert model.to_asr(500.0) == pytest.approx(505.0)
    assert model.offset_at(1000.0) == pytest.approx(10.0)


def test_shifted_subtitles_still_produce_a_tight_interval():
    shift = 3.0
    cues = [Cue(1, 10.0, 12.0, "Oh fuck this")]
    asr = words(
        ("Oh", 10.0 + shift, 10.3 + shift, 0.9),
        ("fuck", 10.4 + shift, 10.8 + shift, 0.9),
        ("this", 10.9 + shift, 11.2 + shift, 0.9),
    )
    intervals, _ = resolve(cues, asr, matcher(), settings(sub_offset=shift))
    assert intervals[0].start == pytest.approx(13.34, abs=0.02)


# --------------------------------------------------------------------------- #
# Merging
# --------------------------------------------------------------------------- #


def interval(start, end, level="L1"):
    return Interval(start=start, end=end, level=level, token="x", tier="t", confidence=1.0)


def test_overlapping_intervals_merge():
    merged = merge_intervals([interval(1.0, 2.0), interval(1.9, 3.0)], gap=0.08)
    assert len(merged) == 1
    assert merged[0].end == pytest.approx(3.0)


def test_intervals_separated_by_more_than_the_gap_stay_apart():
    merged = merge_intervals([interval(1.0, 2.0), interval(2.5, 3.0)], gap=0.08)
    assert len(merged) == 2


def test_merge_keeps_the_least_certain_level():
    merged = merge_intervals([interval(1.0, 2.0, "L1"), interval(2.0, 3.0, "L4")])
    assert merged[0].level == "L4"


def test_runaway_intervals_are_capped():
    merged = merge_intervals([interval(10.0, 90.0)], max_duration=6.0)
    assert merged[0].end - merged[0].start == pytest.approx(6.0)


def test_intervals_are_clamped_to_the_file_length():
    merged = merge_intervals([interval(-5.0, 3.0), interval(98.0, 120.0)], limit=100.0)
    assert merged[0].start == pytest.approx(0.0)
    assert merged[-1].end <= 100.0
