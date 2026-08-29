import pytest

from censor import edl as edl_mod
from censor.config import load_wordlist
from censor.matcher import Matcher
from censor.render import between, build_plan, volume_chain
from censor.subs import Cue, censor_text, clean_cue_text, parse_subtitle_file, write_srt


# --------------------------------------------------------------------------- #
# Subtitles
# --------------------------------------------------------------------------- #


def test_clean_strips_tags_sdh_and_speaker_labels():
    assert clean_cue_text("<i>MAN: [SIGHS] Get out.</i>") == "Get out."
    assert clean_cue_text("{\\an8}\u266a music \u266a") == "music"
    assert clean_cue_text("- Hey.\n- Hi.") == "Hey. Hi."


def test_censor_replaces_only_the_target_word():
    m = Matcher(load_wordlist(tiers=["profanity"]))
    text, hits = censor_text("Oh, fuck this shit.", m, "___")
    assert hits == 2
    assert text == "Oh, ___ this ___."


def test_censor_preserves_surrounding_markup():
    m = Matcher(load_wordlist(tiers=["profanity"]))
    text, _ = censor_text("<i>You fucking idiot</i>", m, "<censored>")
    assert text == "<i>You <censored> idiot</i>"


def test_srt_roundtrip(tmp_path):
    cues = [Cue(1, 1.5, 3.25, "Hello there"), Cue(2, 10.0, 11.0, "Bye")]
    path = write_srt(cues, tmp_path / "out.srt")
    assert "00:00:01,500 --> 00:00:03,250" in path.read_text(encoding="utf-8")
    reparsed = parse_subtitle_file(path)
    assert [c.text for c in reparsed] == ["Hello there", "Bye"]
    assert reparsed[0].start == pytest.approx(1.5)


# --------------------------------------------------------------------------- #
# Edit list
# --------------------------------------------------------------------------- #


def test_all_timestamp_formats_parse():
    assert edl_mod.parse_time("12.5") == pytest.approx(12.5)
    assert edl_mod.parse_time("1:02") == pytest.approx(62.0)
    assert edl_mod.parse_time("01:02:03.250") == pytest.approx(3723.25)
    assert edl_mod.parse_time("00:12:34,500") == pytest.approx(754.5)


def test_csv_edit_list_parses(tmp_path):
    path = tmp_path / "edits.csv"
    path.write_text(
        "# comment\n"
        "00:42:10.500,00:43:58.000,cut,pool scene\n"
        "01:12:04.000,01:13:20.500,blackout,plot,fade=0.5\n"
        "01:31:00,01:31:40,blur,,strength=40\n"
        "18:22,18:26,mute,slur\n",
        encoding="utf-8",
    )
    edits = edl_mod.validate(edl_mod.load(path), duration=7200.0)
    assert [e.action for e in edits] == ["mute", "cut", "blackout", "blur"]
    assert edits[2].param_float("fade", 0.0) == pytest.approx(0.5)
    assert edits[3].param_float("strength", 30.0) == pytest.approx(40.0)


def test_action_aliases_are_accepted(tmp_path):
    path = tmp_path / "e.csv"
    path.write_text("1,2,skip\n3,4,black\n5,6,silence\n", encoding="utf-8")
    assert [e.action for e in edl_mod.load(path)] == ["cut", "blackout", "mute"]


def test_overlapping_cuts_are_rejected():
    edits = [
        edl_mod.Edit(10.0, 20.0, "cut"),
        edl_mod.Edit(15.0, 25.0, "cut"),
    ]
    with pytest.raises(edl_mod.EDLError, match="overlap"):
        edl_mod.validate(edits, duration=100.0)


def test_overlapping_non_cuts_are_allowed():
    edits = [edl_mod.Edit(10.0, 20.0, "blur"), edl_mod.Edit(15.0, 25.0, "mute")]
    assert len(edl_mod.validate(edits, duration=100.0)) == 2


def test_backwards_interval_is_rejected():
    with pytest.raises(edl_mod.EDLError, match="end must be after start"):
        edl_mod.validate([edl_mod.Edit(20.0, 10.0, "cut")], duration=100.0)


def test_edits_are_clamped_to_the_file_length():
    edits = edl_mod.validate([edl_mod.Edit(90.0, 500.0, "cut")], duration=100.0, log=lambda *a: None)
    assert edits[0].end == pytest.approx(100.0)


def test_json_roundtrip(tmp_path):
    original = edl_mod.validate(
        [edl_mod.Edit(10.0, 20.0, "cut", "note"), edl_mod.Edit(30.0, 40.0, "blur", "", {"strength": "12"})],
        duration=100.0,
    )
    path = edl_mod.save_json(original, tmp_path / "e.json")
    reloaded = edl_mod.load(path)
    assert [e.action for e in reloaded] == ["cut", "blur"]
    assert reloaded[1].params["strength"] == "12"


def test_mute_only_list_does_not_force_a_video_encode():
    assert not edl_mod.requires_video_encode([edl_mod.Edit(1.0, 2.0, "mute")])
    assert edl_mod.requires_video_encode([edl_mod.Edit(1.0, 2.0, "blackout")])


# --------------------------------------------------------------------------- #
# Filter construction
# --------------------------------------------------------------------------- #


def test_long_interval_lists_are_split_across_several_filters():
    """Keep sample-accurate expressions bounded on long files."""
    intervals = [(float(i), float(i) + 0.5) for i in range(400)]
    chain = volume_chain(intervals, chunk=60)
    assert len(chain) == 7
    assert sum(f.count("min(1,max(0,max(") for f in chain) == 400
    assert max(len(f) for f in chain) < 5000


def test_mute_filter_uses_sample_accurate_ten_millisecond_fades():
    chain = volume_chain([(1.0, 2.0)])
    assert chain == [
        "aeval=exprs='val(ch)*min(1,max(0,max((1.000-t)/0.010,(t-2.000)/0.010)))':channel_layout=1c"
    ]
    assert "volume=" not in chain[0]


def test_between_expression_shape():
    assert between([(1.0, 2.0)]) == "between(t,1.000,2.000)"


class FakeInfo:
    def __init__(self, audio=2, chapters=None):
        self.path = "in.mkv"
        self.duration = 100.0
        self.chapters = chapters or []
        self._audio = audio

    def of_type(self, kind):
        if kind == "audio":
            return [
                {
                    "_ordinal": i,
                    "codec_name": "eac3",
                    "channels": 6,
                    "channel_layout": "5.1",
                }
                for i in range(self._audio)
            ]
        if kind == "video":
            return [{"_ordinal": 0, "pix_fmt": "yuv420p"}]
        return []


def plan_for(edits, intervals=None, **kwargs):
    from pathlib import Path

    from censor.align import Interval
    from censor.config import Settings
    from censor.timeline import TimeMap

    settings = Settings(input=Path("in.mkv"), **kwargs)
    intervals = intervals or [Interval(1.0, 2.0, "L1", "fuck", "profanity", 1.0)]
    return build_plan(
        settings=settings,
        info=FakeInfo(),
        intervals=intervals,
        edits=edits,
        timemap=TimeMap(edl_mod.cuts(edits), duration=100.0),
        output=Path("out.mkv"),
    )


def test_mute_only_run_copies_the_video_stream():
    plan = plan_for([])
    assert not plan.reencodes_video
    assert "-c:v" in plan.command and "copy" in plan.command


def test_a_blackout_forces_a_video_encode():
    plan = plan_for([edl_mod.Edit(10.0, 20.0, "blackout")])
    assert plan.reencodes_video
    graph = plan.command[plan.command.index("-filter_complex") + 1]
    assert "drawbox" in graph


def test_a_blur_uses_gblur_with_a_time_window():
    plan = plan_for([edl_mod.Edit(10.0, 20.0, "blur", "", {"strength": "40"})])
    graph = plan.command[plan.command.index("-filter_complex") + 1]
    assert "gblur=sigma=40" in graph
    assert "between(t,10.000,20.000)" in graph


def test_cuts_produce_trim_and_concat_for_video_and_audio():
    plan = plan_for([edl_mod.Edit(10.0, 20.0, "cut"), edl_mod.Edit(50.0, 55.0, "cut")])
    graph = plan.command[plan.command.index("-filter_complex") + 1]
    assert plan.segments == 3
    assert graph.count("trim=start=") >= 3
    assert "concat=n=3:v=1:a=0" in graph
    assert "concat=n=3:v=0:a=1" in graph
    assert "asetpts=PTS-STARTPTS" in graph


def test_every_audio_track_is_muted_by_default():
    plan = plan_for([])
    graph = plan.command[plan.command.index("-filter_complex") + 1]
    assert "[0:a:0]" in graph and "[0:a:1]" in graph
    assert plan.audio_targets == [0, 1]


def test_only_target_audio_leaves_the_other_track_untouched():
    plan = plan_for([], mute_all_audio=False)
    graph = plan.command[plan.command.index("-filter_complex") + 1]
    assert "[0:a:0]" in graph
    assert "[0:a:1]" not in graph
    assert "-c:a:1" in plan.command


def test_manual_mute_rows_are_applied_alongside_detected_ones():
    plan = plan_for([edl_mod.Edit(70.0, 72.0, "mute")])
    graph = plan.command[plan.command.index("-filter_complex") + 1]
    assert "(70.000-t)/0.010" in graph
    assert "(t-72.000)/0.010" in graph
    assert plan.intervals == 2
