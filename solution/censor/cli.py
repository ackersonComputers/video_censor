"""Command line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import ConfigError, Settings, load_wordlist
from .edl import EDLError, template
from .media import MediaError

EPILOG = """
examples:
  # Mute profanity using the file's own English subtitle track for accuracy
  censor run "Movie.mkv"

  # Inspect what would be muted without writing any video
  censor run "Movie.mkv" --report-only

  # Also apply a scene edit list (cuts, blackouts, blurs)
  censor run "Movie.mkv" --edl movie.edl.csv --edl-preview

  # Whole folder, unattended
  censor batch "E:/Television/Show" -o "E:/Clean/Show" --recursive --yes

  # Start a scene edit list
  censor edl-init movie.edl.csv
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="censor",
        description="Subtitle-anchored profanity muting and scene editing for video files.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"censor {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="process a single video file")
    _add_common(run)
    run.add_argument("input", type=Path, help="video file to process")
    run.add_argument("-o", "--output", type=Path, help="output file (default: <name>_censored.mkv)")

    batch = sub.add_parser("batch", help="process every video in a folder")
    _add_common(batch)
    batch.add_argument("input_dir", type=Path, help="folder to scan")
    batch.add_argument("-o", "--output-dir", type=Path, required=True, help="mirror output here")
    batch.add_argument("-r", "--recursive", action="store_true", help="descend into subfolders")
    batch.add_argument("--suffix", default="_censored", help="output filename suffix")
    batch.add_argument(
        "--edl-dir",
        type=Path,
        help="folder of per-video edit lists named <video stem>.edl.csv",
    )
    batch.add_argument(
        "--continue-on-error",
        action="store_true",
        help="keep going after a failure instead of stopping",
    )

    manual = sub.add_parser(
        "manual", help="apply an edit list only, with no detection (touch-up tool)"
    )
    _add_common(manual)
    manual.add_argument("input", type=Path)
    manual.add_argument("-o", "--output", type=Path)

    init = sub.add_parser("edl-init", help="write a commented starter edit list")
    init.add_argument("path", type=Path)

    words = sub.add_parser("words", help="show the resolved word list")
    words.add_argument("--wordlist", type=Path)
    words.add_argument("--tiers", help="comma-separated tiers, or 'all'")
    words.add_argument("--test", help="check whether a word or phrase would be censored")

    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    io = parser.add_argument_group("input / output")
    io.add_argument("--overwrite", action="store_true", help="replace an existing output")
    io.add_argument("--dry-run", action="store_true", help="print the ffmpeg command, change nothing")
    io.add_argument("--report-only", action="store_true", help="analyse and report; write no video")
    io.add_argument("-y", "--yes", action="store_true", help="assume yes to every prompt")
    io.add_argument("--report", type=Path, help="path for the JSON audit report")
    io.add_argument("--no-censored-subs", action="store_true", help="do not build a censored subtitle track")
    io.add_argument("--keep-original-subs", action="store_true", help="also keep the uncensored subtitle tracks")
    io.add_argument("--mask-style", default="___", help="text used to mask a word in subtitles")
    io.add_argument("--keep-temp", action="store_true", help="keep the extracted WAV")
    io.add_argument("--work-dir", type=Path, help="scratch directory")

    words = parser.add_argument_group("word list")
    words.add_argument("--wordlist", type=Path, help="custom word list YAML")
    words.add_argument("--tiers", help="comma-separated tiers to enable, or 'all'")
    words.add_argument("--add-word", action="append", default=[], metavar="WORD", help="censor an extra word")
    words.add_argument("--allow-word", action="append", default=[], metavar="WORD", help="never censor this word")

    streams = parser.add_argument_group("streams")
    streams.add_argument("--audio-index", type=int, default=0, help="audio stream to transcribe (0:a:N)")
    streams.add_argument("--video-index", type=int, default=0)
    streams.add_argument("--sub-lang", default="eng")
    streams.add_argument("--sub-stream", type=int, help="force a subtitle stream ordinal (0:s:N)")
    streams.add_argument("--subs", dest="external_subs", type=Path, help="use this subtitle file")
    streams.add_argument(
        "--only-target-audio",
        action="store_true",
        help="mute only --audio-index instead of every audio track",
    )

    asr = parser.add_argument_group("speech recognition")
    asr.add_argument("--model", default="large-v3")
    asr.add_argument("--language", default="en")
    asr.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    asr.add_argument("--engine", default="auto", choices=["auto", "whisperx", "faster-whisper", "openai-whisper"])
    asr.add_argument("--compute-type", default="float16")
    asr.add_argument("--beam-size", type=int, default=5)
    asr.add_argument("--batch-size", type=int, default=8)
    asr.add_argument("--no-asr-cache", action="store_true")
    asr.add_argument("--skip-asr", action="store_true", help="subtitle timing only (fast, coarse)")
    asr.add_argument("--allow-asr-fallback", action="store_true", help="do not prompt when subtitles are missing")

    fusion = parser.add_argument_group("timing fusion")
    fusion.add_argument("--sub-offset", type=float, help="force a subtitle offset in seconds")
    fusion.add_argument("--no-auto-offset", action="store_true")
    fusion.add_argument("--window-slack", type=float, default=2.5)
    fusion.add_argument("--fuzzy-threshold", type=float, default=0.80)
    fusion.add_argument("--blind-pad", type=float, default=0.45, help="buffer when timing is unknown")
    fusion.add_argument("--paranoid", action="store_true", help="mute the whole cue when unsure")
    fusion.add_argument("--extras-policy", default="mute", choices=["mute", "report", "ignore"],
                        help="what to do with profanity heard but absent from subtitles")
    fusion.add_argument("--merge-gap", type=float, default=0.08)
    fusion.add_argument("--max-interval", type=float, default=6.0)

    scene = parser.add_argument_group("scene edits")
    scene.add_argument("-e", "--edl", type=Path, help="scene edit list (CSV or JSON)")
    scene.add_argument("--edl-out", type=Path, help="write the normalised edit list here")
    scene.add_argument("--edl-preview", action="store_true", help="show subtitle cues around each edit")
    scene.add_argument("--blur-strength", type=float, default=30.0)
    scene.add_argument("--blur-mode", default="gblur", choices=["gblur", "pixelate"])

    enc = parser.add_argument_group("encoding")
    enc.add_argument("--video-encoder", default="hevc_nvenc")
    enc.add_argument("--video-quality", type=int, default=19, help="NVENC cq / x26x crf")
    enc.add_argument("--nvenc-preset", default="p7")
    enc.add_argument("--audio-codec", default="flac")
    enc.add_argument("--audio-bitrate", default="448k")
    enc.add_argument("--ffmpeg", default="ffmpeg")
    enc.add_argument("--ffprobe", default="ffprobe")


def settings_from_args(args: argparse.Namespace) -> Settings:
    tiers = None
    if getattr(args, "tiers", None):
        tiers = [t.strip() for t in str(args.tiers).split(",") if t.strip()]
    return Settings(
        input=getattr(args, "input", Path()),
        output=getattr(args, "output", None),
        report=args.report,
        dry_run=args.dry_run,
        report_only=args.report_only,
        overwrite=args.overwrite,
        assume_yes=args.yes,
        wordlist=args.wordlist,
        tiers=tiers,
        extra_words=list(args.add_word),
        allow_words=list(args.allow_word),
        audio_index=args.audio_index,
        video_index=args.video_index,
        sub_lang=args.sub_lang,
        sub_stream=args.sub_stream,
        external_subs=args.external_subs,
        mute_all_audio=not args.only_target_audio,
        model=args.model,
        language=args.language,
        device=args.device,
        engine=args.engine,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        batch_size=args.batch_size,
        no_asr_cache=args.no_asr_cache,
        skip_asr=args.skip_asr,
        allow_asr_fallback=args.allow_asr_fallback,
        no_detect=args.command == "manual",
        sub_offset=args.sub_offset,
        no_auto_offset=args.no_auto_offset,
        window_slack=args.window_slack,
        fuzzy_threshold=args.fuzzy_threshold,
        blind_pad=args.blind_pad,
        paranoid=args.paranoid,
        extras_policy=args.extras_policy,
        merge_gap=args.merge_gap,
        max_interval=args.max_interval,
        mask_style=args.mask_style,
        keep_original_subs=args.keep_original_subs,
        no_censored_subs=args.no_censored_subs or args.command == "manual",
        edl=args.edl,
        edl_out=args.edl_out,
        edl_preview=args.edl_preview,
        blur_strength=args.blur_strength,
        blur_mode=args.blur_mode,
        video_encoder=args.video_encoder,
        video_quality=args.video_quality,
        nvenc_preset=args.nvenc_preset,
        audio_codec=args.audio_codec,
        audio_bitrate=args.audio_bitrate,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
        work_dir=args.work_dir,
        keep_temp=args.keep_temp,
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Convenience: `censor "Movie.mkv"` behaves like `censor run "Movie.mkv"`.
    known = {"run", "batch", "manual", "edl-init", "words", "-h", "--help", "--version"}
    if argv and argv[0] not in known and not argv[0].startswith("-"):
        argv.insert(0, "run")

    args = build_parser().parse_args(argv)

    if args.command == "manual" and not args.edl:
        print("error: manual mode needs an edit list: --edl <file>", file=sys.stderr)
        return 2

    try:
        if args.command == "edl-init":
            path = template(args.path)
            print(f"Wrote starter edit list to {path}")
            return 0
        if args.command == "words":
            return _show_words(args)

        from . import pipeline  # deferred: keeps --help instant

        if args.command == "batch":
            from .batch import run_batch  # noqa: PLC0415

            return run_batch(args, settings_from_args(args))

        result = pipeline.process(settings_from_args(args))
        if result.skipped:
            print(f"[skip] {result.skipped}")
        return 0
    except (ConfigError, EDLError, MediaError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pipeline.PipelineAbort and anything unexpected
        if type(exc).__name__ == "PipelineAbort":
            print(f"error: {exc}", file=sys.stderr)
            return 3
        raise
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


def _show_words(args: argparse.Namespace) -> int:
    tiers = [t.strip() for t in str(args.tiers).split(",")] if args.tiers else None
    wordlist = load_wordlist(args.wordlist, tiers)
    if args.test:
        from .matcher import Matcher  # noqa: PLC0415

        matches = Matcher(wordlist).find(args.test)
        if not matches:
            print(f"{args.test!r} would NOT be censored")
            return 1
        for match in matches:
            print(f"{match.text!r} -> tier={match.tier} rule={match.rule} entry={match.key!r}")
        return 0
    print(wordlist.summary())
    for label, entries in (
        ("exact", wordlist.exact),
        ("substring", wordlist.substring),
        ("phrase", wordlist.phrases),
    ):
        for key, tier in sorted(entries.items()):
            print(f"  {label:<9} {tier:<10} {key}")
    print(f"  exclusions: {len(wordlist.exclusions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
