"""End-to-end orchestration for a single file.

Pipeline order matters, and it is deliberate:

  probe -> subtitles -> ASR+align -> fusion (SOURCE timeline mute intervals)
        -> load EDL -> drop mutes inside cuts -> build TimeMap
        -> censor + remap subtitles and chapters -> one filtergraph -> mux

Analysis always happens on the source timeline; cuts are applied last through a
single ``TimeMap``, so nothing downstream can quietly desynchronise.
"""

from __future__ import annotations

import datetime as _dt
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import align, edl as edl_mod, media, render as render_mod, report as report_mod, subs as subs_mod
from .asr import Transcript, load_cache, save_cache, transcribe
from .config import Settings, confirm, load_wordlist
from .matcher import Matcher, norm_token
from .timeline import TimeMap, write_chapters_file


class PipelineAbort(RuntimeError):
    """Raised when the user declines a fallback or a prerequisite is missing."""


@dataclass
class Result:
    input: Path
    output: Path | None
    intervals: list[align.Interval] = field(default_factory=list)
    stats: align.AlignStats = field(default_factory=align.AlignStats)
    report_path: Path | None = None
    srt_path: Path | None = None
    warnings: list[str] = field(default_factory=list)
    skipped: str = ""


def process(settings: Settings, *, log=print) -> Result:
    media.ensure_tools(settings.ffmpeg, settings.ffprobe)

    source = settings.input
    if not source.is_file():
        raise PipelineAbort(f"Input not found: {source}")

    output = settings.output or settings.default_output()
    if output.exists() and not settings.overwrite and not (settings.dry_run or settings.report_only):
        return Result(input=source, output=output, skipped="output exists (use --overwrite)")

    work = settings.work_dir or (output.parent / ".censor" / source.stem)
    work.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    log(f"[probe] {source.name}")
    info = media.probe(source, settings.ffprobe)
    if not info.of_type("audio"):
        raise PipelineAbort(f"{source} has no audio stream")
    log(media.describe_streams(info))
    log(f"[probe] duration {info.duration / 60:.1f} min")

    wordlist = load_wordlist(settings.wordlist, settings.tiers)
    for extra in settings.extra_words:
        token = norm_token(extra)
        if token:
            wordlist.exact[token] = "custom"
    for allowed in settings.allow_words:
        token = norm_token(allowed)
        if token:
            wordlist.exclusions.add(token)
            wordlist.exact.pop(token, None)
    log(f"[words] {wordlist.summary()}")
    matcher = Matcher(wordlist)

    # --- 1. subtitles ---------------------------------------------------- #
    if settings.no_detect:
        cues, sub_origin = [], "not loaded (manual mode)"
        transcript = Transcript(
            words=[], engine="none", model=settings.model, language=settings.language
        )
        intervals, stats = [], align.AlignStats()
        log("[detect] manual mode: applying the edit list only")
    else:
        cues, sub_origin = _load_cues(settings, info, work, warnings, log=log)

        # --- 2. ASR ------------------------------------------------------- #
        transcript = _run_asr(settings, info, work, cues, warnings, log=log)

        # --- 3. fusion ---------------------------------------------------- #
        if cues:
            intervals, stats = align.resolve(cues, transcript.words, matcher, settings, log=log)
        elif transcript.words:
            log("[align] no subtitles: relying on ASR alone with a wider buffer")
            wide = _widen_for_asr_only(settings)
            intervals, stats = align.resolve([], transcript.words, matcher, wide, log=log)
        else:
            intervals, stats = [], align.AlignStats()
            warnings.append("no subtitles and no transcript: nothing was detected")

    intervals = align.merge_intervals(
        intervals,
        gap=settings.merge_gap,
        min_duration=settings.min_duration,
        max_duration=settings.max_interval,
        limit=info.duration or None,
    )
    log(f"[align] {len(intervals)} mute interval(s) on the source timeline")

    # --- 4. scene edit list ------------------------------------------------ #
    edits: list[edl_mod.Edit] = []
    if settings.edl:
        edits = edl_mod.validate(edl_mod.load(settings.edl), info.duration, log=log)
        log(
            f"[edl] {len(edits)} edit(s); "
            + ", ".join(
                f"{action}={len(edl_mod.of_action(edits, action))}"
                for action in edl_mod.ACTIONS
                if edl_mod.of_action(edits, action)
            )
        )
        _warn_mid_word_cuts(edits, transcript, warnings, log=log)
        if settings.edl_preview:
            report_mod.preview_edits(edits, cues, log=log)
        if settings.edl_out:
            edl_mod.save_json(edits, settings.edl_out)
            log(f"[edl] canonical JSON written to {settings.edl_out}")

    cut_regions = edl_mod.cuts(edits)
    if cut_regions:
        before = len(intervals)
        intervals = align.subtract(intervals, cut_regions)
        if before != len(intervals):
            log(f"[edl] {before - len(intervals)} mute interval(s) fell inside cuts")

    timemap = TimeMap(cut_regions, duration=info.duration)

    # --- 5. censored subtitles --------------------------------------------- #
    srt_path: Path | None = None
    censored_cues = []
    if cues and not settings.no_censored_subs:
        censored_cues, hits = subs_mod.censor_cues(cues, matcher, settings.mask_style)
        censored_cues = timemap.map_cues(censored_cues) if timemap else censored_cues
        srt_path = output.with_suffix("").with_name(f"{output.stem}.censored.srt")
        subs_mod.write_srt(censored_cues, srt_path)
        log(f"[subs] {hits} caption word(s) masked -> {srt_path.name}")

    chapters_file: Path | None = None
    if timemap and info.chapters:
        mapped = timemap.map_chapters(info.chapters)
        chapters_file = write_chapters_file(mapped, work / "chapters.ffmeta")
        log(f"[edl] {len(info.chapters)} chapter(s) -> {len(mapped)} after cuts")

    # --- 6. report ---------------------------------------------------------- #
    report = report_mod.Report(
        input=str(source),
        output=str(output),
        generated=_dt.datetime.now().isoformat(timespec="seconds"),
        settings=settings.to_dict(),
        sources={
            "subtitles": sub_origin,
            "asr": f"{transcript.engine}/{transcript.model}" if transcript.words else "none",
            "audio_stream": settings.audio_index,
        },
        stats=stats.to_dict(),
        intervals=report_mod.build_intervals(intervals, timemap if timemap else None),
        edits=[e.to_dict() for e in edits],
        warnings=warnings,
        timing={
            "source_duration": round(info.duration, 3),
            "output_duration": round(timemap.output_duration if timemap else info.duration, 3),
            "removed": round(timemap.total_removed, 3),
        },
    )
    report_path = settings.report or output.with_name(f"{output.stem}.censor.json")
    report.save(report_path)

    log(
        report_mod.summarize(
            intervals=intervals,
            stats=stats,
            edits=edits,
            timemap=timemap if timemap else None,
            duration=info.duration,
            sources=report.sources,
            warnings=warnings,
        )
    )
    log(f"[report] {report_path}")

    if settings.report_only:
        return Result(
            input=source,
            output=None,
            intervals=intervals,
            stats=stats,
            report_path=report_path,
            srt_path=srt_path,
            warnings=warnings,
            skipped="report only",
        )

    if not intervals and not edits:
        log("[render] nothing to change; no output written")
        return Result(
            input=source,
            output=None,
            intervals=intervals,
            stats=stats,
            report_path=report_path,
            srt_path=srt_path,
            warnings=warnings,
            skipped="nothing detected",
        )

    # --- 7. render ----------------------------------------------------------- #
    plan = render_mod.build_plan(
        settings=settings,
        info=info,
        intervals=intervals,
        edits=edits,
        timemap=timemap,
        output=output,
        censored_srt=srt_path if (srt_path and not settings.no_censored_subs) else None,
        chapters_file=chapters_file,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and settings.overwrite and not settings.dry_run:
        output.unlink()
    render_mod.render(plan, dry_run=settings.dry_run, log=log)

    if not settings.dry_run:
        log(f"[done] {output}")
    if not settings.keep_temp:
        for leftover in work.glob("*.wav"):
            leftover.unlink(missing_ok=True)

    return Result(
        input=source,
        output=output,
        intervals=intervals,
        stats=stats,
        report_path=report_path,
        srt_path=srt_path,
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


def _load_cues(
    settings: Settings,
    info: media.MediaInfo,
    work: Path,
    warnings: list[str],
    *,
    log,
) -> tuple[list[subs_mod.Cue], str]:
    """Embedded text track -> sidecar file -> nothing (ASR only)."""
    if settings.external_subs:
        cues = subs_mod.parse_subtitle_file(settings.external_subs)
        log(f"[subs] {len(cues)} cue(s) from {settings.external_subs.name} (explicit)")
        return cues, f"external:{settings.external_subs.name}"

    choice = media.pick_subtitle_stream(info, settings.sub_lang, settings.sub_stream)
    if choice is not None and choice.is_text:
        target = work / f"embedded.s{choice.ordinal}.srt"
        try:
            media.extract_subtitles(info, choice.ordinal, target, settings.ffmpeg)
            cues = subs_mod.parse_subtitle_file(target)
            if cues:
                log(
                    f"[subs] {len(cues)} cue(s) from embedded stream "
                    f"0:s:{choice.ordinal} ({choice.codec}, {choice.reason})"
                )
                return cues, f"embedded:0:s:{choice.ordinal}"
            warnings.append(f"embedded subtitle stream 0:s:{choice.ordinal} was empty")
        except (media.MediaError, subs_mod.SubtitleError) as exc:
            warnings.append(f"embedded subtitle extraction failed: {exc}")
            log(f"[subs] ! {exc}")
    elif choice is not None:
        warnings.append(f"subtitles unusable: {choice.reason}")
        log(f"[subs] ! {choice.reason}")

    sidecar = media.find_sidecar_subtitle(info.path, settings.sub_lang)
    if sidecar is not None:
        try:
            cues = subs_mod.parse_subtitle_file(sidecar)
            if cues:
                log(f"[subs] {len(cues)} cue(s) from sidecar {sidecar.name}")
                return cues, f"sidecar:{sidecar.name}"
        except subs_mod.SubtitleError as exc:
            warnings.append(f"sidecar subtitle unreadable: {exc}")

    message = (
        f"No usable English subtitles found for {info.path.name}.\n"
        "Falling back to speech recognition alone. That is markedly less "
        "reliable - misheard words over music are exactly how swears slip "
        "through - so wider mute buffers will be used."
    )
    log(f"[subs] ! {message}")
    if not (settings.allow_asr_fallback or settings.report_only or settings.dry_run):
        if not confirm("Continue with ASR-only mode?", assume_yes=settings.assume_yes):
            raise PipelineAbort(
                "Aborted: no subtitles available and ASR-only mode was declined. "
                "Pass --allow-asr-fallback (or --yes) to accept it automatically."
            )
    warnings.append("ASR-only mode: no subtitle track was available")
    return [], "none (asr only)"


def _run_asr(
    settings: Settings,
    info: media.MediaInfo,
    work: Path,
    cues: list[subs_mod.Cue],
    warnings: list[str],
    *,
    log,
) -> Transcript:
    empty = Transcript(words=[], engine="none", model=settings.model, language=settings.language)
    if settings.skip_asr:
        if not cues:
            raise PipelineAbort("--skip-asr requires a subtitle track to work from")
        log("[asr] skipped; every hit will use subtitle cue timing (level L4/L5)")
        return empty

    wav = work / "audio.wav"
    cache = settings.asr_cache or work / "asr.json"

    if not settings.no_asr_cache and wav.is_file():
        cached = load_cache(cache, wav, settings.model, settings.language)
        if cached:
            log(f"[asr] reusing cached transcript ({len(cached.words)} words)")
            return cached

    log(f"[asr] extracting audio stream 0:a:{settings.audio_index}")
    media.extract_wav(info, settings.audio_index, wav, settings.ffmpeg)

    if not settings.no_asr_cache:
        cached = load_cache(cache, wav, settings.model, settings.language)
        if cached:
            log(f"[asr] reusing cached transcript ({len(cached.words)} words)")
            return cached

    try:
        transcript = transcribe(
            wav,
            model=settings.model,
            language=settings.language,
            device=settings.resolved_device(),
            compute_type=settings.compute_type,
            beam_size=settings.beam_size,
            batch_size=settings.batch_size,
            engine=settings.engine,
            log=log,
        )
    except Exception as exc:
        if not cues:
            raise
        warnings.append(f"ASR failed ({exc}); subtitle cue timing used instead")
        log(f"[asr] ! {exc}\n[asr] continuing with subtitle cue timing only")
        return empty

    if not settings.no_asr_cache:
        save_cache(cache, wav, transcript)
    return transcript


def _widen_for_asr_only(settings: Settings) -> Settings:
    """Without subtitles the transcript is the only evidence, so pad harder."""
    from dataclasses import replace

    return replace(
        settings,
        pad_exact=(settings.pad_exact[0] + 0.08, settings.pad_exact[1] + 0.15),
        pad_fuzzy=(settings.pad_fuzzy[0] + 0.10, settings.pad_fuzzy[1] + 0.20),
        extras_min_score=min(settings.extras_min_score, 0.35),
    )


def _warn_mid_word_cuts(
    edits: list[edl_mod.Edit],
    transcript: Transcript,
    warnings: list[str],
    *,
    log,
) -> None:
    """Flag cut boundaries that land in the middle of a spoken word."""
    if not transcript.words:
        return
    for edit in edl_mod.of_action(edits, "cut"):
        for label, moment in (("start", edit.start), ("end", edit.end)):
            for word in transcript.words:
                if word.start < moment < word.end and (word.end - word.start) > 0.05:
                    message = (
                        f"cut {label} at {edl_mod.format_time(moment)} lands inside the "
                        f"spoken word {word.text!r} "
                        f"({edl_mod.format_time(word.start)}-{edl_mod.format_time(word.end)}); "
                        "nudge it to a pause for a cleaner seam"
                    )
                    warnings.append(message)
                    log(f"[edl] ! {message}")
                    break


def cleanup(work: Path) -> None:
    shutil.rmtree(work, ignore_errors=True)
