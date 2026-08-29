"""Audit reporting: machine-readable JSON plus a human summary."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .align import AlignStats, Interval, LEVEL_NAMES
from .edl import Edit, format_time
from .timeline import TimeMap


@dataclass
class Report:
    input: str = ""
    output: str = ""
    generated: str = ""
    settings: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    intervals: list[dict] = field(default_factory=list)
    edits: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timing: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "input": self.input,
            "output": self.output,
            "generated": self.generated,
            "sources": self.sources,
            "stats": self.stats,
            "timing": self.timing,
            "edits": self.edits,
            "intervals": self.intervals,
            "warnings": self.warnings,
            "settings": self.settings,
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def build_intervals(
    intervals: list[Interval], timemap: TimeMap | None
) -> list[dict]:
    """Record both source and output times so every mute stays traceable."""
    out: list[dict] = []
    for interval in intervals:
        record = interval.to_dict()
        record["start_hms"] = format_time(interval.start)
        record["end_hms"] = format_time(interval.end)
        if timemap:
            mapped = timemap.map_interval(interval.start, interval.end)
            record["output"] = [
                {"start": round(s, 3), "end": round(e, 3), "start_hms": format_time(s)}
                for s, e in mapped
            ]
        out.append(record)
    return out


def summarize(
    *,
    intervals: list[Interval],
    stats: AlignStats,
    edits: list[Edit],
    timemap: TimeMap | None,
    duration: float,
    sources: dict,
    warnings: list[str],
) -> str:
    total_muted = sum(i.duration for i in intervals)
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 66)
    lines.append("CENSOR SUMMARY")
    lines.append("=" * 66)

    lines.append(f"  subtitle source : {sources.get('subtitles', 'none')}")
    lines.append(f"  asr engine      : {sources.get('asr', 'none')}")
    if stats.offset_confidence:
        lines.append(
            f"  subtitle sync   : {stats.offset_seconds:+.3f}s "
            f"({stats.offset_confidence:.0%} confidence, {stats.drift_anchors} anchors)"
        )

    lines.append("")
    lines.append(
        f"  mute intervals  : {len(intervals)} covering {total_muted:.1f}s "
        f"({_percent(total_muted, duration)} of runtime)"
    )
    lines.append(f"  subtitle hits   : {stats.subtitle_hits}")
    lines.append(f"  asr-only hits   : {stats.asr_only_hits} ({stats.extras_dropped} not muted)")

    if stats.levels:
        lines.append("")
        lines.append("  timing quality:")
        for level in ("L1", "L2", "L3", "L4", "L5", "ASR"):
            count = stats.levels.get(level, 0)
            if count:
                lines.append(f"    {level}  {count:>4}  {LEVEL_NAMES[level]}")
        vague = sum(stats.levels.get(level, 0) for level in ("L4", "L5"))
        if vague:
            lines.append(
                f"    -> {vague} hit(s) could not be pinned to a word and were "
                "muted with a wide buffer."
            )

    if stats.tiers:
        lines.append("")
        lines.append("  by tier: " + ", ".join(f"{k}={v}" for k, v in sorted(stats.tiers.items())))

    if edits:
        counts = Counter(e.action for e in edits)
        lines.append("")
        lines.append("  scene edits: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        if timemap and timemap.total_removed:
            lines.append(
                f"    removed {timemap.total_removed:.1f}s; runtime "
                f"{format_time(duration)} -> {format_time(timemap.output_duration)}"
            )

    if warnings:
        lines.append("")
        lines.append("  warnings:")
        for warning in warnings:
            lines.append(f"    ! {warning}")

    lines.append("=" * 66)
    return "\n".join(lines)


def _percent(part: float, whole: float) -> str:
    if whole <= 0:
        return "n/a"
    return f"{100.0 * part / whole:.2f}%"


def preview_edits(edits: list[Edit], cues, log=print) -> None:
    """Print the subtitle lines each scene edit touches, to check boundaries."""
    if not edits:
        log("[edl] no edits to preview")
        return
    for edit in edits:
        log("")
        log(
            f"[edl] {edit.action.upper()} {format_time(edit.start)} -> "
            f"{format_time(edit.end)}  ({edit.duration:.1f}s)"
            + (f"  # {edit.note}" if edit.note else "")
        )
        touching = [c for c in cues if c.end > edit.start and c.start < edit.end]
        if not touching:
            log("       (no subtitle cues in this range)")
            continue
        before = [c for c in cues if c.end <= edit.start][-1:]
        after = [c for c in cues if c.start >= edit.end][:1]
        for cue in before:
            log(f"       before {format_time(cue.start)}  {cue.clean[:70]}")
        for cue in touching[:12]:
            marker = "  IN   " if edit.action == "cut" else "  ..   "
            log(f"     {marker}{format_time(cue.start)}  {cue.clean[:70]}")
        if len(touching) > 12:
            log(f"       ... {len(touching) - 12} more cue(s)")
        for cue in after:
            log(f"       after  {format_time(cue.start)}  {cue.clean[:70]}")
