"""Source <-> output time remapping across cuts.

Everything upstream of this module works exclusively on the *source* timeline.
Cuts are applied last, and every timestamp that survives them - mute intervals,
subtitle cues, chapter markers, report entries - is pushed through one shared
``TimeMap``. Because no timing is ever computed against a post-cut timeline,
subtitle drift after a cut is structurally impossible rather than merely
unlikely.
"""

from __future__ import annotations

import bisect
from dataclasses import replace

from .align import Interval
from .subs import Cue

EPSILON = 1e-6


class TimeMap:
    """Piecewise cumulative-offset map defined by a list of removed regions."""

    def __init__(self, cuts: list[tuple[float, float]] | None = None, duration: float = 0.0):
        self.duration = duration
        self.cuts: list[tuple[float, float]] = _normalize(cuts or [])
        self._starts = [c[0] for c in self.cuts]
        self._removed_before: list[float] = []
        running = 0.0
        for start, end in self.cuts:
            self._removed_before.append(running)
            running += end - start
        self.total_removed = running

    # -- basics ----------------------------------------------------------- #

    def __bool__(self) -> bool:
        return bool(self.cuts)

    @property
    def output_duration(self) -> float:
        return max(0.0, self.duration - self.total_removed)

    def is_cut(self, t: float) -> bool:
        index = bisect.bisect_right(self._starts, t) - 1
        if index < 0:
            return False
        start, end = self.cuts[index]
        return start <= t < end

    # -- forward ---------------------------------------------------------- #

    def map(self, t: float) -> float | None:
        """Source time -> output time, or None if the moment was cut away."""
        index = bisect.bisect_right(self._starts, t) - 1
        if index < 0:
            return t
        start, end = self.cuts[index]
        if t < end:
            return None
        return t - (self._removed_before[index] + (end - start))

    def map_clamped(self, t: float, *, prefer: str = "left") -> float:
        """Like ``map`` but snaps a cut moment to the nearest surviving edge.

        Both edges of a cut collapse to the same output time, so ``prefer`` only
        documents intent at the call site.
        """
        del prefer
        index = bisect.bisect_right(self._starts, t) - 1
        if index < 0:
            return t
        start, end = self.cuts[index]
        if t < end:
            return start - self._removed_before[index]
        return t - (self._removed_before[index] + (end - start))

    def unmap(self, t: float) -> float:
        """Output time -> source time (the inverse; always well defined)."""
        source = t
        for start, end in self.cuts:
            if source >= start - EPSILON:
                source += end - start
            else:
                break
        return source

    # -- intervals -------------------------------------------------------- #

    def split(self, start: float, end: float) -> list[tuple[float, float]]:
        """Remove cut regions from a source interval; returns source pieces."""
        pieces = [(start, end)]
        for cut_start, cut_end in self.cuts:
            next_pieces: list[tuple[float, float]] = []
            for piece_start, piece_end in pieces:
                if cut_end <= piece_start or cut_start >= piece_end:
                    next_pieces.append((piece_start, piece_end))
                    continue
                if piece_start < cut_start:
                    next_pieces.append((piece_start, cut_start))
                if piece_end > cut_end:
                    next_pieces.append((cut_end, piece_end))
            pieces = next_pieces
            if not pieces:
                break
        return pieces

    def map_interval(self, start: float, end: float, *, join: bool = True) -> list[tuple[float, float]]:
        """Source interval -> output interval(s).

        Pieces that become adjacent once the material between them is removed
        are rejoined, which is what makes a subtitle cue spanning a cut come out
        as one continuous caption instead of two.
        """
        mapped: list[tuple[float, float]] = []
        for piece_start, piece_end in self.split(start, end):
            out_start = self.map(piece_start)
            out_end = self.map(max(piece_end - EPSILON, piece_start))
            if out_start is None or out_end is None:
                continue
            out_end = max(out_end, out_start)
            if join and mapped and out_start - mapped[-1][1] <= 1e-3:
                mapped[-1] = (mapped[-1][0], max(mapped[-1][1], out_end))
            else:
                mapped.append((out_start, out_end))
        return [(s, e) for s, e in mapped if e - s > EPSILON]

    def map_intervals(self, intervals: list[Interval]) -> list[Interval]:
        out: list[Interval] = []
        for interval in intervals:
            for start, end in self.map_interval(interval.start, interval.end):
                out.append(replace(interval, start=start, end=end))
        return out

    # -- subtitles and chapters ------------------------------------------- #

    def map_cues(self, cues: list[Cue]) -> list[Cue]:
        out: list[Cue] = []
        for cue in cues:
            spans = self.map_interval(cue.start, cue.end)
            for start, end in spans:
                # Sub-40 ms leftovers at a cut boundary are visual noise.
                if end - start < 0.04 and len(spans) > 1:
                    continue
                out.append(replace(cue, start=start, end=max(end, start + 0.04)))
        out.sort(key=lambda c: (c.start, c.end))
        for position, cue in enumerate(out, start=1):
            cue.index = position
        return out

    def map_chapters(self, chapters: list[dict]) -> list[dict]:
        """Rebuild ffmpeg chapter records, dropping any fully inside a cut."""
        out: list[dict] = []
        for chapter in chapters:
            try:
                start = float(chapter.get("start_time"))
                end = float(chapter.get("end_time"))
            except (TypeError, ValueError):
                continue
            spans = self.map_interval(start, end)
            if not spans:
                continue
            new_start = spans[0][0]
            new_end = spans[-1][1]
            if new_end - new_start < 1.0:
                continue
            out.append(
                {
                    "start_time": new_start,
                    "end_time": new_end,
                    "title": (chapter.get("tags") or {}).get("title", ""),
                }
            )
        return out

    def segments(self) -> list[tuple[float, float]]:
        """The surviving source ranges, in order - what a concat filter keeps."""
        if not self.cuts:
            return [(0.0, self.duration)] if self.duration > 0 else []
        kept: list[tuple[float, float]] = []
        cursor = 0.0
        for start, end in self.cuts:
            if start > cursor + EPSILON:
                kept.append((cursor, start))
            cursor = max(cursor, end)
        if self.duration > cursor + EPSILON:
            kept.append((cursor, self.duration))
        elif not self.duration:
            kept.append((cursor, 0.0))
        return [(s, e) for s, e in kept if e > s + EPSILON]


def _normalize(cuts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((max(0.0, s), e) for s, e in cuts if e > s)
    merged: list[tuple[float, float]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1] + EPSILON:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def write_chapters_file(chapters: list[dict], path):
    """Write an ffmetadata chapter file for remuxing after a cut."""
    lines = [";FFMETADATA1"]
    for chapter in chapters:
        lines.append("[CHAPTER]")
        lines.append("TIMEBASE=1/1000")
        lines.append(f"START={int(round(chapter['start_time'] * 1000))}")
        lines.append(f"END={int(round(chapter['end_time'] * 1000))}")
        title = str(chapter.get("title") or "").replace("=", r"\=").replace("\n", " ")
        if title:
            lines.append(f"title={title}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
