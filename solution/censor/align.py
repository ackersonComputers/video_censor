"""The core: fuse subtitle text (what was said) with ASR timings (when).

The subtitle track is treated as ground truth for *content* and the aligned ASR
output as ground truth for *timing*. Every profanity found in the subtitles is
resolved to a mute interval through a five-level ladder; when timing cannot be
pinned down the interval widens rather than disappearing, because too much
silence beats a missed swear.
"""

from __future__ import annotations

import bisect
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace as dc_replace

from .asr import Word
from .config import Settings
from .matcher import Matcher, Match, Token, tokenize, strip_apostrophes
from .subs import Cue

LEVEL_NAMES = {
    "L1": "exact ASR word match",
    "L2": "fuzzy ASR match",
    "L3": "interpolated between aligned anchors",
    "L4": "proportional slice of subtitle cue",
    "L5": "whole subtitle cue",
    "ASR": "ASR-only (not in subtitles)",
}


@dataclass
class Interval:
    """A mute interval on the source timeline, with full provenance."""

    start: float
    end: float
    level: str
    token: str
    tier: str
    confidence: float
    source: str = "subtitle"
    cue_index: int | None = None
    cue_text: str = ""
    note: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "level": self.level,
            "level_meaning": LEVEL_NAMES.get(self.level, ""),
            "token": self.token,
            "tier": self.tier,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "cue_index": self.cue_index,
            "cue_text": self.cue_text,
            "note": self.note,
        }


@dataclass
class AlignStats:
    subtitle_hits: int = 0
    asr_only_hits: int = 0
    levels: Counter = field(default_factory=Counter)
    tiers: Counter = field(default_factory=Counter)
    offset_seconds: float = 0.0
    offset_confidence: float = 0.0
    drift_anchors: int = 0
    extras_dropped: int = 0
    merged_from: int = 0

    def to_dict(self) -> dict:
        return {
            "subtitle_hits": self.subtitle_hits,
            "asr_only_hits": self.asr_only_hits,
            "levels": dict(self.levels),
            "tiers": dict(self.tiers),
            "offset_seconds": round(self.offset_seconds, 3),
            "offset_confidence": round(self.offset_confidence, 3),
            "drift_anchors": self.drift_anchors,
            "extras_dropped": self.extras_dropped,
            "merged_from": self.merged_from,
        }


# --------------------------------------------------------------------------- #
# Subtitle <-> ASR timing model
# --------------------------------------------------------------------------- #


class TimingModel:
    """Maps subtitle time to ASR/audio time, handling constant offset and drift.

    Home-library subtitle tracks are routinely shifted by a second or two, and
    23.976 <-> 25 fps conversions introduce linear drift that grows to many
    seconds by the end of a feature. Both break naive cue-window matching, so
    they are measured and corrected before any word is compared.
    """

    def __init__(self, anchors: list[tuple[float, float]] | None = None, offset: float = 0.0):
        self.offset = offset
        self._times: list[float] = []
        self._offsets: list[float] = []
        if anchors:
            anchors = sorted(anchors)
            self._times = [a for a, _ in anchors]
            self._offsets = [b - a for a, b in anchors]

    def to_asr(self, sub_time: float) -> float:
        if not self._times:
            return sub_time + self.offset
        position = bisect.bisect_left(self._times, sub_time)
        if position == 0:
            return sub_time + self._offsets[0]
        if position >= len(self._times):
            return sub_time + self._offsets[-1]
        t0, t1 = self._times[position - 1], self._times[position]
        o0, o1 = self._offsets[position - 1], self._offsets[position]
        if t1 <= t0:
            return sub_time + o1
        ratio = (sub_time - t0) / (t1 - t0)
        return sub_time + o0 + ratio * (o1 - o0)

    def offset_at(self, sub_time: float) -> float:
        return self.to_asr(sub_time) - sub_time


def _cue_word_times(cue: Cue, tokens: list[Token]) -> list[float]:
    """Approximate each token's time by its character position within the cue."""
    if not tokens:
        return []
    span = max(cue.end - cue.start, 0.001)
    text_len = max(tokens[-1].end, 1)
    return [cue.start + span * (t.start + (t.end - t.start) / 2) / text_len for t in tokens]


def _rare_word_anchors(
    cues: list[Cue],
    words: list[Word],
    max_shift: float,
) -> list[tuple[float, float]]:
    """Pair distinctive words that appear in both streams."""
    sub_points: dict[str, list[float]] = defaultdict(list)
    for cue in cues:
        tokens = tokenize(cue.clean)
        for token, when in zip(tokens, _cue_word_times(cue, tokens)):
            key = strip_apostrophes(token.norm)
            if len(key) >= 5:
                sub_points[key].append(when)

    asr_points: dict[str, list[float]] = defaultdict(list)
    for word in words:
        key = strip_apostrophes(word.norm)
        if len(key) >= 5:
            asr_points[key].append(word.mid)

    pairs: list[tuple[float, float]] = []
    for key, sub_times in sub_points.items():
        asr_times = asr_points.get(key)
        if not asr_times or len(sub_times) > 4 or len(asr_times) > 4:
            continue
        for sub_time in sub_times:
            best = min(asr_times, key=lambda a: abs(a - sub_time))
            if abs(best - sub_time) <= max_shift:
                pairs.append((sub_time, best))
    pairs.sort()
    return pairs


def build_timing_model(
    cues: list[Cue],
    words: list[Word],
    *,
    forced_offset: float | None = None,
    auto: bool = True,
    max_shift: float = 60.0,
    stats: AlignStats | None = None,
    log=print,
) -> TimingModel:
    if forced_offset is not None:
        if stats:
            stats.offset_seconds = forced_offset
            stats.offset_confidence = 1.0
        return TimingModel(offset=forced_offset)
    if not auto or not cues or not words:
        return TimingModel()

    pairs = _rare_word_anchors(cues, words, max_shift)
    if len(pairs) < 20:
        log(f"[align] only {len(pairs)} sync anchors found; assuming zero offset")
        return TimingModel()

    # Vote on a global offset in 100 ms bins, then keep anchors that agree.
    votes = Counter(round((asr - sub) * 10) for sub, asr in pairs)
    best_bin, best_count = votes.most_common(1)[0]
    # Widen to the winning bin plus its neighbours.
    window = {best_bin - 1, best_bin, best_bin + 1}
    agreeing = [
        (sub, asr) for sub, asr in pairs if round((asr - sub) * 10) in window
    ]
    global_offset = _median([asr - sub for sub, asr in agreeing]) if agreeing else 0.0
    confidence = best_count / len(pairs)

    # Keep anchors within 1.5 s of the global offset, then bin them over time to
    # capture drift as a piecewise-linear correction.
    kept = [(sub, asr) for sub, asr in pairs if abs((asr - sub) - global_offset) <= 1.5]
    if stats:
        stats.offset_seconds = global_offset
        stats.offset_confidence = confidence
        stats.drift_anchors = len(kept)

    if len(kept) < 20:
        log(f"[align] subtitle offset {global_offset:+.3f}s (confidence {confidence:.0%})")
        return TimingModel(offset=global_offset)

    bins: dict[int, list[tuple[float, float]]] = defaultdict(list)
    bin_size = 120.0
    for sub, asr in kept:
        bins[int(sub // bin_size)].append((sub, asr))

    anchors: list[tuple[float, float]] = []
    for key in sorted(bins):
        group = bins[key]
        if len(group) < 4:
            continue
        centre = _median([s for s, _ in group])
        shift = _median([a - s for s, a in group])
        anchors.append((centre, centre + shift))

    if len(anchors) < 2:
        log(f"[align] subtitle offset {global_offset:+.3f}s (confidence {confidence:.0%})")
        return TimingModel(offset=global_offset)

    drift = (anchors[-1][1] - anchors[-1][0]) - (anchors[0][1] - anchors[0][0])
    log(
        f"[align] subtitle offset {global_offset:+.3f}s "
        f"(confidence {confidence:.0%}, {len(kept)} anchors, drift {drift:+.2f}s)"
    )
    return TimingModel(anchors=anchors, offset=global_offset)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


# --------------------------------------------------------------------------- #
# Resolution ladder
# --------------------------------------------------------------------------- #


def _similarity(a: str, b: str) -> float:
    try:
        from rapidfuzz.distance import JaroWinkler  # noqa: PLC0415

        return JaroWinkler.similarity(a, b)
    except ImportError:
        import difflib  # noqa: PLC0415

        return difflib.SequenceMatcher(None, a, b).ratio()


def resolve(
    cues: list[Cue],
    words: list[Word],
    matcher: Matcher,
    settings: Settings,
    *,
    log=print,
) -> tuple[list[Interval], AlignStats]:
    """Turn subtitle profanity plus ASR timings into merged mute intervals."""
    stats = AlignStats()
    model = build_timing_model(
        cues,
        words,
        forced_offset=settings.sub_offset,
        auto=not settings.no_auto_offset,
        stats=stats,
        log=log,
    )

    starts = [w.start for w in words]
    used: set[int] = set()
    intervals: list[Interval] = []

    for cue in cues:
        clean = cue.clean
        if not clean:
            continue
        tokens = tokenize(clean)
        matches = matcher.find_in_tokens(tokens)
        if not matches:
            continue
        window_start = model.to_asr(cue.start) - settings.window_slack
        window_end = model.to_asr(cue.end) + settings.window_slack
        candidates = _words_in_window(words, starts, window_start, window_end)
        anchors = _cue_anchors(tokens, candidates, matcher)

        for match in matches:
            stats.subtitle_hits += 1
            stats.tiers[match.tier] += 1
            interval = _resolve_match(
                match=match,
                cue=cue,
                tokens=tokens,
                candidates=candidates,
                anchors=anchors,
                used=used,
                settings=settings,
                model=model,
            )
            stats.levels[interval.level] += 1
            intervals.append(interval)

    intervals.extend(
        _asr_only_intervals(words, matcher, settings, intervals, stats, log=log)
    )

    merged = merge_intervals(
        intervals,
        gap=settings.merge_gap,
        min_duration=settings.min_duration,
        max_duration=settings.max_interval,
    )
    stats.merged_from = len(intervals)
    return merged, stats


def _words_in_window(
    words: list[Word], starts: list[float], lo: float, hi: float
) -> list[tuple[int, Word]]:
    left = bisect.bisect_left(starts, lo - 1.0)
    right = bisect.bisect_right(starts, hi)
    return [(i, words[i]) for i in range(left, min(right + 1, len(words))) if words[i].end >= lo]


def _cue_anchors(
    tokens: list[Token],
    candidates: list[tuple[int, Word]],
    matcher: Matcher,
) -> dict[int, Word]:
    """Map cue token index -> ASR word, for non-profane words that match exactly.

    These anchors are what make level 3 possible: even when the swear itself was
    misheard over music, the words around it usually were not.
    """
    anchors: dict[int, Word] = {}
    cursor = 0
    for token in tokens:
        key = strip_apostrophes(token.norm)
        if len(key) < 3 or matcher.is_target(token.norm):
            continue
        for position in range(cursor, len(candidates)):
            _, word = candidates[position]
            if strip_apostrophes(word.norm) == key:
                anchors[token.index] = word
                cursor = position + 1
                break
    return anchors


def _resolve_match(
    *,
    match: Match,
    cue: Cue,
    tokens: list[Token],
    candidates: list[tuple[int, Word]],
    anchors: dict[int, Word],
    used: set[int],
    settings: Settings,
    model: TimingModel,
) -> Interval:
    base = dict(
        token=match.norm,
        tier=match.tier,
        source="subtitle",
        cue_index=cue.index,
        cue_text=cue.clean,
    )
    target_norms = [strip_apostrophes(t.norm) for t in match.tokens]

    # --- L1: an ASR word in the window is literally this word ---------------- #
    span = _find_sequence(candidates, target_norms, used, exact=True)
    if span is not None:
        start, end, indices = span
        used.update(indices)
        pad_lo, pad_hi = settings.pad_exact
        return Interval(
            start=start - pad_lo, end=end + pad_hi, level="L1", confidence=1.0, **base
        )

    # --- L2: a near-miss (misheard over music, or a variant spelling) -------- #
    span = _find_sequence(
        candidates, target_norms, used, exact=False, threshold=settings.fuzzy_threshold
    )
    if span is not None:
        start, end, indices = span
        used.update(indices)
        pad_lo, pad_hi = settings.pad_fuzzy
        return Interval(
            start=start - pad_lo, end=end + pad_hi, level="L2", confidence=0.75, **base
        )

    # --- L3: interpolate from surrounding words that DID align --------------- #
    left = max(
        (i for i in anchors if i < match.first_index),
        default=None,
    )
    right = min(
        (i for i in anchors if i > match.last_index),
        default=None,
    )
    if left is not None or right is not None:
        estimate = _interpolate(
            match, tokens, anchors, left, right, cue, model, settings
        )
        if estimate is not None:
            start, end, note = estimate
            pad_lo, pad_hi = settings.pad_anchor
            return Interval(
                start=start - pad_lo,
                end=end + pad_hi,
                level="L3",
                confidence=0.55,
                note=note,
                **base,
            )

    # --- L4/L5: nothing aligned; fall back to the cue window ----------------- #
    cue_start = model.to_asr(cue.start)
    cue_end = model.to_asr(cue.end)
    if settings.paranoid or cue_end - cue_start <= 0.35 or not tokens:
        return Interval(
            start=cue_start - settings.blind_pad,
            end=cue_end + settings.blind_pad,
            level="L5",
            confidence=0.2,
            note="whole cue muted",
            **base,
        )

    total_chars = max(tokens[-1].end, 1)
    span_seconds = cue_end - cue_start
    slice_start = cue_start + span_seconds * (match.start_char / total_chars)
    slice_end = cue_start + span_seconds * (match.end_char / total_chars)
    return Interval(
        start=slice_start - settings.blind_pad,
        end=slice_end + settings.blind_pad,
        level="L4",
        confidence=0.35,
        note="proportional slice of cue",
        **base,
    )


def _find_sequence(
    candidates: list[tuple[int, Word]],
    target_norms: list[str],
    used: set[int],
    *,
    exact: bool,
    threshold: float = 0.8,
) -> tuple[float, float, list[int]] | None:
    """Find consecutive ASR words matching the target token sequence."""
    length = len(target_norms)
    best: tuple[float, float, float, list[int]] | None = None

    for position in range(len(candidates) - length + 1):
        window = candidates[position : position + length]
        if any(index in used for index, _ in window):
            continue
        score = 0.0
        ok = True
        for (_, word), target in zip(window, target_norms):
            observed = strip_apostrophes(word.norm)
            if not observed:
                ok = False
                break
            if observed == target or (len(target) >= 4 and target in observed):
                score += 1.0
                continue
            if exact:
                ok = False
                break
            similarity = _similarity(observed, target)
            if similarity < threshold:
                ok = False
                break
            score += similarity
        if not ok:
            continue
        score /= length
        if best is None or score > best[0]:
            best = (
                score,
                window[0][1].start,
                window[-1][1].end,
                [index for index, _ in window],
            )
    if best is None:
        return None
    return best[1], best[2], best[3]


def _interpolate(
    match: Match,
    tokens: list[Token],
    anchors: dict[int, Word],
    left: int | None,
    right: int | None,
    cue: Cue,
    model: TimingModel,
    settings: Settings,
) -> tuple[float, float, str] | None:
    """Place the swear proportionally between the nearest aligned neighbours."""
    total = len(tokens)
    if total == 0:
        return None

    if left is not None and right is not None:
        left_time = anchors[left].end
        right_time = anchors[right].start
        if right_time <= left_time:
            return None
        gap_tokens = right - left
        if gap_tokens <= 0:
            return None
        span = right_time - left_time
        start_ratio = (match.first_index - left) / gap_tokens
        end_ratio = (match.last_index + 1 - left) / gap_tokens
        return (
            left_time + span * start_ratio,
            left_time + span * end_ratio,
            f"between anchors '{tokens[left].text}' and '{tokens[right].text}'",
        )

    # Only one side aligned: step outward using the cue's average word rate.
    cue_span = max(model.to_asr(cue.end) - model.to_asr(cue.start), 0.001)
    rate = cue_span / total
    if left is not None:
        base = anchors[left].end
        offset = (match.first_index - left) * rate
        start = base + max(0.0, offset - rate * 0.5)
        return start, start + rate * len(match.tokens), f"after anchor '{tokens[left].text}'"

    base = anchors[right].start  # type: ignore[index]
    offset = (right - match.last_index) * rate  # type: ignore[operator]
    end = base - max(0.0, offset - rate * 0.5)
    return end - rate * len(match.tokens), end, f"before anchor '{tokens[right].text}'"  # type: ignore[index]


def _asr_only_intervals(
    words: list[Word],
    matcher: Matcher,
    settings: Settings,
    existing: list[Interval],
    stats: AlignStats,
    *,
    log=print,
) -> list[Interval]:
    """Catch profanity the subtitles missed.

    Subtitle tracks are frequently abridged or softened, so ASR-only detections
    are muted by default; a whole-word match plus a confidence floor keeps the
    false-positive rate down.
    """
    if settings.extras_policy == "ignore":
        return []

    covered = sorted((i.start, i.end) for i in existing)
    starts = [c[0] for c in covered]
    out: list[Interval] = []
    pad_lo, pad_hi = settings.pad_fuzzy

    for word in words:
        hit = matcher.match_token(word.norm)
        if hit is None:
            continue
        if _covered(covered, starts, word.start, word.end):
            continue
        stats.asr_only_hits += 1
        if word.score < settings.extras_min_score or settings.extras_policy == "report":
            stats.extras_dropped += 1
            continue
        tier, _, key = hit
        out.append(
            Interval(
                start=word.start - pad_lo,
                end=word.end + pad_hi,
                level="ASR",
                token=word.norm,
                tier=tier,
                confidence=float(word.score),
                source="asr",
                note=f"not present in subtitles (rule '{key}')",
            )
        )
    if out:
        log(f"[align] {len(out)} target(s) heard but absent from the subtitles")
    return out


def _covered(covered: list[tuple[float, float]], starts: list[float], lo: float, hi: float) -> bool:
    position = bisect.bisect_right(starts, hi)
    for index in range(max(0, position - 8), position):
        c_start, c_end = covered[index]
        if c_end >= lo and c_start <= hi:
            return True
    return False


# --------------------------------------------------------------------------- #
# Merging
# --------------------------------------------------------------------------- #


def merge_intervals(
    intervals: list[Interval],
    *,
    gap: float = 0.08,
    min_duration: float = 0.1,
    max_duration: float = 6.0,
    limit: float | None = None,
) -> list[Interval]:
    """Clamp, cap and merge overlapping intervals, keeping the best provenance."""
    prepared: list[Interval] = []
    for interval in intervals:
        start = max(0.0, interval.start)
        end = interval.end
        if limit is not None:
            end = min(end, limit)
            start = min(start, limit)
        if end <= start:
            continue
        if end - start < min_duration:
            pad = (min_duration - (end - start)) / 2.0
            start = max(0.0, start - pad)
            end = end + pad
        if end - start > max_duration:
            # A runaway interval usually means a bad cue; keep it centred.
            centre = (start + end) / 2.0
            start, end = centre - max_duration / 2.0, centre + max_duration / 2.0
        prepared.append(
            Interval(
                start=start,
                end=end,
                level=interval.level,
                token=interval.token,
                tier=interval.tier,
                confidence=interval.confidence,
                source=interval.source,
                cue_index=interval.cue_index,
                cue_text=interval.cue_text,
                note=interval.note,
            )
        )

    prepared.sort(key=lambda i: (i.start, i.end))
    merged: list[Interval] = []
    for interval in prepared:
        if merged and interval.start <= merged[-1].end + gap:
            previous = merged[-1]
            previous.end = max(previous.end, interval.end)
            if interval.token not in previous.token.split(" + "):
                previous.token = f"{previous.token} + {interval.token}"
            previous.confidence = min(previous.confidence, interval.confidence)
            if _level_rank(interval.level) > _level_rank(previous.level):
                previous.level = interval.level
        else:
            merged.append(interval)
    return merged


def _level_rank(level: str) -> int:
    return {"L1": 1, "L2": 2, "ASR": 2, "L3": 3, "L4": 4, "L5": 5}.get(level, 3)


def subtract(intervals: list[Interval], removed: list[tuple[float, float]]) -> list[Interval]:
    """Drop or trim intervals that fall inside removed (cut) regions."""
    if not removed:
        return intervals
    out: list[Interval] = []
    for interval in intervals:
        pieces = [(interval.start, interval.end)]
        for cut_start, cut_end in removed:
            next_pieces: list[tuple[float, float]] = []
            for start, end in pieces:
                if cut_end <= start or cut_start >= end:
                    next_pieces.append((start, end))
                    continue
                if start < cut_start:
                    next_pieces.append((start, cut_start))
                if end > cut_end:
                    next_pieces.append((cut_end, end))
            pieces = next_pieces
        for start, end in pieces:
            if end - start <= 0.02:
                continue
            out.append(dc_replace(interval, start=start, end=end))
    return out
