"""Scene edit list: hand-authored intervals with an action each.

CSV (hand editing) and JSON (canonical / generated) are both accepted.

    # start, end, action, note, params
    00:42:10,500, 00:43:58,000, cut,      pool scene
    1:12:04.0,    1:13:20.5,    blackout, plot-critical, fade=0.5
    1:31:00,      1:31:40,      blur,     ,              strength=40
    18:22,        18:26,        mute,     slur the model missed

Times accept ``SS.mmm``, ``MM:SS.mmm``, ``HH:MM:SS.mmm`` and SRT-style
``HH:MM:SS,mmm`` (so timestamps pasted straight out of a subtitle file work).
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ACTIONS = ("cut", "blackout", "blur", "mute")

_CLOCK_RE = re.compile(
    r"^\s*(?:(?:(?P<h>\d+):)?(?P<m>\d{1,2}):)?(?P<s>\d{1,2}(?:[.,]\d+)?)\s*$"
)


class EDLError(RuntimeError):
    """Raised when an edit list is malformed."""


def parse_time(text: str) -> float:
    """Parse a timestamp in any supported form into seconds."""
    raw = str(text).strip()
    if not raw:
        raise EDLError("Empty timestamp")
    # SRT style uses a comma as the decimal separator: 00:01:02,500
    if raw.count(",") == 1 and ":" in raw:
        raw = raw.replace(",", ".")
    match = _CLOCK_RE.match(raw)
    if not match:
        raise EDLError(f"Unrecognised timestamp: {text!r}")
    hours = int(match.group("h") or 0)
    minutes = int(match.group("m") or 0)
    seconds = float(match.group("s").replace(",", "."))
    return hours * 3600 + minutes * 60 + seconds


def format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"


@dataclass
class Edit:
    """One scene action on the source timeline."""

    start: float
    end: float
    action: str
    note: str = ""
    params: dict[str, str] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def param_float(self, key: str, default: float) -> float:
        raw = self.params.get(key)
        if raw is None:
            return default
        try:
            return float(raw)
        except (TypeError, ValueError) as exc:
            raise EDLError(
                f"{self.action} at {format_time(self.start)}: "
                f"parameter {key}={raw!r} is not a number"
            ) from exc

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "start_hms": format_time(self.start),
            "end_hms": format_time(self.end),
            "action": self.action,
            "note": self.note,
            "params": self.params,
        }


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load(path: Path) -> list[Edit]:
    if not path.is_file():
        raise EDLError(f"Edit list not found: {path}")
    text = path.read_text(encoding="utf-8-sig")
    stripped = text.lstrip()
    if path.suffix.lower() == ".json" or stripped.startswith(("[", "{")):
        return _load_json(text, path)
    return _load_csv(text, path)


def _load_json(text: str, path: Path) -> list[Edit]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EDLError(f"{path}: invalid JSON ({exc})") from exc
    records = data.get("edits") if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise EDLError(f"{path}: expected a list of edits or {{'edits': [...]}}")

    edits: list[Edit] = []
    for position, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise EDLError(f"{path}: entry {position} is not an object")
        try:
            edits.append(
                Edit(
                    start=_coerce_time(record.get("start")),
                    end=_coerce_time(record.get("end")),
                    action=_check_action(record.get("action"), path, position),
                    note=str(record.get("note") or ""),
                    params={str(k): str(v) for k, v in (record.get("params") or {}).items()},
                )
            )
        except EDLError as exc:
            raise EDLError(f"{path}: entry {position}: {exc}") from exc
    return edits


def _coerce_time(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        raise EDLError("missing start/end")
    return parse_time(str(value))


def _load_csv(text: str, path: Path) -> list[Edit]:
    edits: list[Edit] = []
    lines = [
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    for position, row in enumerate(csv.reader(lines), start=1):
        cells = [cell.strip() for cell in row]
        while cells and cells[-1] == "":
            cells.pop()
        if len(cells) < 3:
            raise EDLError(
                f"{path}: row {position} needs at least start,end,action -> got {row!r}"
            )
        if position == 1 and cells[0].lower() in {"start", "start_time", "begin"}:
            continue  # header
        try:
            start = parse_time(cells[0])
            end = parse_time(cells[1])
        except EDLError as exc:
            raise EDLError(f"{path}: row {position}: {exc}") from exc
        action = _check_action(cells[2], path, position)
        note = cells[3] if len(cells) > 3 else ""
        params = _parse_params(cells[4]) if len(cells) > 4 else {}
        edits.append(Edit(start=start, end=end, action=action, note=note, params=params))
    return edits


def _check_action(value, path, position) -> str:
    action = str(value or "").strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "cut": "cut",
        "remove": "cut",
        "skip": "cut",
        "delete": "cut",
        "blackout": "blackout",
        "black": "blackout",
        "blank": "blackout",
        "blur": "blur",
        "pixelate": "blur",
        "mute": "mute",
        "silence": "mute",
    }
    resolved = aliases.get(action)
    if resolved is None:
        raise EDLError(
            f"{path}: row {position}: unknown action {value!r}; expected one of {', '.join(ACTIONS)}"
        )
    return resolved


def _parse_params(text: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for chunk in re.split(r"[;\s]+", text.strip()):
        if not chunk:
            continue
        key, _, value = chunk.partition("=")
        if not value:
            raise EDLError(f"parameter {chunk!r} must be key=value")
        params[key.strip().lower()] = value.strip()
    return params


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate(
    edits: list[Edit],
    duration: float = 0.0,
    *,
    log=print,
) -> list[Edit]:
    """Sort, clamp and sanity-check an edit list. Raises on unrecoverable errors."""
    checked: list[Edit] = []
    for edit in edits:
        if edit.end <= edit.start:
            raise EDLError(
                f"{edit.action} at {format_time(edit.start)}: end must be after start"
            )
        start, end = edit.start, edit.end
        if duration > 0:
            if start >= duration:
                log(
                    f"[edl] dropping {edit.action} at {format_time(start)}: "
                    f"beyond end of file ({format_time(duration)})"
                )
                continue
            end = min(end, duration)
        checked.append(Edit(start=start, end=end, action=edit.action, note=edit.note, params=dict(edit.params)))

    checked.sort(key=lambda e: (e.start, e.end))

    previous_cut: Edit | None = None
    for edit in checked:
        if edit.action != "cut":
            continue
        if previous_cut and edit.start < previous_cut.end:
            raise EDLError(
                f"cut intervals overlap: {format_time(previous_cut.start)}-"
                f"{format_time(previous_cut.end)} and {format_time(edit.start)}-"
                f"{format_time(edit.end)}. Merge them into one row."
            )
        previous_cut = edit

    for edit in checked:
        if edit.action == "blur":
            edit.param_float("strength", 30.0)
        elif edit.action == "blackout":
            edit.param_float("fade", 0.0)

    return checked


def cuts(edits: list[Edit]) -> list[tuple[float, float]]:
    return [(e.start, e.end) for e in edits if e.action == "cut"]


def of_action(edits: list[Edit], action: str) -> list[Edit]:
    return [e for e in edits if e.action == action]


def total_cut_time(edits: list[Edit]) -> float:
    return sum(e.duration for e in edits if e.action == "cut")


def requires_video_encode(edits: list[Edit]) -> bool:
    return any(e.action in {"cut", "blackout", "blur"} for e in edits)


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #


def save_json(edits: list[Edit], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "edits": [e.to_dict() for e in edits]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def save_csv(edits: list[Edit], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# start,end,action,note,params"]
    for edit in edits:
        params = " ".join(f"{k}={v}" for k, v in edit.params.items())
        lines.append(
            f"{format_time(edit.start)},{format_time(edit.end)},"
            f"{edit.action},{edit.note},{params}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def template(path: Path) -> Path:
    """Write a commented starter edit list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Scene edit list for `censor`.\n"
        "# Columns: start, end, action, note, params\n"
        "# Times: SS.mmm | MM:SS.mmm | HH:MM:SS.mmm | HH:MM:SS,mmm (SRT paste)\n"
        "#\n"
        "# Actions:\n"
        "#   cut       remove the interval entirely (video, audio and subtitles);\n"
        "#             everything after it shifts earlier, and subtitles follow.\n"
        "#   blackout  video goes black, audio and subtitles keep playing.\n"
        "#             params: fade=<seconds> for a soft in/out.\n"
        "#   blur      heavy blur over the interval. params: strength=<sigma>,\n"
        "#             mode=gblur|pixelate\n"
        "#   mute      silence the audio, picture untouched.\n"
        "#\n"
        "# Note: any cut/blackout/blur forces a video re-encode. A file with only\n"
        "# mute rows keeps the original video stream untouched.\n"
        "#\n"
        "# 00:42:10.500,00:43:58.000,cut,pool scene\n"
        "# 01:12:04.000,01:13:20.500,blackout,plot-critical,fade=0.5\n"
        "# 01:31:00.000,01:31:40.000,blur,,strength=40\n",
        encoding="utf-8",
    )
    return path
