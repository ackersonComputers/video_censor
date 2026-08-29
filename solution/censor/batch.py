"""Folder driver: process every video under a directory."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import Settings

VIDEO_EXTS = {
    ".mkv",
    ".mp4",
    ".m4v",
    ".mov",
    ".avi",
    ".wmv",
    ".flv",
    ".webm",
    ".ts",
    ".m2ts",
    ".mpg",
    ".mpeg",
}


def iter_videos(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    found = [
        path
        for path in sorted(root.glob(pattern))
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS
    ]
    # Never re-process our own output.
    return [p for p in found if ".censor" not in p.parts and not p.stem.endswith("_censored")]


def run_batch(args: argparse.Namespace, base: Settings) -> int:
    from . import pipeline  # noqa: PLC0415

    root: Path = args.input_dir
    if not root.is_dir():
        print(f"error: not a directory: {root}")
        return 2

    videos = iter_videos(root, args.recursive)
    if not videos:
        print(f"No video files found under {root}")
        return 0

    print(f"[batch] {len(videos)} file(s) under {root}")
    ok = skipped = failed = 0
    failures: list[tuple[Path, str]] = []

    for position, video in enumerate(videos, start=1):
        relative = video.relative_to(root)
        output = (args.output_dir / relative).with_name(
            f"{video.stem}{args.suffix}.mkv"
        )
        edl = base.edl
        if args.edl_dir:
            candidate = args.edl_dir / f"{video.stem}.edl.csv"
            edl = candidate if candidate.is_file() else None

        print("")
        print("=" * 66)
        print(f"[batch] {position}/{len(videos)}  {relative}")
        print("=" * 66)

        settings = replace(
            base,
            input=video,
            output=output,
            report=None,
            edl=edl,
            edl_out=None,
            work_dir=None,
            # Unattended runs must never block on a prompt.
            assume_yes=True,
            allow_asr_fallback=True,
        )
        try:
            result = pipeline.process(settings)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
            failed += 1
            failures.append((video, str(exc)))
            print(f"[batch] ! failed: {exc}")
            if not args.continue_on_error:
                break
            continue
        if result.skipped:
            skipped += 1
            print(f"[batch] skipped: {result.skipped}")
        else:
            ok += 1

    print("")
    print(f"[batch] done: {ok} processed, {skipped} skipped, {failed} failed")
    for video, message in failures:
        print(f"  ! {video.name}: {message}")
    return 1 if failed else 0
