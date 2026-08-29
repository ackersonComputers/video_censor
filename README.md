# Censor — Portable Handoff Package

**Read this file completely before running anything.**

This folder is a self-contained, portable copy of the `censor` tool. It is
designed to be copied to another computer and set up by an AI coding assistant
with no additional context. Everything needed is in this folder except the
external programs listed under Prerequisites.

---

## 1. What this tool does

`censor` removes profanity from a video file by **muting the audio** at the
exact moments profane words are spoken.

It finds those moments by combining two sources of truth:

1. **The subtitle track embedded in the video** — this gives the authoritative
   list of what words are actually said and roughly when.
2. **Speech recognition (Whisper) on the audio** — this gives precise
   word-level timestamps.

The two are fused so the mute lands tightly on the word rather than blanking
whole sentences.

---

## 2. THE DEFAULT CASE — the contract

When the user says **"run default case"**, this is exactly what you do:

```powershell
.\run_default.ps1 -Path "C:\path\to\video.mkv"
```

To apply that same mute-only algorithm to every video directly inside a folder:

```powershell
.\run_default.ps1 -Path "C:\path\to\episodes"
```

Each output is written next to its source as `<original name>_censored.mkv`.
Use `-Recursive` to include subfolders, or `-OutputDirectory "D:\Clean"` to
write a mirrored output tree elsewhere.

That is equivalent to:

```powershell
.\.venv\Scripts\python.exe -m censor.cli run "C:\path\to\video.mkv"
```

### What the default case guarantees

| Aspect | Behaviour |
| --- | --- |
| Audio | Profane words muted (volume forced to 0 over each detected interval, with 10 ms edge fades to prevent clicks; the source channel count/layout is preserved) |
| Video | **Copied bit-for-bit. Never re-encoded. Never cut, blurred, or blacked out.** |
| Subtitles | A censored subtitle track is added; words masked with `___` |
| Output | `<original name>_censored.mkv`, written next to the input; folder runs do this for every video |
| Input file | Never modified. It is opened read-only |

### Why the video is guaranteed safe

In `solution/censor/render.py`, video re-encoding is triggered by exactly one
condition: `edl_mod.requires_video_encode(edits)`. The `edits` list is only ever
non-empty when an **edit list** is supplied. With no edit list, the code takes
the `-c:v copy` branch. The picture is passed through untouched.

### ABSOLUTE RULES for the default case

Do **not** add any of these flags. Each one breaks the "no video changes"
guarantee or the "mute only" contract:

- `--edl` / `-e` — supplies a scene edit list, enables cuts/blur/blackout
- the `manual` subcommand — edit-list-only mode, re-encodes video
- `--edl-dir` (in `batch` mode) — same problem
- `--video-encoder` — irrelevant in mute-only mode; setting it signals intent to re-encode

**Do not invent flags. Do not "improve" the command. Do not add scene editing.
If the user asks only for the default case, run the command above verbatim.**

---

## 3. Prerequisites (must be installed on the target machine)

| Requirement | Notes |
| --- | --- |
| **Python 3.10 or newer** | 3.12 recommended. Must be on `PATH` as `python`. |
| **ffmpeg and ffprobe** | Both must be on `PATH`. On Windows, use the FFmpeg **7.x full-shared** build so WhisperX/TorchCodec can load the FFmpeg DLLs. Verify with `ffmpeg -version` and `ffprobe -version`. |
| **Disk space** | Roughly 1.5–3× the size of the source video, free, on the output drive. |
| **NVIDIA GPU (optional)** | Greatly speeds up speech recognition. CPU works but is much slower. |

If `ffmpeg` is missing on Windows, or if the installed build does not include
shared DLLs, install the compatible shared build:

```powershell
winget install --id Gyan.FFmpeg.Shared --version 7.1.1 --exact
```

Then open a new terminal so `PATH` refreshes. The shared build's `bin` folder
must appear somewhere on `PATH` and contain `avcodec-61.dll`; it does not need
to replace another FFmpeg executable that is already installed. The ordinary
`Gyan.FFmpeg` package is a static build: its executables work for rendering,
but TorchCodec cannot load it by itself. The tool registers compatible FFmpeg
shared-library folders from `PATH` with Python's Windows DLL loader before
importing WhisperX. It also discovers a Winget-installed `Gyan.FFmpeg.Shared`
build directly, so an already-open terminal does not need a `PATH` refresh.

---

## 4. Setup

Run this **once**, from inside this folder:

```powershell
.\setup.ps1
```

For a machine with an NVIDIA GPU, use the CUDA 12.8 build of PyTorch instead:

```powershell
.\setup.ps1 -Cuda
```

The CUDA setup pins the mutually compatible stack used by this handoff:
PyTorch 2.8.0 + cu128, torchvision 0.23.0, torchaudio 2.8.0,
WhisperX 3.8.6, and TorchCodec 0.7.0. Do not later run an unpinned
`pip install torch`; on Windows it can replace the CUDA wheel with a CPU wheel.

### CRITICAL setup warnings

1. **Run the installer exactly once. Wait for it to finish.**
   Two `pip install` processes writing to the same environment at the same time
   will deadlock and appear frozen. If you think it has hung, check for multiple
   Python processes before assuming failure:
   ```powershell
   Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' } |
     Select-Object ProcessId, CommandLine | Format-List
   ```
   If more than one `pip install` is running, stop all but one.

2. **Installation is genuinely slow.** PyTorch and the Whisper models are large
   downloads (multiple GB). Several minutes of apparent silence is normal, not a
   hang. Do not interrupt and restart repeatedly.

3. **Quote paths containing `[` or `]` in PowerShell.** Square brackets are
   wildcard characters. Always use `-LiteralPath` with PowerShell cmdlets, and
   always wrap the path in quotes when passing it to the tool. This matters
   because scene-release filenames frequently contain brackets.

4. **The first run downloads the Whisper model** (~3 GB for `large-v3`). This
   happens once and is cached; later runs are much faster.

### Verify the setup

```powershell
.\.venv\Scripts\python.exe -m censor.cli --version
.\.venv\Scripts\python.exe -m censor.cli words
.\.venv\Scripts\python.exe -m pytest solution\tests -q
```

All three should succeed before you process any video.

---

## 5. Recommended workflow for a new file

**Step 1 — dry inspection. Writes no video, costs nothing.**

```powershell
.\run_default.ps1 -Path "C:\path\to\video.mkv" -ReportOnly
```

This performs full detection and prints what *would* be muted, plus a JSON
report. Review it. If the number of detected intervals looks obviously wrong
(zero, or many hundreds), stop and investigate before rendering.

**Step 2 — the real run.**

```powershell
.\run_default.ps1 -Path "C:\path\to\video.mkv"
```

**Step 3 — verify.** Play the output and spot-check two or three of the
timestamps listed in the report.

### Folder run

For a folder you already trust, first inspect every direct video file without
rendering:

```powershell
.\run_default.ps1 -Path "C:\path\to\episodes" -ReportOnly
```

Then run the same command without `-ReportOnly` to render each file. Outputs
remain alongside their inputs and already-censored files are skipped. Add
`-Recursive` to include subfolders, or use `-OutputDirectory "D:\Clean\episodes"`
to keep the results in a separate, mirrored directory tree.

---

## 6. Verified compatibility with the target file type

The intended input is a scene-release episode such as:

```
reacher.s04e01.1080p.web.h264-cakes[EZTVx.to].mkv
```

This file was inspected with `ffprobe` and its structure confirmed compatible:

- **Video:** 1 stream, `h264`. Will be stream-copied, untouched.
- **Audio:** 1 stream, `eac3`, 6 channels, language `eng`. This is the track
  that gets muted.
- **Subtitles:** 38 `subrip` (text) streams. Text-based, so they extract cleanly.
  Two are English:
  - subtitle ordinal `0` — *"American English [Forced]"*, `forced=1`
  - subtitle ordinal `1` — *"American English [SDH]"*, `forced=0`

### Important subtitle detail — already handled

A **Forced** subtitle track only contains foreign-language dialogue, not the
full script. Using it would cause the tool to miss almost all profanity.

`pick_subtitle_stream` in `solution/censor/media.py` scores candidate tracks and
explicitly **deprioritises forced tracks**. For this file it therefore selects
ordinal `1`, the full SDH track. **No flag is needed. The default case is
already correct for this file.**

The tool prints which subtitle track it chose. Confirm that line says the SDH /
non-forced English track. If a differently-authored file ever picks the forced
track, and only then, override it:

```powershell
.\.venv\Scripts\python.exe -m censor.cli run "video.mkv" --sub-stream 1
```

Note that `--sub-stream` takes the **subtitle ordinal** (the `N` in `0:s:N`),
not the global stream index shown by `ffprobe`.

---

## 7. Options you may legitimately use

These do **not** violate the mute-only contract:

| Flag | Purpose |
| --- | --- |
| `--report-only` | Analyse and report; write no video |
| `--dry-run` | Print the ffmpeg command only; change nothing |
| `-o <path>` | Choose the output path |
| `--overwrite` | Replace an existing output file |
| `-y` | Assume yes to prompts (unattended) |
| `--sub-stream N` | Force a subtitle ordinal |
| `--subs <file>` | Use an external subtitle file |
| `--device cuda` / `cpu` | Force the compute device |
| `--compute-type float16` / `float32` | ASR numeric precision. Defaults to fast `float16`; CPU automatically falls back to `float32`. |
| `--model medium` | Smaller, faster, slightly less accurate than `large-v3` |
| `--skip-asr` | Subtitle timing only. Very fast, but coarse and over-mutes |
| `--paranoid` | Mute the whole subtitle cue when timing is uncertain |
| `--add-word W` | Censor an extra word |
| `--allow-word W` | Never censor this word |
| `--tiers a,b` | Choose word list severity tiers |
| `--audio-codec eac3` | See the file-size note below |

### File size note

The default audio codec is `flac`, which is lossless but produces a large file
from a 6-channel source. To keep the output near the original size, and only if
the user asks about size:

```powershell
.\.venv\Scripts\python.exe -m censor.cli run "video.mkv" --audio-codec eac3 --audio-bitrate 448k
```

The video is still copied untouched either way.

---

## 8. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `ffmpeg not found` | ffmpeg/ffprobe are not on `PATH`. Install, then open a new terminal. |
| `Could not load libtorchcodec` | No compatible FFmpeg shared-library folder is visible. Install `Gyan.FFmpeg.Shared` 7.1.1, open a new terminal, and confirm its `bin` folder is present on `PATH` and contains `avcodec-61.dll`. |
| `CUDA available to PyTorch: False` after `-Cuda` | The CUDA torch wheel was replaced or failed to load. Verify `torch==2.8.0+cu128`, `torchvision==0.23.0+cu128`, and `torchaudio==2.8.0+cu128`; then repair those three together from the cu128 PyTorch index. |
| Setup appears frozen | Almost always a large download, or two concurrent `pip install` processes. See section 4. |
| `no subtitles` warning | The file has no usable text subtitle track. Supply one with `--subs`, or accept ASR-only with `--allow-asr-fallback`. |
| Zero words detected | Wrong subtitle track picked, or wrong audio track. Re-run with `--report-only` and check the reported subtitle choice. |
| Very slow | Running on CPU. Install the CUDA build of PyTorch (`.\setup.ps1 -Cuda`) or use `--model medium`. |
| Out of GPU memory | `--batch-size 4`, or `--device cpu`. |
| Clicks at mute boundaries | Current builds use sample-accurate 10 ms fades around every fully silent interval. Re-render an older output to apply the fix. |
| Lightning checkpoint / TF32 warning | Informational. The checkpoint is upgraded in memory and TF32 stays disabled for reproducibility; no action is required. |
| Mutes land slightly early or late | Subtitle offset drift. Try `--sub-offset <seconds>`, or `--paranoid` for safety. |
| Output file already exists | Add `--overwrite`. |

---

## 9. Folder contents

```
censor-handoff/
  README.md          <- this file
  setup.ps1          <- one-time environment setup
  run_default.ps1    <- mute-only run for one file or a whole folder
  requirements.txt   <- runtime dependencies
  solution/
    pyproject.toml
    README.md          <- upstream project documentation
    TESTING_GUIDE.md   <- upstream testing notes
    censor/            <- the Python package
    tests/             <- unit tests
```

---

## 10. Instructions to the AI assistant on the target machine

- Do not rewrite, refactor, or "improve" the code in `solution/censor/`. It is
  complete and tested. Your job is environment setup and correct invocation.
- Do not create a second virtual environment. Use `.\.venv` created by `setup.ps1`.
- Do not run `pip install` concurrently with itself.
- Always quote file paths. Filenames contain spaces and square brackets.
- Never write output over the source video. The tool defaults to a `_censored`
  suffix; keep it that way.
- For "run default case", run the mute-only command in section 2 and nothing else.
  It accepts either one video file or a directory of video files.
- Before any full render on a new file, run `--report-only` first and show the
  user the result.
