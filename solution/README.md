# censor

Mutes profanity in video files by **combining the file's own subtitle track with
forced-aligned speech recognition**, and applies a hand-authored **scene edit
list** (cuts, blackouts, blurs) in the same pass.

The subtitle track is treated as the truth about *what was said*; the ASR pass is
treated as the truth about *when*. That split is the whole idea: a model can
mishear "fuck" over a loud music cue, but the subtitle file already knows the
word is there, so the only remaining question is where to put the silence.

## Table of contents

- [Why this exists](#why-this-exists)
- [Testing guide for high-power PC](TESTING_GUIDE.md)
- [Install](#install)
- [Quick start](#quick-start)
- [How the timing works](#how-the-timing-works)
- [Scene edit lists](#scene-edit-lists)
- [Word lists](#word-lists)
- [Output layout](#output-layout)
- [Command reference](#command-reference)
- [Tuning](#tuning)
- [Tests](#tests)
- [Project layout](#project-layout)

## Why this exists

This solution was built to avoid common failure modes in audio-only profanity
pipelines, especially on long episodes and music-heavy scenes:

| Old failure | Cause | Fix here |
| --- | --- | --- |
| Worked, then quietly stopped muting for long stretches | Several hundred intervals in a single `enable='between(t,a,b)+...'` expression | The expression is split across chained `volume` filters, 60 intervals each |
| Swears missed over music | Whisper mishears, or hallucinates | Subtitles decide *what*; ASR only decides *when* |
| Generated subtitles full of wrong words | ASR transcript used as the caption source | The real subtitle track is edited in place; only the swear becomes `___` |
| Timings drifted late in long files | Whisper's own word timestamps | WhisperX wav2vec2 forced alignment |
| Muted the wrong audio track | Transcribed one stream, muted another | One explicit stream choice used for both; every audio track is muted by default |
| Subtitles desynced after manual edits | Times computed on a post-edit timeline | All analysis happens on the source timeline; cuts are applied last through one `TimeMap` |

## Install

Python 3.10+ and `ffmpeg`/`ffprobe` on `PATH`.

```powershell
pip install -e .

# GPU stack for the recommended ASR engine (matches a 4090)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install whisperx
```

WhisperX is optional. Without it the tool falls back to `faster-whisper`, then
`openai-whisper`, then to subtitle cue timing alone.

## Quick start

```powershell
# Use the file's own English subtitles for accuracy
censor run "E:\Television\Reacher\S01E01.mkv"

# See what would be muted, and how confident each hit is, without encoding
censor run "S01E01.mkv" --report-only

# Add scene edits, and check their boundaries against the dialogue first
censor run "Movie.mkv" --edl movie.edl.csv --edl-preview

# A whole folder, unattended
censor batch "E:\Television\Show" -o "E:\Clean\Show" --recursive --yes

# Apply an edit list only, no detection (touch-up for a missed word)
censor manual "Movie.mkv" --edl fixes.csv
```

## How the timing works

Each swear found in the subtitles is resolved through a five-level ladder. The
level is recorded per interval in the JSON report and summarised at the end of
every run, so you can see exactly how much of a file was guesswork.

| Level | Condition | Mute span |
| --- | --- | --- |
| **L1** | An ASR word inside the cue window *is* the word | The word's own span, padded 0.06 s / 0.12 s |
| **L2** | An ASR word is a close match (misheard, variant spelling) | The word's span, padded 0.10 s / 0.20 s |
| **L3** | The swear didn't align, but its neighbours did | Interpolated between the anchors, padded 0.25 s / 0.30 s |
| **L4** | Nothing aligned | A proportional slice of the cue, plus a 0.45 s buffer |
| **L5** | The cue window is unusable, or `--paranoid` | The whole cue |

A subtitle hit is **never** discarded for lack of timing evidence. Excess silence
is the intended failure mode.

Before any of that, the subtitle track is synchronised to the audio. Distinctive
words present in both streams become anchors; a histogram vote finds the global
offset, and the surviving anchors are binned over time into a piecewise-linear
correction so that 23.976 ↔ 25 fps drift is absorbed too. Override with
`--sub-offset` if you already know the shift.

Profanity that the ASR hears but the subtitles omit (abridged or softened tracks
are common) is muted as well, subject to a confidence floor. Use
`--extras-policy report` to log those without muting them.

## Scene edit lists

An edit list is a CSV you write by hand. `censor edl-init movie.edl.csv` writes a
commented starter file.

```csv
# start,end,action,note,params
00:42:10.500,00:43:58.000,cut,pool scene
01:12:04.000,01:13:20.500,blackout,plot-critical,fade=0.5
01:31:00.000,01:31:40.000,blur,,strength=40
00:18:22.000,00:18:26.000,mute,word the model missed
```

Timestamps accept `SS.mmm`, `MM:SS.mmm`, `HH:MM:SS.mmm` and SRT-style
`HH:MM:SS,mmm`, so lines pasted straight out of a subtitle file work.

| Action | Effect | Params |
| --- | --- | --- |
| `cut` | Removed entirely. Everything after shifts earlier, and the subtitles follow. | – |
| `blackout` | Picture goes black; audio and subtitles keep playing. | `fade=<seconds>` |
| `blur` | Heavy blur or pixelation. | `strength=<sigma>`, `mode=gblur\|pixelate` |
| `mute` | Audio silenced, picture untouched. | – |

Aliases are accepted (`skip`/`remove` → `cut`, `black` → `blackout`,
`silence` → `mute`).

**Subtitles stay in sync across cuts.** Every mute interval, caption and chapter
marker is remapped through a single `TimeMap`. Captions wholly inside a cut are
dropped, captions straddling a boundary are clipped, and a caption spanning an
entire cut is rejoined across the seam. `--edl-preview` prints the dialogue
around each edit so you can check boundaries before committing to an encode, and
cut boundaries landing mid-word are flagged as warnings.

Any `cut`, `blackout` or `blur` forces a video re-encode (NVENC HEVC by default,
`--video-encoder libx264` if you'd rather). A file with only `mute` rows keeps
the original video stream bit-for-bit.

## Word lists

Rules live in [censor/wordlists/default.yaml](censor/wordlists/default.yaml),
split into `profanity`, `religious`, `sexual` and `slurs` tiers.

```powershell
censor run "Movie.mkv" --tiers profanity,slurs
censor run "Movie.mkv" --add-word bollocks --allow-word hell
censor words --tiers all --test "you christian son of a bitch"
```

Three rule kinds: `exact` (whole word), `substring` (catches inflections, so
`fuck` covers `unfuckingbelievable`), and `phrases` (multi-word, greedy
longest-first, hyphen and space interchangeable). An `exclusions` list protects
innocent words — `christian`, `analyst`, `cocktail`, `Scunthorpe` and friends.

## Output layout

For `-o out.mkv` you get:

| File | Contents |
| --- | --- |
| `out.mkv` | Video (copied unless scene edits force an encode), every audio track muted, censored subtitles as a **soft, toggleable, default** track |
| `out.censored.srt` | The same censored captions as a sidecar |
| `out.censor.json` | Every interval with its level, confidence, tier, source cue text, and both source and output timestamps |
| `.censor/<name>/` | Scratch: extracted subtitles and the cached transcript, so re-runs skip transcription |

Subtitles are never burned in. The original uncensored tracks are dropped by
default; `--keep-original-subs` retains them as non-default tracks.

## Command reference

| Command | Purpose |
| --- | --- |
| `censor run <file>` | Process one video (`censor <file>` also works) |
| `censor batch <dir> -o <dir>` | Process a folder, mirroring the tree |
| `censor manual <file> --edl <f>` | Apply an edit list with no detection |
| `censor edl-init <path>` | Write a starter edit list |
| `censor words` | Show or test the resolved word list |

`censor <command> --help` lists every flag.

## Tuning

| Flag | Use when |
| --- | --- |
| `--report-only` | You want the audit JSON without an encode |
| `--paranoid` | You'd rather lose a line than miss a word |
| `--blind-pad 0.8` | L4/L5 hits are still slipping through |
| `--sub-offset -2.5` | The subtitle track has a known shift |
| `--skip-asr` | Quick pass using cue timing alone |
| `--extras-policy ignore` | ASR-only detections are producing false positives |
| `--only-target-audio` | Only one audio track should be muted |
| `--audio-codec copy` | Never re-encode audio (mutes will not apply to copied tracks) |

Exit codes: `0` success, `2` bad input or config, `3` aborted (for example, no
subtitles and ASR-only mode declined), `1` a batch run had failures.

## Tests

```powershell
pip install pytest
python -m pytest
```

63 tests cover the matcher and its exclusions, the L1–L5 ladder, subtitle offset
recovery, interval merging, edit-list parsing and validation, `TimeMap` edge
cases (cuts at `t=0`, at EOF, adjacent cuts, cues straddling a seam), and
filtergraph construction including the 400-interval chunking regression.

## Project layout

- [censor](censor): current implementation
- [tests](tests): automated test suite
- [pyproject.toml](pyproject.toml): packaging and dependencies
- [TESTING_GUIDE.md](TESTING_GUIDE.md): copy-to-new-machine runbook
