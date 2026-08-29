# Testing Guide For High-Power PC

This guide is the fastest path to validate the new censor pipeline on your stronger machine.

## Recommended platform

Use native Windows first.

Reason:
- Your workflow and paths are already Windows-native.
- ffmpeg + NVIDIA encoding and CUDA setup is usually simpler on native Windows.
- You avoid cross-filesystem and path translation issues while you are validating behavior.

WSL2 is still a good option later if you specifically want Linux tooling or job orchestration, but it is unnecessary for first-pass validation.

## 1. Move project to target machine

Copy the full folder and open a terminal in the project root (same folder as pyproject.toml).

## 2. Install prerequisites

1. Install Python 3.12 (or 3.10+).
2. Install ffmpeg and ensure both ffmpeg and ffprobe are on PATH.
3. Create and activate a virtual environment.
4. Install package and core dependencies:

```powershell
pip install -e .
```

5. Install GPU PyTorch (CUDA 12.4 wheels):

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

6. Install WhisperX:

```powershell
pip install whisperx
```

Optional:

```powershell
pip install pytest
```

## 3. Validate install quickly

```powershell
python -m censor.cli --help
python -m censor.cli words --tiers profanity,religious --test "son of a bitch"
python -m pytest -q
```

## 4. First run without encoding

Do this first to inspect detection quality and timing confidence.

```powershell
python -m censor.cli run "D:\media\movie.mkv" --report-only --engine whisperx --device cuda --model large-v3
```

This writes the JSON audit report and does not write output video.

## 5. Normal single-file run

```powershell
python -m censor.cli run "D:\media\movie.mkv" -o "D:\media\movie_censored.mkv" --engine whisperx --device cuda --model large-v3 --overwrite
```

Default behavior:
- Subtitle-guided swear detection plus ASR timing alignment.
- Censored soft subtitle track is added and can be toggled on or off in player.
- All audio tracks are muted by default where intervals apply.

## 6. Scene edit list workflow

Create starter file:

```powershell
python -m censor.cli edl-init "D:\media\movie.edl.csv"
```

Example rows:

```csv
00:42:10.500,00:43:58.000,cut,pool scene
01:12:04.000,01:13:20.500,blackout,plot-critical,fade=0.5
01:31:00.000,01:31:40.000,blur,,strength=40
00:18:22.000,00:18:26.000,mute,missed line
```

Preview boundary impact before rendering:

```powershell
python -m censor.cli run "D:\media\movie.mkv" --edl "D:\media\movie.edl.csv" --edl-preview --report-only
```

Render with edits:

```powershell
python -m censor.cli run "D:\media\movie.mkv" -o "D:\media\movie_censored.mkv" --edl "D:\media\movie.edl.csv" --engine whisperx --device cuda --overwrite
```

## 7. Batch mode

```powershell
python -m censor.cli batch "D:\media\TV" -o "D:\media\TV_clean" --recursive --yes --engine whisperx --device cuda --model large-v3 --overwrite
```

Optional per-file EDL folder:
- Place files as <video stem>.edl.csv
- Use --edl-dir "D:\media\edl"

## 8. Outputs to check

For output named movie_censored.mkv:
- movie_censored.mkv
- movie_censored.censored.srt
- movie_censored.censor.json
- .censor scratch/cache folder in the working area

## 9. Useful tuning flags

- More conservative censoring: --paranoid
- Wider uncertain mute windows: --blind-pad 0.8
- Known subtitle offset: --sub-offset -2.4
- Skip fallback prompt when no subtitles: --allow-asr-fallback or --yes
- Mute only one audio track: --only-target-audio --audio-index 0

## 10. Windows vs WSL quick decision

Use Windows now for first real testing.

Choose WSL2 later only if one of these is true:
- You want Linux-first automation tooling.
- You are integrating with Linux media pipelines.
- You are comfortable validating CUDA + ffmpeg + NVENC in WSL separately.

If you later try WSL2, keep media files inside the Linux filesystem during runs for best IO behavior.