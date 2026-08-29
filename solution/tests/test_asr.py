import os

from censor.asr import _ffmpeg_dll_directories, _winget_ffmpeg_dll_directories
from censor.config import Settings


def test_ffmpeg_dll_directories_ignores_static_builds(tmp_path):
    static_bin = tmp_path / "static" / "bin"
    shared_bin = tmp_path / "shared" / "bin"
    static_bin.mkdir(parents=True)
    shared_bin.mkdir(parents=True)
    (static_bin / "ffmpeg.exe").touch()
    (shared_bin / "ffmpeg.exe").touch()
    (shared_bin / "avcodec-61.dll").touch()

    path_value = os.pathsep.join((str(static_bin), str(shared_bin)))

    assert _ffmpeg_dll_directories(path_value) == [shared_bin]


def test_winget_ffmpeg_dll_directories_finds_shared_build(tmp_path):
    shared_bin = (
        tmp_path
        / "Microsoft"
        / "WinGet"
        / "Packages"
        / "Gyan.FFmpeg.Shared_Microsoft.Winget.Source_test"
        / "ffmpeg-7.1.1-full_build-shared"
        / "bin"
    )
    shared_bin.mkdir(parents=True)
    (shared_bin / "avcodec-61.dll").touch()

    assert _winget_ffmpeg_dll_directories(str(tmp_path)) == [shared_bin]


def test_cuda_friendly_compute_type_is_the_default():
    assert Settings().compute_type == "float16"
