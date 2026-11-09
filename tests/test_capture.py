from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from listen_in import capture
from listen_in.capture import build_capture_command, capture_stream, find_ffmpeg
from listen_in.exceptions import (
    CaptureError,
    FFmpegNotFoundError,
    InvalidInputError,
    OutputExistsError,
    StreamUnavailableError,
)
from listen_in.models import CaptureOptions

from .conftest import FIXED_NOW, write_wav


@pytest.fixture
def fake_ffmpeg(tmp_path) -> str:
    exe = tmp_path / "bin" / "ffmpeg"
    exe.parent.mkdir()
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(0o755)
    return str(exe)


@pytest.fixture
def opts(fake_ffmpeg) -> CaptureOptions:
    return CaptureOptions(ffmpeg_path=fake_ffmpeg)


class FakeRun:
    """Stand-in for subprocess.run that writes a WAV to ffmpeg's output argument."""

    def __init__(self, seconds: float | None = None, returncode: int = 0, stderr: str = "",
                 timeout: bool = False):
        self.seconds, self.returncode, self.stderr, self.timeout = seconds, returncode, stderr, timeout
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        if self.timeout:
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))
        if self.seconds is not None:
            write_wav(Path(cmd[-1]), self.seconds)
        return subprocess.CompletedProcess(cmd, self.returncode, "", self.stderr)


# ------------------------------------------------------------------- ffmpeg
def test_missing_ffmpeg_gives_install_instructions(monkeypatch):
    monkeypatch.setattr(capture.shutil, "which", lambda name: None)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)  # import -> ImportError
    with pytest.raises(FFmpegNotFoundError) as exc:
        find_ffmpeg()
    msg = str(exc.value)
    assert "brew install ffmpeg" in msg
    assert "apt-get install -y ffmpeg" in msg
    assert "ffmpeg_path" in msg


def test_explicit_ffmpeg_path_must_exist(tmp_path):
    with pytest.raises(FFmpegNotFoundError, match="ffmpeg_path"):
        find_ffmpeg(str(tmp_path / "nope"))


def test_env_var_ffmpeg_path(monkeypatch, fake_ffmpeg):
    monkeypatch.setenv("LISTEN_IN_FFMPEG", fake_ffmpeg)
    assert find_ffmpeg() == fake_ffmpeg


def test_import_does_not_require_ffmpeg(monkeypatch):
    monkeypatch.setattr(capture.shutil, "which", lambda name: None)
    import importlib

    import listen_in

    importlib.reload(listen_in)  # would raise if import-time checks existed


# ------------------------------------------------------------- command build
def test_command_is_safe_argument_list(fake_ffmpeg, tmp_path):
    nasty = "https://example.com/live; rm -rf / && echo $(whoami)"
    out = tmp_path / "x.partial.wav"
    cmd = build_capture_command(fake_ffmpeg, nasty, out, 60, CaptureOptions())
    assert isinstance(cmd, list) and all(isinstance(a, str) for a in cmd)
    assert cmd[0] == fake_ffmpeg
    assert cmd[cmd.index("-i") + 1] == nasty  # one argv item, never shell-split
    assert cmd[cmd.index("-t") + 1] == "60"
    assert cmd[cmd.index("-ac") + 1] == "1"
    assert cmd[cmd.index("-ar") + 1] == "16000"
    assert cmd[cmd.index("-c:a") + 1] == "pcm_s16le"
    assert cmd[-1] == str(out)
    assert "-nostdin" in cmd


def test_command_rejects_non_http_urls(fake_ffmpeg, tmp_path):
    with pytest.raises(InvalidInputError):
        build_capture_command(fake_ffmpeg, "file:///etc/passwd", tmp_path / "x.wav", 5, CaptureOptions())


def test_extra_input_args_cannot_override_output():
    with pytest.raises(ValueError):
        CaptureOptions(extra_input_args=["-i", "file:///etc/passwd"])


# ------------------------------------------------------------------- capture
def test_capture_success_atomic_and_deterministic(monkeypatch, tmp_path, news_station, opts):
    monkeypatch.setattr(capture, "utc_now", lambda: FIXED_NOW)
    fake = FakeRun(seconds=60)
    monkeypatch.setattr(capture.subprocess, "run", fake)

    audio = capture_stream(news_station, 60, tmp_path / "audio", options=opts)

    assert audio.path.name == "france-info_31074f8a_20261109T100000Z.wav"
    assert audio.path.exists()
    assert not list((tmp_path / "audio").glob("*.partial*"))
    assert audio.duration_seconds == pytest.approx(60)
    assert (audio.sample_rate, audio.channels) == (16_000, 1)
    cmd, kwargs = fake.calls[0]
    assert cmd[-1].endswith(".partial.wav")
    assert kwargs.get("shell") in (None, False)
    assert kwargs["timeout"] == 60 + opts.connect_timeout_seconds + opts.process_timeout_grace_seconds


def test_never_overwrites_without_flag(monkeypatch, tmp_path, news_station, opts):
    monkeypatch.setattr(capture.subprocess, "run", FakeRun(seconds=5))
    existing = write_wav(tmp_path / "keep.wav", 1)
    with pytest.raises(OutputExistsError):
        capture_stream(news_station, 5, tmp_path, options=opts, filename="keep.wav")
    assert existing.stat().st_size == 44 + 32_000  # untouched
    capture_stream(news_station, 5, tmp_path, options=opts, filename="keep.wav", overwrite=True)
    assert existing.stat().st_size == 44 + 5 * 32_000


def test_short_recording_fails_and_cleans_up(monkeypatch, tmp_path, news_station, opts):
    monkeypatch.setattr(capture.subprocess, "run", FakeRun(seconds=10))
    with pytest.raises(CaptureError, match="too short"):
        capture_stream(news_station, 60, tmp_path / "out", options=opts)
    assert list((tmp_path / "out").iterdir()) == []


def test_ffmpeg_error_raises_stream_unavailable(monkeypatch, tmp_path, news_station, opts):
    monkeypatch.setattr(capture.subprocess, "run",
                        FakeRun(returncode=1, stderr="Server returned 404 Not Found"))
    with pytest.raises(StreamUnavailableError, match="404"):
        capture_stream(news_station, 5, tmp_path / "out", options=opts)
    assert list((tmp_path / "out").iterdir()) == []


def test_process_timeout(monkeypatch, tmp_path, news_station, opts):
    monkeypatch.setattr(capture.subprocess, "run", FakeRun(timeout=True))
    with pytest.raises(CaptureError, match="did not finish"):
        capture_stream(news_station, 5, tmp_path, options=opts)


@pytest.mark.parametrize("bad", [0, -5, 1.5, "ten", 10_000_000])
def test_invalid_durations(tmp_path, news_station, opts, bad):
    with pytest.raises(InvalidInputError):
        capture_stream(news_station, bad, tmp_path, options=opts)


def test_configurable_max_duration(tmp_path, news_station, fake_ffmpeg):
    with pytest.raises(InvalidInputError, match="maximum"):
        capture_stream(news_station, 120, tmp_path,
                       options=CaptureOptions(ffmpeg_path=fake_ffmpeg, max_duration_seconds=60))


def test_filename_must_be_plain_wav(tmp_path, news_station, opts):
    with pytest.raises(InvalidInputError):
        capture_stream(news_station, 5, tmp_path, options=opts, filename="../escape.wav")
