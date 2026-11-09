"""Exception hierarchy for listen-in.

Every error raised deliberately by the package derives from :class:`ListenInError`
so callers can catch one type. Messages never contain API keys.
"""

from __future__ import annotations


class ListenInError(Exception):
    """Base class for all listen-in errors."""


class InvalidInputError(ListenInError, ValueError):
    """A user-supplied value (URL, UUID, duration, path...) failed validation."""


class ConfigurationError(ListenInError):
    """The package is configured incorrectly (unknown provider, missing key, ...)."""


class OptionalDependencyError(ListenInError, ImportError):
    """An optional extra is needed for this feature but is not installed."""

    def __init__(self, package: str, extra: str, feature: str) -> None:
        self.package = package
        self.extra = extra
        super().__init__(
            f"{feature} requires the optional dependency '{package}'. "
            f'Install it with:  pip install "listen-in[{extra}]"'
        )


# --------------------------------------------------------------------- stations
class StationResolutionError(ListenInError):
    """A station reference could not be resolved to a stream URL."""


class StationNotFoundError(StationResolutionError, LookupError):
    """Radio Browser has no station with the requested UUID."""


class RadioBrowserUnavailableError(StationResolutionError):
    """No Radio Browser API server could be reached."""


# ---------------------------------------------------------------------- capture
FFMPEG_INSTALL_HELP = """\
ffmpeg is required to record radio streams but was not found.

Install it with one of:
  macOS:          brew install ffmpeg
  Ubuntu/Debian:  sudo apt-get install -y ffmpeg
  Windows:        winget install Gyan.FFmpeg   (or choco install ffmpeg)
  Databricks:     add a cluster init script that runs `apt-get install -y ffmpeg`
  Any platform:   pip install "listen-in[ffmpeg]"   (bundled static binary)

Or point listen-in at a binary explicitly:
  CaptureOptions(ffmpeg_path="/path/to/ffmpeg")   or   export LISTEN_IN_FFMPEG=/path/to/ffmpeg
"""


class FFmpegNotFoundError(ListenInError, FileNotFoundError):
    """No usable ffmpeg executable was found."""

    def __init__(self, detail: str | None = None) -> None:
        msg = FFMPEG_INSTALL_HELP if detail is None else f"{detail}\n\n{FFMPEG_INSTALL_HELP}"
        super().__init__(msg)


class CaptureError(ListenInError):
    """Recording from the stream failed or produced unusable audio."""


class StreamUnavailableError(CaptureError):
    """The stream could not be opened."""


class OutputExistsError(ListenInError, FileExistsError):
    """The target output already exists and ``overwrite`` was not set."""


# ---------------------------------------------------------------- transcription
class TranscriptionError(ListenInError):
    """Whisper failed to load or to transcribe the audio."""


class LanguageMismatchError(TranscriptionError):
    """The language hint clearly disagrees with the language spoken in the audio.

    Raised only with ``on_language_mismatch="error"``.
    """


# --------------------------------------------------------------------- analysis
class AnalysisError(ListenInError):
    """The LLM call failed or returned output that could not be validated."""


class AnalysisRefusedError(AnalysisError):
    """The model declined the request (``stop_reason == "refusal"``)."""
