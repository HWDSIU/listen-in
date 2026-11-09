"""listen-in: record internet radio, transcribe it locally with Whisper, summarise news with an LLM.

Importing the package needs neither Whisper nor ffmpeg; they are checked when used.
"""

from __future__ import annotations

import logging

from ._version import __version__
from .analysis import LLMProvider, available_providers, register_provider
from .api import (
    analyse_transcript,
    capture_stream,
    probe_stream,
    process_station,
    resolve_station,
    transcribe_audio,
    translate_transcript,
)
from .exceptions import (
    AnalysisError,
    AnalysisRefusedError,
    CaptureError,
    ConfigurationError,
    FFmpegNotFoundError,
    InvalidInputError,
    LanguageMismatchError,
    ListenInError,
    OptionalDependencyError,
    OutputExistsError,
    RadioBrowserUnavailableError,
    StationNotFoundError,
    StationResolutionError,
    StreamUnavailableError,
    TranscriptionError,
)
from .models import (
    AnalysisResult,
    AudioCapture,
    BilingualTranscript,
    CaptureOptions,
    LLMConfig,
    NewsAnalysis,
    NewsStory,
    ProcessingResult,
    StageResult,
    Station,
    Transcript,
    TranscriptSegment,
    TranslatedSegment,
    Translation,
)

analyze_transcript = analyse_transcript  # US-spelling alias

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "AnalysisError",
    "AnalysisRefusedError",
    "AnalysisResult",
    "AudioCapture",
    "BilingualTranscript",
    "CaptureError",
    "CaptureOptions",
    "ConfigurationError",
    "FFmpegNotFoundError",
    "InvalidInputError",
    "LLMConfig",
    "LLMProvider",
    "LanguageMismatchError",
    "ListenInError",
    "NewsAnalysis",
    "NewsStory",
    "OptionalDependencyError",
    "OutputExistsError",
    "ProcessingResult",
    "RadioBrowserUnavailableError",
    "StageResult",
    "Station",
    "StationNotFoundError",
    "StationResolutionError",
    "StreamUnavailableError",
    "Transcript",
    "TranscriptSegment",
    "TranscriptionError",
    "TranslatedSegment",
    "Translation",
    "__version__",
    "analyse_transcript",
    "analyze_transcript",
    "available_providers",
    "capture_stream",
    "probe_stream",
    "process_station",
    "register_provider",
    "resolve_station",
    "transcribe_audio",
    "translate_transcript",
]
