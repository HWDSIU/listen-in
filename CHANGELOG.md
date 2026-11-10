# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.4.1] - 2026-11-10

### Fixed
- A correct hint for a closely related language is no longer overridden. Whisper's
  detector can't tell Hindi/Urdu, Serbian/Croatian/Bosnian, Malay/Indonesian,
  Norwegian Bokmål/Nynorsk or Persian/Tajik apart from audio; an Urdu station hinted
  `"ur"` was being transcribed as Hindi. The mismatch check now scores the hint's
  whole group.
- Set-up mistakes now raise before recording instead of after it: a missing Whisper
  or LLM package, an unknown language code, Whisper model or LLM provider, CUDA
  requested but unavailable, or a missing Azure endpoint. The LLM client is built once
  and reused for translation and analysis.
- Tracebacks logged for failed stages are redacted, so an API key inside an exception
  message never reaches the logs.
- `keep_audio=False` now deletes the recording only after the transcript is saved; if
  transcription fails the recording is kept for a retry. The manifest and transcript
  file no longer point at a deleted file (`audio.deleted` is `true`, paths `null`).
- `station_overrides` rejects impossible coordinates.

### Added
- `NewsStory.time_ranges`: one range per stretch of the recording a story covers, so
  a story trailed in the headlines and covered later is located correctly
  (`start`/`end` still give the overall span).
- `TimeRange` in the public API, and `listen_in.transcription.check_transcription_ready()`
  for checking Whisper set-up without recording.
- README note on recording length: designed for snippets up to about 20 minutes.

### Changed
- Manifests leave out the output folder's location, the ffmpeg path and the LLM
  endpoint URL. The transcript file's `audio_path` is relative to the output folder.
- CI uses the current major versions of the GitHub Actions it relies on.

### Removed
- Unused `platformdirs` dependency.

## [0.4.0] - 2026-11-09

### Added
- Per-story `entities`: type, name as heard, `canonical_name` (standardised so
  the same entity can be matched across stories, stations and languages),
  `role` as stated, and an `uncertain` flag. The broadcasting station itself is
  excluded. The plain `people` / `organisations` / `locations` lists are kept.
- Per-story `topics` from a fixed list, and `segment_ids` with `start` / `end`
  seconds (computed by listen-in from the transcript segments) to link a story
  back to the audio. The analysis prompt now numbers transcript segments.
- `run` block at the top of the analysis file (run id, recording time,
  duration, language, station name, country, region and coordinates), so it
  stands on its own.
- `Station.location_precision` (`station`, `subdivision`, `country` or none),
  a built-in country reference point used when Radio Browser has no
  coordinates, and `process_station(station_overrides=...)` to supply or
  correct station details.

### Changed
- Manifests no longer repeat `audio.station` or `analysis.transcript`, and
  store file paths relative to `output_dir`.
- `ProcessingResult.language` is now the transcript's language; the
  requested language moved to `language_hint` (`None` when detecting).
- README: platform-specific deployment and scheduling sections and the
  notebook example removed; new sections on the analysis file, station
  location and using the results (map and entity network examples).

## [0.3.1] - 2026-11-09

### Added
- More Radio Browser metadata on `Station`, and so in every manifest:
  `subdivision` and `subdivision_code` (e.g. `Berlin` / `DE-BE`), `latitude`
  and `longitude`, `language_codes`, `tags` and `favicon`. Missing,
  blank or out-of-range values are left empty.
- Project logo (`assets/listen-in-icon.svg`) at the top of the README.
- README and package description now lead with the structured analysis (stories,
  people, organisations, places, key claims, uncertainties), not just summaries.

## [0.3.0] - 2026-11-09

### Added
- English translation of non-English broadcasts, segment by segment with
  timestamps (`translate_transcript`, and a `translate` stage in
  `process_station`). The original transcript is kept; the English text is
  saved as `transcripts/<run>.en.txt`, and the analysis file and manifest carry
  both side by side (`analysis.transcript`: `text`, `text_en`, `segments`).
  On by default when an LLM is configured; `translate=False` turns it off.
  English broadcasts make no translation call.
- `Translation`, `TranslatedSegment` and `BilingualTranscript` models;
  `StageResult.detail` explains why a stage was skipped.
- `on_language_mismatch` on `transcribe_audio` / `process_station`:
  `"use_detected"` (default), `"use_hint"` or `"error"`
  (raises `LanguageMismatchError`).
- `Transcript.language_hint`, `detected_language`,
  `detected_language_probability` and `language_mismatch`.

### Changed
- `language=None` now uses multi-window language detection rather than
  Whisper's first-30-seconds guess.
- The LLM prompt states the transcript language and detection confidence.

### Fixed
- A wrong `language` hint no longer turns the transcript into a translation.
  Whisper treats its language setting as an instruction, so German audio
  transcribed with `language="en"` came out in English. The hint is now checked
  with Whisper's language detection over up to five 30 s windows. When detection
  is confident (>= 50 %) and the hint gets < 10 %, the detected language is used
  and a warning is logged.

## [0.2.0] - 2026-11-09

### Added
- `openai` provider: the OpenAI API or any OpenAI-compatible server (`base_url`).
- `azure_openai` provider for Azure OpenAI / Azure AI Foundry deployments
  (`model` = deployment name, `base_url` = resource endpoint, `api_version`).
  Aliases: `openai_compatible`, `azure`, `azure_foundry`.
- Chat Completions with strict JSON-schema structured outputs, or
  `structured_output="json_object"` for deployments without schema support.
- `llm-openai` extra (`openai>=1.66`).
- Provider-specific key variables: `OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY`;
  `AZURE_OPENAI_ENDPOINT` for the Azure endpoint.

### Changed
- `LLMConfig.effort` now defaults to unset and is only sent when given (Claude
  Haiku 5.5's own default is `medium`); non-reasoning OpenAI models reject it.
- `LLMConfig.model` defaults to `claude-haiku-5-5` for `anthropic` only and is
  required for the other providers.

## [0.1.0] - 2026-11-09

### Added
- `resolve_station`: direct stream URLs, Radio Browser UUIDs and history URLs, with
  DNS-based server discovery and failover.
- `capture_stream` / `probe_stream`: ffmpeg recording to mono 16 kHz WAV with atomic
  writes, timeouts and input validation.
- `transcribe_audio`: lazy, cached openai-whisper transcription (CPU/CUDA, FP16 on CUDA).
- `analyse_transcript`: structured English news analysis via Anthropic Claude
  (default `claude-haiku-5-5`) with Pydantic validation and one repair retry.
- `process_station`: end-to-end run with per-stage status/timings and JSON manifests,
  local or mounted-volume output.
- Examples for a local two-station test (English + French) and a hosted-notebook task.
- README guidance on choosing a Whisper model: `small` by default, `turbo` for less
  widely spoken languages such as Swahili.

### Fixed
- The `persist` stage now records how long the manifest write actually took.
