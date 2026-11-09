# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
  local or Databricks Volume output.
- Examples for a local two-station test (English + French) and a Databricks notebook task.
- README guidance on choosing a Whisper model: `small` by default, `turbo` for less
  widely spoken languages such as Swahili.

### Fixed
- The `persist` stage now records how long the manifest write actually took.
