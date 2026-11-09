# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
