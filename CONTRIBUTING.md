# Contributing to listen-in

Thanks for helping! Bug reports, docs fixes and pull requests are welcome.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[all,dev]"
```

## Before opening a pull request

```bash
ruff check src tests
pytest
```

- Automated tests must not touch live radio, download Whisper weights or call a paid
  LLM. Mock HTTP (`httpx.MockTransport`), subprocess and Whisper as the existing tests do.
- Live checks go in `tests/test_integration_live.py`, marked `integration`, skipped
  unless `RUN_LIVE_RADIO_TESTS=1`, recording 30 s or less in total.
- Never commit API keys, `.env` files, recordings, transcripts or model weights.
- Keep the core library free of station-specific logic, schedules, credentials and
  workspace details.
- Add a line to `CHANGELOG.md` under *Unreleased*.

## Adding an LLM provider

Implement the `listen_in.analysis.LLMProvider` protocol (`complete_json(...) -> LLMResponse`),
register it with `listen_in.register_provider("name", factory)`, and add mocked tests.
