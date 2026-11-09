# listen-in

Record a short segment from an internet radio station, transcribe it **locally**
with [OpenAI Whisper](https://github.com/openai/whisper), and optionally have an LLM
(Claude, by default `claude-haiku-5-5`) produce a structured **English** news summary,
whatever the broadcast language.

```python
from listen_in import process_station

result = process_station(
    "https://www.radio-browser.info/history/fc939579-455a-4593-9e0d-ee9df38f5c2f",  # Radio Jambo
    language="sw",
    duration_seconds=600,
    output_dir="./output",
    llm={"provider": "anthropic", "model": "claude-haiku-5-5"},  # key from $LLM_API_KEY
)
print(result.statuses)
print(result.analysis.analysis.summary_en)
```

> **Status:** alpha (0.1.0). The API may change before 1.0.

[![tests](https://github.com/HWDSIU/listen-in/actions/workflows/test.yml/badge.svg)](https://github.com/HWDSIU/listen-in/actions/workflows/test.yml)

## What it does

- Resolves a station from a **direct stream URL**, a **[Radio Browser](https://www.radio-browser.info) UUID**,
  or a **Radio Browser history URL**, using the Radio Browser API with automatic server failover.
- Records the stream with `ffmpeg` to mono 16 kHz WAV (atomic writes, timeouts, no shell).
- Transcribes with Whisper on CPU or CUDA. It keeps the **original-language** text and
  timestamped segments and never uses Whisper's translate mode.
- Optionally sends the transcript to an LLM. The model returns JSON (validated with
  Pydantic) with an English summary and a list of stories with people,
  organisations, locations, key claims and uncertainties.
- Writes audio, transcripts, analysis and a per-run JSON **manifest** to a local
  folder or a Databricks Unity Catalog Volume.

## What it does not do

- It does not schedule anything. Call it from cron, Airflow or a Databricks Job.
- It does not verify facts. The analysis covers only what the transcript says; it is not checked against the outside world.
- It does not bundle Whisper weights or (by default) ffmpeg.
- It does not start Spark sessions, background threads or services.

## Responsible use and licensing

Radio broadcasts are copyrighted. Before recording, storing or processing a station,
make sure you are permitted to do so under the station's terms and the laws that
apply to you (copyright, broadcasting and data-protection law). Phone-ins and
interviews may contain personal data. Keep recordings only as long as you need them;
`process_station(..., keep_audio=False)` deletes audio after transcription. Do not
redistribute captured audio or transcripts. Machine transcripts and LLM summaries
contain errors, so check anything important against the source audio.

The code is licensed under [Apache-2.0](LICENSE). Whisper and model weights carry their own licences.

## Installation

Python 3.10 or later.

```bash
pip install "listen-in[whisper,llm-anthropic]"
```

Until the package is on PyPI, install from GitHub:

```bash
pip install "listen-in[whisper,llm-anthropic] @ git+https://github.com/HWDSIU/listen-in.git@main"
```

| Extra | Adds | Needed for |
|---|---|---|
| *(none)* | `httpx`, `pydantic`, `platformdirs` | station resolution, models |
| `whisper` | `openai-whisper` (+ PyTorch), `numpy`, `certifi` | `transcribe_audio` |
| `llm-anthropic` | `anthropic` | `analyse_transcript` with Claude |
| `ffmpeg` | `imageio-ffmpeg` (static ffmpeg binary) | recording where you cannot install ffmpeg |
| `all` | all of the above | |

Importing `listen_in` needs neither Whisper nor ffmpeg. They are checked when first
used, and the error explains how to install them.

### ffmpeg (system dependency)

`ffmpeg` is a separate program; pip cannot reliably install it, except through the
`ffmpeg` extra below.

| Platform | Command |
|---|---|
| macOS | `brew install ffmpeg` |
| Ubuntu / Debian | `sudo apt-get install -y ffmpeg` |
| Windows | `winget install Gyan.FFmpeg` (or `choco install ffmpeg`) |
| Anywhere (fallback) | `pip install "listen-in[ffmpeg]"` |

listen-in looks for ffmpeg in this order: `CaptureOptions(ffmpeg_path=...)`, then `$LISTEN_IN_FFMPEG`,
then `ffmpeg` on `PATH`, then the `imageio-ffmpeg` binary.

```python
from listen_in import CaptureOptions
opts = CaptureOptions(ffmpeg_path="/usr/bin/ffmpeg")
```

### First run downloads Whisper weights

The first transcription downloads the model to `~/.cache/whisper`: about 460 MB for
`small` and 1.5 GB for `medium`. Allow time and disk space, and outbound HTTPS access to
`openaipublic.azureedge.net`. Pass `download_root=` to `transcribe_audio` to use a different cache.

## Usage

### Transcription only

```python
from listen_in import resolve_station, capture_stream, transcribe_audio

station = resolve_station("https://www.radio-browser.info/history/cf0316fa-091e-4dc4-92a7-d8b0bd0cad28")
audio = capture_stream(station, duration_seconds=120, output_dir="./output/audio")
transcript = transcribe_audio(audio.path, language="en", model="small", device="auto")
print(transcript.text)
for seg in transcript.segments:
    print(f"[{seg.start:6.1f}-{seg.end:6.1f}] {seg.text}")
```

Or end to end, without an LLM:

```python
result = process_station("fc939579-455a-4593-9e0d-ee9df38f5c2f", language="sw",
                         duration_seconds=300, output_dir="./output", llm=None)
```

### Capture, transcription and LLM analysis

```python
import os
from listen_in import process_station

result = process_station(
    station="fc939579-455a-4593-9e0d-ee9df38f5c2f",
    language="sw",
    duration_seconds=600,
    output_dir="./output",
    whisper_model="small",
    llm={
        "provider": "anthropic",
        "model": "claude-haiku-5-5",
        "api_key": os.environ["LLM_API_KEY"],   # or omit: $LLM_API_KEY / $ANTHROPIC_API_KEY
        "effort": "medium",                      # low | medium | high
    },
)

if result.ok:
    for story in result.analysis.analysis.stories:
        print(story.headline, "-", story.summary)
else:
    print(result.stages)  # per-stage status, timing and non-secret error
```

`process_station` returns a `ProcessingResult` that holds the resolved station, the
audio metadata and path, the transcript and its path, the analysis and its path, the
manifest path, and a status and timing for each stage. A failed stage never deletes
earlier artefacts. If analysis fails, for example, the transcript is still saved.

You can also call `analyse_transcript(transcript, api_key=..., model="claude-haiku-5-5")` directly.

Analysis schema:

```json
{
  "news_detected": true,
  "broadcast_language": "sw",
  "translation_used": true,
  "summary_en": "Concise English summary based only on the transcript.",
  "stories": [
    {"headline": "...", "summary": "...", "people": [], "organisations": [],
     "locations": [], "key_claims": [], "uncertainties": []}
  ]
}
```

#### LLM providers

The first version ships the `anthropic` provider. It uses Claude structured outputs
so responses follow the schema, then validates them with Pydantic and allows one
repair retry. Other back ends implement the small `LLMProvider` protocol and are
registered with `listen_in.register_provider(name, factory)`. API keys are held as
`SecretStr` and never appear in logs, files, manifests or returned models.

### Two-station test

`examples/local_two_station_test.py` records Radio Jambo (Swahili) and Classic 105
(English), both from Kenya, one after the other:

```bash
cp .env.example .env    # add LLM_API_KEY / LLM_MODEL to enable analysis
python examples/local_two_station_test.py --duration 120 --model small --output-dir ./test-output
```

It prints the resolved stream URLs, the status of each stage and the output paths. The exit code is non-zero
if capture or transcription fails.

### Output layout

```text
output/
├── audio/        radio-jambo_fc939579_20261109T100000Z.wav
├── transcripts/  radio-jambo_fc939579_20261109T100000Z.json  (+ .txt)
├── analysis/     radio-jambo_fc939579_20261109T100000Z.json
└── manifests/    radio-jambo_fc939579_20261109T100000Z.json
```

File names are `<station-slug>_<uuid prefix>_<UTC start time>`. Existing files are
never overwritten unless you pass `overwrite=True`.

## Databricks

### Install

On a **classic or dedicated (single-user) cluster**, add an init script that installs ffmpeg
([`examples/databricks_init_script.sh`](examples/databricks_init_script.sh)):

```bash
#!/bin/bash
apt-get update
apt-get install -y ffmpeg
```

Then in the notebook:

```python
%pip install "listen-in[whisper,llm-anthropic] @ git+https://github.com/HWDSIU/listen-in.git@main"
dbutils.library.restartPython()
```

Serverless compute cannot run init scripts. The supported route there is the
`ffmpeg` extra (`listen-in[whisper,llm-anthropic,ffmpeg]`), but it has not been
tested on serverless yet. Classic compute with the init script is the supported
MVP path. GPU compute (CUDA) makes Whisper much faster, and FP16 is used
automatically on CUDA.

The cluster needs outbound HTTPS access to `*.api.radio-browser.info`, the stream
hosts, `api.anthropic.com` and, on first use, `openaipublic.azureedge.net` for the
Whisper weights.

### Secrets and Unity Catalog Volume output

```python
from listen_in import process_station

api_key = dbutils.secrets.get(scope="listen-in", key="llm-api-key")

result = process_station(
    station="fc939579-455a-4593-9e0d-ee9df38f5c2f",
    language="sw",
    duration_seconds=600,
    output_dir="/Volumes/main/radio/listen_in/outputs",
    whisper_model="small",
    llm={"provider": "anthropic", "model": "claude-haiku-5-5", "api_key": api_key},
)
display(result.model_dump(mode="json"))
```

Audio is recorded to local driver storage (`/tmp`) and copied to the Volume when it
is complete. listen-in never creates a Spark session. To load the run manifests into Delta:

```python
spark.read.option("multiLine", True).json("/Volumes/main/radio/listen_in/outputs/manifests/") \
     .write.mode("append").saveAsTable("main.radio.listen_in_runs")
```

### Scheduling

Use **Databricks Workflows**, not a loop inside a notebook. Create a Job with one notebook
or Python wheel task per station, using
[`examples/databricks_notebook_example.py`](examples/databricks_notebook_example.py)
with `station` and `language` parameters, and give it a cron schedule such as
`0 0,30 * * * ?` (minute 0 and 30). Each task runs capture and transcription on the
driver. Whisper is not run through Spark UDFs.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FFmpegNotFoundError` | Install ffmpeg (see above), set `CaptureOptions(ffmpeg_path=...)` or `$LISTEN_IN_FFMPEG`, or `pip install "listen-in[ffmpeg]"`. |
| `RadioBrowserUnavailableError` | Outbound HTTPS to `*.api.radio-browser.info` is blocked. Allow it, or pass the stream URL directly. |
| `StreamUnavailableError` / "Recording too short" | The station is offline, geo-blocked or has changed its URL. Radio Browser records and stream URLs **change over time**; resolve again or choose another station. |
| `CERTIFICATE_VERIFY_FAILED` while downloading Whisper weights | python.org builds on macOS have no CA bundle until you run *Install Certificates.command*. listen-in falls back to `certifi` automatically; behind a corporate proxy, set `SSL_CERT_FILE` to your organisation's CA bundle. |
| Transcription is very slow | CPU Whisper is roughly real-time or slower for `small` on a laptop. Use a CUDA GPU, a smaller model (`base`, `tiny`) or shorter recordings. |
| Swahili names or words are garbled | `small` has limited Swahili accuracy. Try `medium` or `large-v3`, or pass `initial_prompt=` with expected names. The LLM records doubtful names under `uncertainties`. |
| `ConfigurationError: No API key` | Pass `api_key=` or set `LLM_API_KEY` (or `ANTHROPIC_API_KEY`). |

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[all,dev]"
pytest                    # mocked: no network, no Whisper download, no paid API
ruff check src tests
RUN_LIVE_RADIO_TESTS=1 pytest -m integration   # opt-in: records 2 x 15 s of live radio
```

