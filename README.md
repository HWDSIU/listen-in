# listen-in

Record a short segment from an internet radio station, transcribe it **locally**
with [OpenAI Whisper](https://github.com/openai/whisper), and optionally have an LLM
(Claude, OpenAI, or a model deployed on Azure OpenAI / Azure AI Foundry) produce a
structured **English** news summary, whatever the broadcast language.

```python
from listen_in import process_station

result = process_station(
    "https://www.radio-browser.info/history/31074f8a-e6f4-11e9-a96c-52543be04c81",  # France Info
    language="fr",
    duration_seconds=600,
    output_dir="./output",
    llm={"provider": "anthropic", "model": "claude-haiku-5-5"},  # key from $LLM_API_KEY
)
print(result.statuses)
print(result.analysis.analysis.summary_en)
```

> **Status:** alpha (0.3.0). The API may change before 1.0.

[![tests](https://github.com/HWDSIU/listen-in/actions/workflows/test.yml/badge.svg)](https://github.com/HWDSIU/listen-in/actions/workflows/test.yml)

## What it does

- Resolves a station from a **direct stream URL**, a **[Radio Browser](https://www.radio-browser.info) UUID**,
  or a **Radio Browser history URL**, using the Radio Browser API with automatic server failover.
- Records the stream with `ffmpeg` to mono 16 kHz WAV (atomic writes, timeouts, no shell).
- Transcribes with Whisper on CPU or CUDA. It keeps the **original-language** text and
  timestamped segments and never uses Whisper's translate mode.
- Optionally sends the transcript to an LLM. The model returns JSON (validated with
  Pydantic) with an English summary and a list of stories with people,
  organisations, locations, key claims and uncertainties. Non-English broadcasts are
  also translated into English **segment by segment**, alongside the original text
  and timestamps.
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

listen-in is **not yet published on PyPI**, so install a tagged release from GitHub:

```bash
pip install "listen-in[whisper,llm-anthropic] @ git+https://github.com/HWDSIU/listen-in.git@v0.3.0"
```

Swap `llm-anthropic` for `llm-openai` to use OpenAI or Azure OpenAI / AI Foundry (or
include both). Pin to a release tag (see [tags](https://github.com/HWDSIU/listen-in/tags)) for
reproducible installs, or use `@main` for the latest unreleased code. Once the package
is on PyPI this will become `pip install "listen-in[whisper,llm-anthropic]"`.

| Extra | Adds | Needed for |
|---|---|---|
| *(none)* | `httpx`, `pydantic`, `platformdirs` | station resolution, models |
| `whisper` | `openai-whisper` (+ PyTorch), `numpy`, `certifi` | `transcribe_audio` |
| `llm-anthropic` | `anthropic` | analysis with Claude (`provider="anthropic"`) |
| `llm-openai` | `openai` | analysis with OpenAI or Azure OpenAI / AI Foundry (`provider="openai"` / `"azure_openai"`) |
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
| Anywhere (fallback) | add the `ffmpeg` extra, e.g. `listen-in[whisper,llm-anthropic,ffmpeg] @ git+https://github.com/HWDSIU/listen-in.git@v0.3.0` |

listen-in looks for ffmpeg in this order: `CaptureOptions(ffmpeg_path=...)`, then `$LISTEN_IN_FFMPEG`,
then `ffmpeg` on `PATH`, then the `imageio-ffmpeg` binary.

```python
from listen_in import CaptureOptions
opts = CaptureOptions(ffmpeg_path="/usr/bin/ffmpeg")
```

### Choosing a Whisper model

The Whisper model is a parameter: `whisper_model=` on `process_station`, `model=` on
`transcribe_audio`, `--model` in the example script, and the `whisper_model` widget in the
Databricks notebook. The default is `small`.

| Model | Download | Use it for |
|---|---|---|
| `small` (default) | ~460 MB | English, French and most widely spoken languages. Accurate enough for news monitoring and runs faster than real time on a laptop CPU. |
| `turbo` (large-v3-turbo) | ~1.6 GB | Less widely spoken languages such as **Swahili**, where `small` misspells names, institutions and figures. Needs a GPU or a machine with 16 GB+ RAM; on an 8 GB laptop CPU it runs at roughly half real-time speed. |
| `medium`, `large-v3` | ~1.5 GB / ~2.9 GB | Alternatives when `turbo` is not good enough; slower. |
| `base`, `tiny` | ~140 MB / ~75 MB | Quick tests only. |

In a test on a 5-minute Swahili news bulletin, `turbo` transcribed officials' names,
institutions and figures (percentages, sums of money) correctly. `small` produced phonetic
approximations of the same passages and dropped the last 40 seconds. Haiku's summary
went from vague ("water company (name unclear)") to specific. For English and French
news, `small` was already clean.

```python
# Most languages
process_station(station, language="en", whisper_model="small", ...)
# Less widely spoken languages, ideally on GPU
process_station(station, language="sw", whisper_model="turbo", ...)
```

### Language hints

`language=` takes an ISO 639-1 code (`"en"`, `"fr"`, `"de"`, `"sw"`...) or `None` to
detect it. Whisper treats its language setting as an instruction, not a hint: given
the wrong language, it quietly **translates** the speech into that language, and the
original wording is lost. So listen-in checks the hint first, running Whisper's
language detection over several 30-second windows of the recording. If detection is
confident (at least 50 %) and gives the hint almost no weight (under 10 %), the
recording is transcribed in the detected language and a warning is logged.
`on_language_mismatch="use_hint"` keeps your hint anyway, and `"error"` stops instead.

The transcript records what happened: `language` (what Whisper transcribed in),
`language_hint`, `detected_language`, `detected_language_probability` and
`language_mismatch`. A correct hint is never overridden by short stretches of another
language, such as adverts or quotes in English.

### First run downloads Whisper weights

The first transcription downloads the model to `~/.cache/whisper`: about 460 MB for
`small` and 1.6 GB for `turbo`. Allow time and disk space, and outbound HTTPS access to
`openaipublic.azureedge.net`. Pass `download_root=` to `transcribe_audio` to use a different cache.

## Usage

### Transcription only

```python
from listen_in import resolve_station, capture_stream, transcribe_audio

station = resolve_station("https://www.radio-browser.info/history/598c4d0e-6b06-43fb-bff4-717c591213a9")  # BBC World Service
audio = capture_stream(station, duration_seconds=120, output_dir="./output/audio")
transcript = transcribe_audio(audio.path, language="en", model="small", device="auto")
print(transcript.text)
for seg in transcript.segments:
    print(f"[{seg.start:6.1f}-{seg.end:6.1f}] {seg.text}")
```

Or end to end, without an LLM:

```python
result = process_station("31074f8a-e6f4-11e9-a96c-52543be04c81", language="fr",
                         duration_seconds=300, output_dir="./output", llm=None)
```

### Capture, transcription and LLM analysis

```python
import os
from listen_in import process_station

result = process_station(
    station="31074f8a-e6f4-11e9-a96c-52543be04c81",
    language="fr",
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

#### Original transcript and English translation

When an `llm` is configured and the broadcast isn't English, `process_station` also
translates the transcript into English, one Whisper segment at a time. Use the same
`llm` settings; `translate=False` turns it off. The original is never replaced:

- `transcripts/<run>.txt`: the original-language transcript
- `transcripts/<run>.en.txt`: the English translation
- the analysis file and manifest carry both side by side, per segment with timestamps:

```json
"transcript": {
  "language": "fr",
  "translated": true,
  "text": "Il est treize heures. Le prix du carburant augmente. ...",
  "text_en": "It is one o'clock. The price of fuel is rising. ...",
  "segments": [
    {"id": 0, "start": 0.0, "end": 2.5, "text": "Il est treize heures.", "text_en": "It is one o'clock."}
  ]
}
```

For English broadcasts no translation call is made: `text_en` is the original and
`translated` is `false`. In Python the same data is available as
`result.translation` (segments, `text_en`) and `result.analysis.transcript`. To
translate a transcript on its own, call `translate_transcript(transcript, ...)`. It
takes the same provider settings as `analyse_transcript`. Names are kept as heard,
and unclear passages are marked `[unclear]`.

Analysis schema (in addition to the `transcript` block above):

```json
{
  "news_detected": true,
  "broadcast_language": "fr",
  "translation_used": true,
  "summary_en": "Concise English summary based only on the transcript.",
  "stories": [
    {"headline": "...", "summary": "...", "people": [], "organisations": [],
     "locations": [], "key_claims": [], "uncertainties": []}
  ]
}
```

#### LLM providers

| `provider` | Extra | `model` | Other settings | Key from (if `api_key` not passed) |
|---|---|---|---|---|
| `anthropic` (default) | `llm-anthropic` | defaults to `claude-haiku-5-5` | — | `$LLM_API_KEY`, `$ANTHROPIC_API_KEY` |
| `openai` | `llm-openai` | required, e.g. `gpt-6.1` | `base_url` for any OpenAI-compatible server | `$LLM_API_KEY`, `$OPENAI_API_KEY` |
| `azure_openai` | `llm-openai` | required: your **deployment name** | `base_url` = resource endpoint (or `$AZURE_OPENAI_ENDPOINT`); `api_version` (default `2024-12-01-preview`) | `$LLM_API_KEY`, `$AZURE_OPENAI_API_KEY` |

Common settings: `effort` (reasoning effort, sent only when set; leave it unset for
models without reasoning controls), `max_tokens`, `timeout_seconds`, `max_retries`.
`openai_compatible` is accepted as an alias of `openai`, and `azure` / `azure_foundry`
of `azure_openai`.

Azure OpenAI / Azure AI Foundry, with the key held in Azure Key Vault:

```python
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

api_key = SecretClient(vault_url="https://<vault>.vault.azure.net",
                       credential=DefaultAzureCredential()).get_secret("<secret-name>").value

result = process_station(
    station="598c4d0e-6b06-43fb-bff4-717c591213a9",   # BBC World Service
    language="en",
    duration_seconds=300,
    output_dir="./output",
    llm={
        "provider": "azure_openai",
        "model": "<deployment-name>",                  # your Azure deployment name
        "base_url": "https://<resource>.openai.azure.com",
        "api_version": "2024-12-01-preview",
        "api_key": api_key,
    },
)
```

OpenAI API: `llm={"provider": "openai", "model": "gpt-6.1"}` with `$OPENAI_API_KEY` set.

How it works: the `anthropic` provider uses Claude structured outputs. The `openai`
and `azure_openai` providers use Chat Completions with a strict JSON-schema
`response_format`; set `structured_output="json_object"` for a deployment that does
not support JSON schemas. Every response is validated with Pydantic and gets one
repair retry. Other back ends implement the small `LLMProvider` protocol and are
registered with `listen_in.register_provider(name, factory)`. API keys are held as
`SecretStr` and never appear in logs, files, manifests or returned models.

### Two-station test

`examples/local_two_station_test.py` records France Info (French) and BBC World Service
(English), one after the other. The French bulletin is transcribed in French and
summarised in English, which shows the translation step:

```bash
cp .env.example .env    # add LLM_API_KEY / LLM_MODEL to enable analysis
python examples/local_two_station_test.py --duration 120 --model small --output-dir ./test-output
```

It prints the resolved stream URLs, the status of each stage and the output paths. The exit code is non-zero
if capture or transcription fails.

### Output layout

```text
output/
├── audio/        france-info_31074f8a_20261109T100000Z.wav
├── transcripts/  france-info_31074f8a_20261109T100000Z.json  (+ .txt original, .en.txt English)
├── analysis/     france-info_31074f8a_20261109T100000Z.json
└── manifests/    france-info_31074f8a_20261109T100000Z.json
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
%pip install "listen-in[whisper,llm-openai] @ git+https://github.com/HWDSIU/listen-in.git@v0.3.0"
dbutils.library.restartPython()
```

Pin scheduled jobs to a release tag so they keep running the same code when `main`
changes. To upgrade, change the tag.

Serverless compute cannot run init scripts. The supported route there is the
`ffmpeg` extra (`listen-in[whisper,llm-anthropic,ffmpeg]`), but it has not been
tested on serverless yet. Classic compute with the init script is the supported
MVP path. GPU compute (CUDA) makes Whisper much faster, and FP16 is used
automatically on CUDA.

Use `llm-anthropic` instead of (or as well as) `llm-openai` for Claude.

The cluster needs outbound HTTPS access to `*.api.radio-browser.info`, the stream
hosts, your LLM endpoint (your Azure OpenAI resource, `api.openai.com` or
`api.anthropic.com`) and, on first use, `openaipublic.azureedge.net` for the
Whisper weights.

### Secrets and Unity Catalog Volume output

```python
from listen_in import process_station

# Fetch the key however your workspace does it: a Databricks secret scope
# (including Key Vault-backed scopes) or Azure Key Vault directly (see above).
api_key = dbutils.secrets.get(scope="listen-in", key="llm-api-key")

result = process_station(
    station="31074f8a-e6f4-11e9-a96c-52543be04c81",   # France Info
    language="fr",
    duration_seconds=600,
    output_dir="/Volumes/main/radio/listen_in/outputs",
    whisper_model="small",
    llm={
        "provider": "azure_openai",
        "model": "<deployment-name>",
        "base_url": "https://<resource>.openai.azure.com",
        "api_key": api_key,
    },
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
| `FFmpegNotFoundError` | Install ffmpeg (see above), set `CaptureOptions(ffmpeg_path=...)` or `$LISTEN_IN_FFMPEG`, or reinstall with the `ffmpeg` extra (`listen-in[...,ffmpeg]`). |
| `RadioBrowserUnavailableError` | Outbound HTTPS to `*.api.radio-browser.info` is blocked. Allow it, or pass the stream URL directly. |
| `StreamUnavailableError` / "Recording too short" | The station is offline, geo-blocked or has changed its URL. Radio Browser records and stream URLs **change over time**; resolve again or choose another station. |
| `CERTIFICATE_VERIFY_FAILED` while downloading Whisper weights | python.org builds on macOS have no CA bundle until you run *Install Certificates.command*. listen-in falls back to `certifi` automatically; behind a corporate proxy, set `SSL_CERT_FILE` to your organisation's CA bundle. |
| Transcription is very slow | CPU Whisper is roughly real-time or slower for `small` on a laptop. Use a CUDA GPU, a smaller model (`base`, `tiny`) or shorter recordings. |
| Names or words are garbled | Smaller Whisper models are weaker on lower-resource languages, phone-ins and music beds. Try `turbo`, `medium` or `large-v3`, or pass `initial_prompt=` with expected names. The LLM records doubtful names under `uncertainties`. |
| Transcript is in the wrong language (e.g. English text for a German station) | The `language` hint was wrong, and Whisper translated into it. Since 0.3.0 a clearly wrong hint is replaced by the detected language (`transcript.language_mismatch` is `true`). Pass the right code or `language=None`. |
| `ConfigurationError: No API key` | Pass `api_key=` or set `LLM_API_KEY` (or the provider's own variable: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY`). |
| Azure: `could not find deployment ... (404)` | `model` must be the **deployment name**, `base_url` the resource endpoint (`https://<resource>.openai.azure.com`, no path), and `api_version` one your resource supports. |
| Azure/OpenAI: request rejected mentioning `response_format` / `json_schema` | The deployment does not support JSON-schema structured outputs; add `"structured_output": "json_object"` to `llm`. |
| `AnalysisRefusedError: ... content filter` | Azure's content filter blocked the transcript or the response. The transcript is still saved. |

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[all,dev]"
pytest                    # mocked: no network, no Whisper download, no paid API
ruff check src tests
RUN_LIVE_RADIO_TESTS=1 pytest -m integration   # opt-in: records 2 x 15 s of live radio
```

