<p align="center">
  <img src="https://raw.githubusercontent.com/HWDSIU/listen-in/main/assets/listen-in-icon.svg" width="112" height="112" alt="listen-in logo: a broadcast symbol inside a speech bubble">
</p>

# listen-in

Record internet radio, transcribe it **locally** with
[OpenAI Whisper](https://github.com/openai/whisper), and turn each broadcast into
**structured English news analysis**, whatever language it's in. An LLM (Claude,
OpenAI, or a model deployed on Azure OpenAI / Azure AI Foundry) splits the bulletin
into separate stories. For each story it extracts the **people, organisations and
places** named, the **key claims** made, and anything **uncertain**, such as names
that may have been misheard. Non-English broadcasts also get a timestamped English
translation alongside the original-language transcript.

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

> **Status:** alpha (0.4.1). The API may change before 1.0.

[![tests](https://github.com/HWDSIU/listen-in/actions/workflows/test.yml/badge.svg)](https://github.com/HWDSIU/listen-in/actions/workflows/test.yml)

## What it does

- **Finds the station** from a direct stream URL, a [Radio Browser](https://www.radio-browser.info)
  UUID or a Radio Browser history URL, with automatic failover between API servers. It keeps
  the station's metadata: country, region, coordinates (with how precise they are),
  language codes and tags. Every run can therefore be placed on a map.
- **Records** the stream with `ffmpeg` to mono 16 kHz WAV (atomic writes, timeouts, no shell).
- **Transcribes** with Whisper on CPU or CUDA, keeping the **original-language** text and
  timestamped segments. It checks your language hint against the audio first, so a wrong
  hint can't turn the transcript into a translation.
- **Translates** non-English transcripts into English segment by segment, alongside the
  original (needs an LLM).
- **Analyses** each broadcast into stories with topics, the people, organisations and
  places named (with standardised names for matching across stations and languages), key
  claims, uncertainties, and where each story is in the recording (needs an LLM).
- **Saves** audio, transcripts, analysis and a per-run JSON **manifest** to a local folder
  or any mounted storage path.

## What it does not do

- It does not schedule anything. Call it from cron, Airflow or any job scheduler.
- It does not verify facts. The analysis covers only what the transcript says; it is not
  checked against the outside world.
- It does not bundle Whisper weights or (by default) ffmpeg.
- It does not start background threads or services.

## Responsible use and licensing

Radio broadcasts are copyrighted. Before recording, storing or processing a station,
make sure you are permitted to do so under the station's terms and the laws that
apply to you (copyright, broadcasting and data-protection law). Phone-ins and
interviews may contain personal data. Keep recordings only as long as you need them;
`process_station(..., keep_audio=False)` deletes the audio once the transcript is saved
(it is kept if transcription fails, so the run can be retried). Do not
redistribute captured audio or transcripts. Machine transcripts and LLM output
contain errors, so check anything important against the source audio.

The code is licensed under [Apache-2.0](LICENSE). Whisper and model weights carry their own licences.

## Installation

Python 3.10 or later. listen-in is **not yet published on PyPI**, so install a tagged
release from GitHub:

```bash
pip install "listen-in[whisper,llm-anthropic] @ git+https://github.com/HWDSIU/listen-in.git@v0.4.1"
```

Swap `llm-anthropic` for `llm-openai` to use OpenAI or Azure OpenAI / AI Foundry (or
include both). Pin to a release tag (see [tags](https://github.com/HWDSIU/listen-in/tags))
for reproducible installs, or use `@main` for the latest unreleased code. Once the
package is on PyPI this will become `pip install "listen-in[whisper,llm-anthropic]"`.

| Extra | Adds | Needed for |
|---|---|---|
| *(none)* | `httpx`, `pydantic` | station resolution, models |
| `whisper` | `openai-whisper` (+ PyTorch), `numpy`, `certifi` | transcription |
| `llm-anthropic` | `anthropic` | translation and analysis with Claude |
| `llm-openai` | `openai` | translation and analysis with OpenAI or Azure OpenAI / AI Foundry |
| `ffmpeg` | `imageio-ffmpeg` (static ffmpeg binary) | recording where you can't install ffmpeg |
| `all` | all of the above | |

Importing `listen_in` needs neither Whisper nor ffmpeg. They are checked when first
used, and the error explains how to install them.

### ffmpeg

`ffmpeg` is a separate program; pip can't reliably install it, except through the
`ffmpeg` extra.

| Platform | Command |
|---|---|
| macOS | `brew install ffmpeg` |
| Ubuntu / Debian | `sudo apt-get install -y ffmpeg` |
| Windows | `winget install Gyan.FFmpeg` (or `choco install ffmpeg`) |
| Anywhere (fallback) | add the `ffmpeg` extra, e.g. `listen-in[whisper,llm-anthropic,ffmpeg] @ git+https://github.com/HWDSIU/listen-in.git@v0.4.1` |

listen-in looks for ffmpeg in this order: `CaptureOptions(ffmpeg_path=...)`, then
`$LISTEN_IN_FFMPEG`, then `ffmpeg` on `PATH`, then the `imageio-ffmpeg` binary.

### Whisper weights

The first transcription downloads the model to `~/.cache/whisper`: about 460 MB for
`small` and 1.6 GB for `turbo`. Allow time and disk space, and outbound HTTPS access to
`openaipublic.azureedge.net`. Pass `download_root=` to `transcribe_audio` to use a
different cache.

## Quick start

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

Or end to end, without an LLM: `process_station(..., llm=None)`.

### Full pipeline

```python
import os
from listen_in import process_station

result = process_station(
    station="31074f8a-e6f4-11e9-a96c-52543be04c81",   # France Info
    language="fr",
    duration_seconds=600,
    output_dir="./output",
    whisper_model="small",
    llm={
        "provider": "anthropic",
        "model": "claude-haiku-5-5",
        "api_key": os.environ["LLM_API_KEY"],   # or omit: $LLM_API_KEY / $ANTHROPIC_API_KEY
    },
)

if result.ok:
    for story in result.analysis.analysis.stories:
        print(story.headline, story.topics, [e.canonical_name for e in story.entities])
else:
    print(result.stages)  # per-stage status, timing and non-secret error
```

`process_station` runs resolve → capture → transcribe → translate → analyse → persist
and returns a `ProcessingResult` with everything it produced, the file paths, and a
status and timing for each stage. A failed stage never deletes earlier artefacts: if
analysis fails, for example, the transcript is still saved. Set-up problems are caught
before recording starts and raise straight away: a missing Whisper or LLM package, an
unknown language code, Whisper model or LLM provider, a missing API key or Azure
endpoint. Each step can also be called on its own: `resolve_station`, `capture_stream`,
`transcribe_audio`, `translate_transcript` and `analyse_transcript`.

**Recording length.** listen-in is designed for snippets, such as a bulletin or an
interview, typically up to about 20 minutes. Each analysis is a single LLM response
covering every story, so much longer recordings of dense news (roughly 25 minutes or
more) can exceed the default output limit and fail with *"Response truncated at
max_tokens"*. The transcript and translation are still saved. Record shorter segments,
or raise the limit if your model allows it, with a longer timeout to match:
`llm={..., "max_tokens": 32000, "timeout_seconds": 300}`.

### Two-station example

`examples/local_two_station_test.py` records France Info (French) and BBC World Service
(English), one after the other. The French bulletin is transcribed in French, translated
and analysed in English:

```bash
cp .env.example .env    # add LLM_API_KEY / LLM_MODEL to enable translation and analysis
python examples/local_two_station_test.py --duration 120 --model small --output-dir ./test-output
```

It prints the resolved stream URLs, each stage's status and the output paths, and exits
non-zero if capture or transcription fails.

## Configuration

### LLM providers

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
        "model": "<deployment-name>",
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

Translation runs automatically for non-English broadcasts when an `llm` is configured
(one extra call per run); `translate=False` turns it off.

### Whisper model

Set with `whisper_model=` on `process_station`, `model=` on `transcribe_audio`, or
`--model` in the example script. The default is `small`.

| Model | Download | Use it for |
|---|---|---|
| `small` (default) | ~460 MB | English, French and most widely spoken languages. Accurate enough for news monitoring and runs faster than real time on a laptop CPU. |
| `turbo` (large-v3-turbo) | ~1.6 GB | Less widely spoken languages such as **Swahili**, where `small` misspells names, institutions and figures. Needs a GPU or a machine with 16 GB+ RAM; on an 8 GB laptop CPU it runs at roughly half real-time speed. |
| `medium`, `large-v3` | ~1.5 GB / ~2.9 GB | Alternatives when `turbo` is not good enough; slower. |
| `base`, `tiny` | ~140 MB / ~75 MB | Quick tests only. |

In a test on a 5-minute Swahili news bulletin, `turbo` transcribed officials' names,
institutions and figures (percentages, sums of money) correctly. `small` produced
phonetic approximations of the same passages and dropped the last 40 seconds, and the
analysis went from vague ("water company (name unclear)") to specific. For English and
French news, `small` was already clean.

### Language hints

`language=` takes an ISO 639-1 code (`"en"`, `"fr"`, `"de"`, `"sw"`...) or `None` to
detect it. Whisper treats its language setting as an instruction, not a hint: given
the wrong language, it quietly **translates** the speech into that language, and the
original wording is lost. So listen-in checks the hint first, running Whisper's
language detection over several 30-second windows of the recording. If detection is
confident (at least 50 %) and gives the hint almost no weight (under 10 %), the
recording is transcribed in the detected language and a warning is logged.
`on_language_mismatch="use_hint"` keeps your hint anyway, and `"error"` stops instead.
A correct hint is never overridden by short stretches of another language, such as
adverts or quotes in English.

Some languages sound the same, or nearly, and differ mainly in script or spelling, so
Whisper's detector can't tell them apart from audio: Hindi/Urdu, Serbian/Croatian/Bosnian,
Malay/Indonesian, Norwegian Bokmål/Nynorsk and Persian/Tajik. A hint in one of these
groups is never swapped for a sibling. For example, an Urdu station hinted `"ur"` stays in
Urdu script even if the detector leans towards Hindi. For these languages, always pass
the hint; without one, the detector's guess decides the script.

The transcript records what happened: `language` (what Whisper transcribed in),
`language_hint`, `detected_language`, `detected_language_probability` and
`language_mismatch`.

## Outputs

```text
output/
├── audio/        france-info_31074f8a_20261109T100000Z.wav
├── transcripts/  france-info_31074f8a_20261109T100000Z.json  (+ .txt original, .en.txt English)
├── analysis/     france-info_31074f8a_20261109T100000Z.json
└── manifests/    france-info_31074f8a_20261109T100000Z.json
```

File names are `<station-slug>_<uuid prefix>_<UTC start time>`. Existing files are
never overwritten unless you pass `overwrite=True`.

### The manifest

The manifest is the complete record of a run: station, recording, transcript,
translation, analysis, LLM settings (never the key) and each stage's status, timing and
errors. It is written even when a stage fails. Nothing in it is repeated, and file paths
in it are relative to the output folder, so the folder can be moved or copied. It leaves
out details of the machine and account it ran on: the output folder's location, the
ffmpeg path and the LLM endpoint URL (the provider, model or deployment name and API
version are kept). With `keep_audio=False`, `audio.deleted` is `true` and the audio paths
are `null`.
`language_hint` is the language you asked for (`null` = detect) and `language` the
language the transcript is in.

### The analysis file

Each `analysis/<run>.json` stands on its own: where and when (`run`), what was said
(`analysis`), and the original transcript with its English translation (`transcript`,
described below).

```json
{
  "run": {
    "run_id": "france-info_31074f8a_20261109T100000Z",
    "recorded_at": "2026-11-09T10:00:00Z", "duration_seconds": 600, "language": "fr",
    "station": {"name": "France Info", "country": "France", "countrycode": "FR",
                "subdivision": null, "latitude": 48.86, "longitude": 2.35,
                "location_precision": "country"}
  },
  "analysis": {
    "news_detected": true,
    "broadcast_language": "fr",
    "translation_used": true,
    "summary_en": "Concise English summary based only on the transcript.",
    "stories": [{
      "headline": "Unions and employers clash over wage rise",
      "summary": "...",
      "topics": ["economy", "labour"],
      "people": ["Patrick Martin"], "organisations": ["CFDT", "MDF"], "locations": ["France"],
      "entities": [
        {"type": "organisation", "name": "MDF", "canonical_name": "MEDEF", "role": "", "uncertain": true},
        {"type": "person", "name": "Patrick Martin", "canonical_name": "Patrick Martin", "role": "", "uncertain": false}
      ],
      "key_claims": ["..."],
      "uncertainties": ["'MDF' is likely a transcription of MEDEF."],
      "segment_ids": [3, 4, 5, 41, 42],
      "time_ranges": [{"start": 12.4, "end": 41.0}, {"start": 302.5, "end": 330.1}],
      "start": 12.4, "end": 330.1
    }]
  },
  "transcript": {"...": "original and English, see below"}
}
```

- **`entities`**: every person, organisation and place in the story. `name` is the
  spelling as heard. `canonical_name` is the standard form, e.g. a misheard acronym
  corrected or a name spelt in full, so the same entity can be matched across stories,
  stations and languages. The model may use general knowledge only to standardise names,
  never to add facts, and marks guesses with `uncertain: true`. `role` is filled only
  when the broadcast states it. The broadcasting station itself is never listed. The
  plain `people` / `organisations` / `locations` lists are kept for simple use.
- **`topics`**: one to three per story, from a fixed list: politics, elections, economy,
  business, labour, conflict, security, crime_justice, health, environment,
  science_technology, education, society, migration, culture, religion, sport, weather,
  transport, disaster, international, other.
- **`segment_ids`, `time_ranges`**: which transcript segments the story comes from, and
  so where it is in the recording (seconds). A story trailed in the headlines and covered
  later has one range per stretch. Use them to show the exact quote or play the clip.
  `start`/`end` are the first range's start and the last range's end, so they can span
  other stories. listen-in computes the times from the segments; the model only names
  the segments.
- `translation_used` is the model's own flag (did it translate to write this analysis);
  `transcript.translated` says whether listen-in produced a full English translation.

### Original transcript and English translation

The original-language transcript is always kept: `transcripts/<run>.txt`, plus
timestamped segments in `transcripts/<run>.json`. For non-English broadcasts the English
translation is saved as `transcripts/<run>.en.txt`, and the analysis file pairs the two
segment by segment:

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
`translated` is `false`. In Python the same data is in `result.translation` and
`result.analysis.transcript`. Translation keeps names as heard and marks unclear
passages `[unclear]`. To translate a transcript on its own, call
`translate_transcript(transcript, ...)` with the same provider settings as
`analyse_transcript`.

### Station location

Every station carries `latitude`, `longitude` and `location_precision`:

| `location_precision` | Coordinates come from |
|---|---|
| `station` | Radio Browser's coordinates for the station, or your own |
| `subdivision` | Your own, for a region (e.g. a state or province) |
| `country` | listen-in's reference point for the country (usually the capital), used when Radio Browser has no coordinates |
| `null` | Unknown country, or a direct stream URL |

Supply or correct a station's details with `station_overrides`. Coordinates given this
way count as `station` precision unless you say otherwise, and must be valid (latitude
−90 to 90, longitude −180 to 180):

```python
process_station("31074f8a-e6f4-11e9-a96c-52543be04c81", language="fr",
                station_overrides={"latitude": 48.853, "longitude": 2.278})
```

### Using the results

Manifests (or analysis files) load straight into pandas, a data warehouse or plain
Python. For example, stations on a map with their latest summary as hover text, and a
network of entities that are mentioned together:

```python
import collections, glob, itertools, json

runs = [json.load(open(p)) for p in glob.glob("output/manifests/*.json")]
analysed = [r for r in runs if r.get("analysis")]

# GeoJSON for any map library: one point per run, hover text from the analysis.
features = [{
    "type": "Feature",
    "geometry": {"type": "Point", "coordinates": [r["station"]["longitude"], r["station"]["latitude"]]},
    "properties": {
        "station": r["station"]["name"],
        "precision": r["station"]["location_precision"],
        "recorded_at": r["audio"]["started_at"],
        "summary": r["analysis"]["analysis"]["summary_en"],
        "headlines": [s["headline"] for s in r["analysis"]["analysis"]["stories"]],
    },
} for r in analysed if r["station"]["latitude"] is not None]

# Entity co-mention network: entities in the same story are linked.
edges = collections.Counter()
for r in analysed:
    for story in r["analysis"]["analysis"]["stories"]:
        nodes = sorted({(e["type"], e["canonical_name"]) for e in story["entities"]})
        edges.update(itertools.combinations(nodes, 2))
```

Use `canonical_name` to merge spellings, `uncertain` to filter or down-weight guesses,
`topics` to filter, and `time_ranges` to link a story back to the audio.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FFmpegNotFoundError` | Install ffmpeg (see above), set `CaptureOptions(ffmpeg_path=...)` or `$LISTEN_IN_FFMPEG`, or reinstall with the `ffmpeg` extra (`listen-in[...,ffmpeg]`). |
| `RadioBrowserUnavailableError` | Outbound HTTPS to `*.api.radio-browser.info` is blocked. Allow it, or pass the stream URL directly. |
| `StreamUnavailableError` / "Recording too short" | The station is offline, geo-blocked or has changed its URL. Radio Browser records and stream URLs **change over time**; resolve again or choose another station. |
| `CERTIFICATE_VERIFY_FAILED` while downloading Whisper weights | python.org builds on macOS have no CA bundle until you run *Install Certificates.command*. listen-in falls back to `certifi` automatically; behind a corporate proxy, set `SSL_CERT_FILE` to your organisation's CA bundle. |
| Transcription is very slow | CPU Whisper is roughly real time or slower for `small` on a laptop. Use a CUDA GPU, a smaller model (`base`, `tiny`) or shorter recordings. |
| Names or words are garbled | Smaller Whisper models are weaker on less widely spoken languages, phone-ins and music beds. Try `turbo`, `medium` or `large-v3`, or pass `initial_prompt=` with expected names. The LLM records doubtful names under `uncertainties` and marks uncertain entities. |
| Transcript is in the wrong language (e.g. English text for a German station) | The `language` hint was wrong and Whisper translated into it. Since 0.3.0 a clearly wrong hint is replaced by the detected language (`transcript.language_mismatch` is `true`). Pass the right code or `language=None`. |
| A station has no location on the map | Radio Browser had no coordinates and no known country. Pass `station_overrides={"latitude": ..., "longitude": ...}`. |
| `AnalysisError: Response truncated at max_tokens` | The recording is too long for one analysis response (see *Recording length*). Record shorter segments, or raise `max_tokens` together with `timeout_seconds`. The transcript and translation are still saved. |
| A correct hint for Urdu, Serbian, Malay... came out in a sibling language | Since 0.4.1, hints in closely related groups are kept. Without a hint, the detector's guess decides, so pass the hint. |
| `ConfigurationError: No API key` | Pass `api_key=` or set `LLM_API_KEY` (or the provider's own variable: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY`). |
| Azure: `could not find deployment ... (404)` | `model` must be the **deployment name**, `base_url` the resource endpoint (`https://<resource>.openai.azure.com`, no path), and `api_version` one your resource supports. |
| Azure/OpenAI: request rejected mentioning `response_format` / `json_schema` | The deployment does not support JSON-schema structured outputs; add `"structured_output": "json_object"` to `llm`. |
| `AnalysisRefusedError: ... content filter` | Azure's content filter blocked the transcript or the response. The transcript is still saved. |

## Development

To work on listen-in itself:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[all,dev]"
pytest                    # mocked: no network, no Whisper download, no paid API
ruff check src tests
RUN_LIVE_RADIO_TESTS=1 pytest -m integration   # opt-in: records 2 x 15 s of live radio
```

GitHub Actions runs the same checks on every push.
