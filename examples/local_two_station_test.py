"""Record, transcribe and (optionally) summarise one Swahili and one English station.

    python examples/local_two_station_test.py --duration 120 --model small --output-dir ./test-output

The LLM step runs only when both LLM_API_KEY and LLM_MODEL are set (a ./.env file
in the current directory is read if present). Stations are processed one after the
other so only one Whisper model is in memory. Exit code is non-zero if capture or
transcription fails for either station.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from listen_in import ListenInError, process_station, resolve_station

SWAHILI_STATION = "https://www.radio-browser.info/history/fc939579-455a-4593-9e0d-ee9df38f5c2f"
ENGLISH_STATION = "https://www.radio-browser.info/history/cf0316fa-091e-4dc4-92a7-d8b0bd0cad28"

STATIONS = [
    ("Swahili", SWAHILI_STATION, "sw"),
    ("English", ENGLISH_STATION, "en"),
]


def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env reader (KEY=VALUE lines); existing environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main() -> int:
    ap = argparse.ArgumentParser(description="listen-in two-station test")
    ap.add_argument("--duration", type=int, default=60, help="seconds to record per station")
    ap.add_argument("--model", default="small", help="Whisper model name")
    ap.add_argument("--output-dir", default="./test-output")
    ap.add_argument("--device", default="auto", help="auto, cpu or cuda")
    ap.add_argument("--no-llm", action="store_true", help="skip the LLM step even if configured")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()

    llm = None
    if not args.no_llm and os.environ.get("LLM_API_KEY") and os.environ.get("LLM_MODEL"):
        llm = {
            "provider": os.environ.get("LLM_PROVIDER", "anthropic"),
            "model": os.environ["LLM_MODEL"],
            "base_url": os.environ.get("LLM_BASE_URL") or None,
            # api_key is read from $LLM_API_KEY by the package; never printed.
        }
    print(f"LLM analysis: {'on (' + llm['model'] + ')' if llm else 'off (set LLM_API_KEY and LLM_MODEL)'}")

    print("\nResolving stations via Radio Browser...")
    for label, ref, _ in STATIONS:
        try:
            st = resolve_station(ref)
            print(f"  {label:8} {st.name}  ->  {st.stream_url}")
        except ListenInError as exc:
            print(f"  {label:8} FAILED to resolve: {exc}")

    exit_code = 0
    for label, ref, lang in STATIONS:
        print(f"\n[{label}] recording {args.duration}s, Whisper '{args.model}', language={lang} ...")
        result = process_station(
            ref,
            language=lang,
            duration_seconds=args.duration,
            output_dir=args.output_dir,
            whisper_model=args.model,
            llm=llm,
            device=args.device,
        )
        for stage, info in result.stages.items():
            secs = f"{info.seconds:.1f}s" if info.seconds is not None else ""
            err = f"  {info.error}" if info.error else ""
            print(f"  {stage:10} {info.status:8} {secs}{err}")
        for name in ("audio_path", "transcript_path", "analysis_path", "manifest_path"):
            value = getattr(result, name)
            if value:
                print(f"  {name:16} {value}")
        if result.analysis:
            a = result.analysis.analysis
            print(f"  summary_en       {a.summary_en}")
            for s in a.stories:
                print(f"    - {s.headline}")
        if result.statuses["capture"] != "ok" or result.statuses["transcribe"] != "ok":
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
