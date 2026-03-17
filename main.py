"""
main.py - Pipeline entry point.

Upgrades over v1
----------------
* [5] Structured output tree: outputs/transcripts/, outputs/summaries/, outputs/logs/
* [4] Cost tracking: reads duration + token usage from cache, logs to
      outputs/logs/costs.json via CostTracker.
* [7] Files already marked status=done are skipped automatically —
      prevents re-processing in incremental / scheduled runs.

Usage
-----
    # Process all audio files in a folder
    python main.py --audio-dir ./lectures

    # Process a single file
    python main.py --audio-dir ./lectures --file lecture01.m4a

    # Skip Notion upload (useful for testing)
    python main.py --audio-dir ./lectures --skip-notion

    # Custom output and cache directories
    python main.py --audio-dir ./lectures --output-dir ./outputs --cache-dir .cache

Environment variables (see .env.example)
-----------------------------------------
    OPENAI_API_KEY          Required
    NOTION_TOKEN            Required (unless --skip-notion)
    NOTION_PARENT_PAGE_ID   Required (unless --skip-notion)
    LOG_LEVEL               Optional (default: INFO)
"""

import argparse
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

from notion_uploader import NotionUploader
from summarizer import Summarizer
from transcriber import Transcriber
from utils import CostTracker, ProgressStore, discover_audio_files, ensure_dir, get_logger

load_dotenv()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Output directory layout  [upgrade #5]
# ---------------------------------------------------------------------------

def _make_output_dirs(output_dir: Path) -> tuple[Path, Path, Path]:
    """Create and return (transcripts_dir, summaries_dir, logs_dir)."""
    transcripts_dir = ensure_dir(output_dir / "transcripts")
    summaries_dir   = ensure_dir(output_dir / "summaries")
    logs_dir        = ensure_dir(output_dir / "logs")
    return transcripts_dir, summaries_dir, logs_dir


# ---------------------------------------------------------------------------
# Single-file pipeline
# ---------------------------------------------------------------------------

def process_file(
    audio_path:      Path,
    store:           ProgressStore,
    cost_tracker:    CostTracker,
    transcripts_dir: Path,
    summaries_dir:   Path,
    uploader:        NotionUploader | None,
) -> bool:
    """
    Run the full pipeline for one audio file.
    Returns True on success, False on failure.
    """
    filename = audio_path.name
    stem     = audio_path.stem

    # [upgrade #7] Skip files that have already completed successfully
    if store.get(filename, "status") == "done":
        logger.info("Skipping '%s' — already completed.", filename)
        return True

    logger.info("=" * 60)
    logger.info("Processing: %s", filename)
    logger.info("=" * 60)

    try:
        # ---- Transcription ------------------------------------------
        logger.info("[Stage 1/3] Transcription …")
        transcriber = Transcriber(store)
        transcript  = transcriber.transcribe(audio_path)

        logger.info(
            "Transcript: %d characters / %.0f words",
            len(transcript),
            len(transcript.split()),
        )

        # Save transcript to outputs/transcripts/
        transcript_path = transcripts_dir / f"{stem}_transcript.txt"
        transcript_path.write_text(transcript, encoding="utf-8")
        logger.info("Transcript saved → %s", transcript_path)

        # Record transcription cost [upgrade #4]
        duration = store.get(filename, "audio_duration_seconds") or 0
        cost_tracker.record_transcription(filename, duration)

        # ---- Summarization ------------------------------------------
        logger.info("[Stage 2/3] Summarization …")
        summarizer = Summarizer(store)
        result     = summarizer.summarize(filename, transcript)
        logger.info("Summary generated.")

        # Save summaries to outputs/summaries/
        (summaries_dir / f"{stem}_overall.txt").write_text(
            result.overall_summary, encoding="utf-8"
        )
        (summaries_dir / f"{stem}_concepts.txt").write_text(
            result.key_concepts, encoding="utf-8"
        )
        (summaries_dir / f"{stem}_detailed.txt").write_text(
            result.detailed_notes, encoding="utf-8"
        )
        logger.info("Summary files saved → %s/", summaries_dir)

        # Record summarization cost [upgrade #4]
        usage = store.get(filename, "summarization_tokens") or {}
        cost_tracker.record_summarization(
            filename,
            input_tokens  = usage.get("input",  0),
            output_tokens = usage.get("output", 0),
        )
        cost_tracker.record_total(filename)
        cost_tracker.save()

        # ---- Notion upload ------------------------------------------
        logger.info("[Stage 3/3] Notion upload …")
        if uploader is not None:
            if store.has(filename, "notion_page_url"):
                logger.info(
                    "Notion page already uploaded: %s",
                    store.get(filename, "notion_page_url"),
                )
            else:
                url = uploader.upload(filename, result)
                store.update(filename, "notion_page_url", url)
                logger.info("Uploaded to Notion: %s", url)
        else:
            logger.info("Notion upload skipped (--skip-notion).")

        store.update(filename, "status", "done")
        logger.info("✓ %s — done.\n", filename)
        return True

    except Exception as exc:  # noqa: BLE001
        logger.error("✗ [%s] Stage failed: %s", filename, exc)
        logger.debug(traceback.format_exc())
        store.update(filename, "status", "failed")
        store.update(filename, "error", str(exc))
        return False


# ---------------------------------------------------------------------------
# Batch pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    audio_dir:  Path,
    output_dir: Path,
    cache_dir:  Path,
    skip_notion: bool,
    target_file: str | None,
) -> None:
    transcripts_dir, summaries_dir, logs_dir = _make_output_dirs(output_dir)

    store        = ProgressStore(cache_dir)
    cost_tracker = CostTracker(logs_dir / "costs.json")
    uploader     = None if skip_notion else NotionUploader()

    # Collect files to process
    if target_file:
        audio_files = [audio_dir / target_file]
        if not audio_files[0].exists():
            logger.error("File not found: %s", audio_files[0])
            sys.exit(1)
    else:
        audio_files = discover_audio_files(audio_dir)

    if not audio_files:
        logger.warning("No supported audio files found in '%s'.", audio_dir)
        sys.exit(0)

    logger.info("Found %d audio file(s) to process.", len(audio_files))

    results: dict[str, bool] = {}
    for audio_path in audio_files:
        success = process_file(
            audio_path      = audio_path,
            store           = store,
            cost_tracker    = cost_tracker,
            transcripts_dir = transcripts_dir,
            summaries_dir   = summaries_dir,
            uploader        = uploader,
        )
        results[audio_path.name] = success

    # Final summary
    passed = sum(1 for v in results.values() if v)
    failed = len(results) - passed
    logger.info("Pipeline complete: %d succeeded, %d failed.", passed, failed)
    logger.info("Outputs saved to: %s/", output_dir)
    logger.info("Cost log:         %s", logs_dir / "costs.json")

    if failed:
        logger.error("Failed files:")
        for name, ok in results.items():
            if not ok:
                logger.error("  - %s", name)
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audio lecture → transcript → summary → Notion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Folder containing audio files (AAC / M4A / WAV).",
    )
    parser.add_argument(
        "--file",
        metavar="FILENAME",
        default=None,
        help="Process a single file inside --audio-dir instead of all files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Root output directory (default: ./outputs).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache"),
        help="Directory for intermediate result caching (default: .cache).",
    )
    parser.add_argument(
        "--skip-notion",
        action="store_true",
        help="Skip Notion upload (useful for testing).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Audio directory : %s", args.audio_dir)
    logger.info("Output directory: %s", args.output_dir)
    logger.info("Cache directory : %s", args.cache_dir)
    logger.info("Notion upload   : %s", "disabled" if args.skip_notion else "enabled")

    run_pipeline(
        audio_dir   = args.audio_dir,
        output_dir  = args.output_dir,
        cache_dir   = args.cache_dir,
        skip_notion = args.skip_notion,
        target_file = args.file,
    )


if __name__ == "__main__":
    main()
