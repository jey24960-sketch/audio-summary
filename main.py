"""
main.py - Pipeline entry point.

Usage
-----
    # Process all audio files in a folder
    python main.py --audio-dir ./lectures

    # Process a single file
    python main.py --audio-dir ./lectures --file lecture01.m4a

    # Skip Notion upload (useful for testing transcription + summarization)
    python main.py --audio-dir ./lectures --skip-notion

    # Resume a previously interrupted run
    python main.py --audio-dir ./lectures --cache-dir .cache

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
from utils import ProgressStore, discover_audio_files, ensure_dir, get_logger

load_dotenv()  # Load .env file if present (noop if not found)
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Single-file pipeline
# ---------------------------------------------------------------------------

def process_file(
    audio_path: Path,
    store: ProgressStore,
    uploader: NotionUploader | None,
) -> bool:
    """
    Run the full pipeline for one audio file.
    Returns True on success, False on failure.
    """
    filename = audio_path.name
    logger.info("=" * 60)
    logger.info("Processing: %s", filename)
    logger.info("=" * 60)

    try:
        # ---- Transcription ------------------------------------------
        transcriber = Transcriber(store)
        transcript  = transcriber.transcribe(audio_path)

        logger.info(
            "Transcript: %d characters / %.0f words",
            len(transcript),
            len(transcript.split()),
        )

        # Persist the transcript as a plain-text file alongside the audio
        transcript_path = audio_path.parent / (audio_path.stem + "_transcript.txt")
        transcript_path.write_text(transcript, encoding="utf-8")
        logger.info("Transcript saved → %s", transcript_path)

        # ---- Summarization ------------------------------------------
        summarizer = Summarizer(store)
        result     = summarizer.summarize(filename, transcript)
        logger.info("Summary generated.")

        # Persist summary as plain-text files for offline reference
        summary_dir = ensure_dir(audio_path.parent / "summaries")
        (summary_dir / f"{audio_path.stem}_overall.txt").write_text(
            result.overall_summary, encoding="utf-8"
        )
        (summary_dir / f"{audio_path.stem}_concepts.txt").write_text(
            result.key_concepts, encoding="utf-8"
        )
        (summary_dir / f"{audio_path.stem}_detailed.txt").write_text(
            result.detailed_notes, encoding="utf-8"
        )
        logger.info("Summary files saved → %s/", summary_dir)

        # ---- Notion upload ------------------------------------------
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
        logger.error("✗ %s — failed: %s", filename, exc)
        logger.debug(traceback.format_exc())
        store.update(filename, "status", "failed")
        store.update(filename, "error", str(exc))
        return False


# ---------------------------------------------------------------------------
# Batch pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    audio_dir: Path,
    cache_dir: Path,
    skip_notion: bool,
    target_file: str | None,
) -> None:
    store    = ProgressStore(cache_dir)
    uploader = None if skip_notion else NotionUploader()

    # Collect files to process
    if target_file:
        audio_files = [audio_dir / target_file]
        missing = [f for f in audio_files if not f.exists()]
        if missing:
            logger.error("File not found: %s", missing[0])
            sys.exit(1)
    else:
        audio_files = discover_audio_files(audio_dir)

    if not audio_files:
        logger.warning("No supported audio files found in '%s'.", audio_dir)
        sys.exit(0)

    logger.info("Found %d audio file(s) to process.", len(audio_files))

    results: dict[str, bool] = {}
    for audio_path in audio_files:
        success = process_file(audio_path, store, uploader)
        results[audio_path.name] = success

    # Summary report
    passed = sum(1 for v in results.values() if v)
    failed = len(results) - passed
    logger.info("Pipeline complete: %d succeeded, %d failed.", passed, failed)
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
    logger.info("Cache directory : %s", args.cache_dir)
    logger.info("Notion upload   : %s", "disabled" if args.skip_notion else "enabled")

    run_pipeline(
        audio_dir   = args.audio_dir,
        cache_dir   = args.cache_dir,
        skip_notion = args.skip_notion,
        target_file = args.file,
    )


if __name__ == "__main__":
    main()
