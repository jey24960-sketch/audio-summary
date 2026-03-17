"""
utils.py - Shared utilities: logging, retry logic, file I/O helpers, cost tracking.
"""

import os
import re
import json
import time
import logging
import functools
import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a logger with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    return logger


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def retry(
    max_attempts: int = 3,
    initial_delay: float = 2.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator that retries a function on failure with exponential backoff.

    Args:
        max_attempts: Total number of attempts (including the first call).
        initial_delay: Seconds to wait before the first retry.
        backoff: Multiplier applied to delay on each subsequent retry.
        exceptions: Tuple of exception types that trigger a retry.
    """
    def decorator(func: F) -> F:
        logger = get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        logger.error(
                            "Function '%s' failed after %d attempts: %s",
                            func.__name__, max_attempts, exc,
                        )
                        raise
                    logger.warning(
                        "Attempt %d/%d for '%s' failed: %s — retrying in %.1fs …",
                        attempt, max_attempts, func.__name__, exc, delay,
                    )
                    time.sleep(delay)
                    delay *= backoff
            raise last_exc  # unreachable, but keeps type-checkers happy

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Intermediate result persistence
# ---------------------------------------------------------------------------

class ProgressStore:
    """
    Lightweight JSON-backed store for saving/resuming intermediate results.

    Each audio file gets its own JSON file under `cache_dir`.
    Keys are stage names (e.g. "transcript", "chunk_summaries", "final_summary").
    """

    def __init__(self, cache_dir: str | Path = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._logger = get_logger(__name__)

    def _path(self, audio_filename: str) -> Path:
        safe_name = Path(audio_filename).stem
        return self.cache_dir / f"{safe_name}.json"

    def load(self, audio_filename: str) -> dict:
        path = self._path(audio_filename)
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._logger.info("Loaded cached progress for '%s'", audio_filename)
            return data
        return {}

    def save(self, audio_filename: str, data: dict) -> None:
        path = self._path(audio_filename)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        self._logger.debug("Saved progress for '%s' → %s", audio_filename, path)

    def update(self, audio_filename: str, key: str, value: Any) -> None:
        data = self.load(audio_filename)
        data[key] = value
        self.save(audio_filename, data)

    def get(self, audio_filename: str, key: str, default: Any = None) -> Any:
        return self.load(audio_filename).get(key, default)

    def has(self, audio_filename: str, key: str) -> bool:
        return key in self.load(audio_filename)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".aac", ".m4a", ".wav", ".mp3", ".ogg", ".flac"}


def discover_audio_files(folder: str | Path) -> list[Path]:
    """Return all supported audio files in *folder* sorted by name."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Audio folder not found: {folder}")

    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return files


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_env(key: str, required: bool = True) -> str:
    """Read an environment variable, raising a clear error if missing."""
    value = os.getenv(key, "").strip()
    if required and not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            "Please add it to your .env file or export it before running."
        )
    return value


# ---------------------------------------------------------------------------
# Sentence splitting (shared by transcriber and summarizer)
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """
    Split *text* into individual sentences, handling Korean (。) and
    Western (.!?) punctuation.  Keeps sentence-final punctuation attached.
    """
    # Split on punctuation that ends a sentence, followed by whitespace
    parts = re.split(r"(?<=[.!?。])\s+", text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

# Pricing constants (USD) — update if OpenAI changes rates
_WHISPER_COST_PER_MINUTE   = 0.006          # $0.006 / min
_GPT4O_INPUT_PER_TOKEN     = 2.50 / 1e6    # $2.50  / 1M input tokens
_GPT4O_OUTPUT_PER_TOKEN    = 10.00 / 1e6   # $10.00 / 1M output tokens


class CostTracker:
    """
    Accumulates API costs per audio file and persists them to a JSON log.

    Usage
    -----
        tracker = CostTracker(Path("outputs/logs/costs.json"))
        tracker.record_transcription("lecture.m4a", duration_seconds=3600)
        tracker.record_summarization("lecture.m4a", input_tokens=5000, output_tokens=800)
        tracker.save()
    """

    def __init__(self, log_path: Path):
        self._log_path = log_path
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = get_logger(__name__)
        # Load existing records so incremental runs accumulate correctly
        self._records: dict = self._load()

    def _load(self) -> dict:
        if self._log_path.exists():
            with open(self._log_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    def record_transcription(self, filename: str, duration_seconds: float) -> None:
        cost = (duration_seconds / 60.0) * _WHISPER_COST_PER_MINUTE
        rec  = self._records.setdefault(filename, {})
        rec["transcription"] = {
            "duration_minutes": round(duration_seconds / 60, 2),
            "cost_usd":         round(cost, 6),
        }
        self._logger.info(
            "[cost] %s transcription: %.1f min → $%.4f",
            filename, duration_seconds / 60, cost,
        )

    def record_summarization(
        self, filename: str, input_tokens: int, output_tokens: int
    ) -> None:
        cost = (
            input_tokens  * _GPT4O_INPUT_PER_TOKEN
            + output_tokens * _GPT4O_OUTPUT_PER_TOKEN
        )
        rec = self._records.setdefault(filename, {})
        rec["summarization"] = {
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "cost_usd":      round(cost, 6),
        }
        self._logger.info(
            "[cost] %s summarization: %d in / %d out tokens → $%.4f",
            filename, input_tokens, output_tokens, cost,
        )

    def record_total(self, filename: str) -> None:
        """Compute and log the combined cost for one file."""
        rec = self._records.get(filename, {})
        total = sum(
            v.get("cost_usd", 0)
            for v in rec.values()
            if isinstance(v, dict)
        )
        rec["total_cost_usd"] = round(total, 6)
        rec["processed_at"]   = datetime.datetime.now().isoformat(timespec="seconds")
        self._logger.info("[cost] %s TOTAL → $%.4f", filename, total)

    def save(self) -> None:
        with open(self._log_path, "w", encoding="utf-8") as fh:
            json.dump(self._records, fh, ensure_ascii=False, indent=2)
        self._logger.debug("Cost log saved → %s", self._log_path)
