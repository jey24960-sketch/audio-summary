"""
audio_processor.py - Audio file handling, format conversion, and chunking.

Strategy
--------
* pydub is used for splitting; it delegates to ffmpeg/avconv under the hood,
  so it works with AAC, M4A, WAV and most other container formats.
* Files are split into overlapping chunks that stay below the Whisper API
  25 MB hard limit.  A 20 MB target + 30-second overlap keeps context at
  chunk boundaries without wasting quota.
* Chunks are exported as WAV (lossless, widely supported) to a temp directory
  that is cleaned up automatically.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterator

from pydub import AudioSegment

from utils import get_logger, retry

logger = get_logger(__name__)

# Whisper API hard limit is 25 MB; we stay well below it.
MAX_CHUNK_BYTES: int = int(os.getenv("MAX_CHUNK_BYTES", 20 * 1024 * 1024))  # 20 MB

# Overlap in milliseconds to avoid losing words at chunk boundaries.
CHUNK_OVERLAP_MS: int = int(os.getenv("CHUNK_OVERLAP_MS", 30_000))  # 30 s

# Export format for chunks sent to the API.
CHUNK_FORMAT: str = "wav"
CHUNK_SUFFIX: str = ".wav"


class AudioChunk:
    """Represents a single chunk ready for transcription."""

    def __init__(self, path: Path, index: int, start_ms: int, end_ms: int):
        self.path = path
        self.index = index
        self.start_ms = start_ms
        self.end_ms = end_ms

    def __repr__(self) -> str:
        return (
            f"AudioChunk(index={self.index}, "
            f"start={self.start_ms / 1000:.1f}s, "
            f"end={self.end_ms / 1000:.1f}s, "
            f"path={self.path.name})"
        )


class AudioProcessor:
    """
    Loads an audio file, optionally converts it, and yields fixed-size chunks.
    The caller owns the lifecycle of the temporary directory; use as a context
    manager to ensure cleanup.

        with AudioProcessor(path) as processor:
            for chunk in processor.chunks():
                transcribe(chunk)
    """

    def __init__(self, audio_path: str | Path):
        self.audio_path = Path(audio_path)
        if not self.audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {self.audio_path}")

        self._tmpdir: tempfile.TemporaryDirectory | None = None
        self._tmpdir_path: Path | None = None
        self._segment: AudioSegment | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "AudioProcessor":
        self._tmpdir = tempfile.TemporaryDirectory(prefix="audio_chunks_")
        self._tmpdir_path = Path(self._tmpdir.name)
        logger.info("Temporary chunk directory: %s", self._tmpdir_path)
        self._load()
        return self

    def __exit__(self, *_) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            logger.info("Cleaned up temporary chunk directory.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @retry(max_attempts=3, initial_delay=2.0)
    def _load(self) -> None:
        """Load the audio file via pydub (delegates to ffmpeg)."""
        suffix = self.audio_path.suffix.lower().lstrip(".")
        # pydub treats m4a as mp4 container
        fmt = "mp4" if suffix in ("m4a", "aac") else suffix
        logger.info(
            "Loading audio file '%s' (format hint: %s) …",
            self.audio_path.name, fmt,
        )
        self._segment = AudioSegment.from_file(str(self.audio_path), format=fmt)
        duration_min = len(self._segment) / 60_000
        logger.info(
            "Loaded %.1f minutes of audio (%.1f MB on disk).",
            duration_min,
            self.audio_path.stat().st_size / 1024 / 1024,
        )

    def _estimate_wav_bytes(self, segment: AudioSegment) -> int:
        """Rough WAV size estimate: sample_rate × channels × 2 bytes × seconds + header."""
        seconds = len(segment) / 1000
        return int(segment.frame_rate * segment.channels * 2 * seconds) + 44

    def _split_points(self) -> list[tuple[int, int]]:
        """
        Return (start_ms, end_ms) pairs that respect MAX_CHUNK_BYTES.

        We binary-search for the largest duration that fits within the limit
        while keeping a minimum chunk of 10 s so we never loop infinitely.
        """
        seg = self._segment
        total_ms = len(seg)
        MIN_CHUNK_MS = 10_000  # 10 s minimum to avoid degenerate splits

        points: list[tuple[int, int]] = []
        cursor = 0

        while cursor < total_ms:
            # Start with the full remaining duration and halve until it fits.
            remaining_ms = total_ms - cursor
            candidate_ms = remaining_ms

            while candidate_ms > MIN_CHUNK_MS:
                estimated = self._estimate_wav_bytes(seg[cursor: cursor + candidate_ms])
                if estimated <= MAX_CHUNK_BYTES:
                    break
                candidate_ms //= 2

            end_ms = min(cursor + candidate_ms, total_ms)
            points.append((cursor, end_ms))

            # Advance, stepping back by overlap so context is preserved.
            next_cursor = end_ms - CHUNK_OVERLAP_MS
            if next_cursor <= cursor:
                # Safety: always make forward progress
                next_cursor = cursor + max(MIN_CHUNK_MS, candidate_ms // 2)
            cursor = min(next_cursor, total_ms)

        return points

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunks(self) -> Iterator[AudioChunk]:
        """
        Yield AudioChunk objects one at a time.
        Each chunk is written to the temp directory as a WAV file.
        """
        if self._segment is None:
            raise RuntimeError("Call AudioProcessor inside a 'with' block first.")

        split_points = self._split_points()
        total = len(split_points)
        logger.info("Splitting into %d chunk(s) …", total)

        for idx, (start_ms, end_ms) in enumerate(split_points, start=1):
            slice_seg = self._segment[start_ms:end_ms]
            chunk_path = self._tmpdir_path / f"chunk_{idx:04d}{CHUNK_SUFFIX}"

            logger.info(
                "Exporting chunk %d/%d  [%.1fs – %.1fs] → %s",
                idx, total,
                start_ms / 1000, end_ms / 1000,
                chunk_path.name,
            )
            slice_seg.export(str(chunk_path), format=CHUNK_FORMAT)
            yield AudioChunk(
                path=chunk_path,
                index=idx,
                start_ms=start_ms,
                end_ms=end_ms,
            )

    @property
    def duration_seconds(self) -> float:
        if self._segment is None:
            raise RuntimeError("Audio not loaded yet.")
        return len(self._segment) / 1000
