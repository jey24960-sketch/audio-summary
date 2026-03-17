"""
audio_processor.py - ffmpeg-based audio chunking for the Whisper API.

Replaces the previous pydub implementation entirely.  Uses only the
system `ffmpeg` / `ffprobe` binaries via subprocess — no pydub, no
audioop, no pyaudioop.

Chunking strategy
-----------------
* Audio duration is read with ffprobe.
* The file is divided into fixed-length windows (default 10 minutes) with
  a 30-second overlap so no speech is lost at chunk boundaries.
* Each chunk is re-encoded as mono 16 kHz MP3 at 32 kbps — roughly
  2.4 MB per 10-minute chunk, safely under Whisper's 25 MB hard limit.
* Chunks are written to a TemporaryDirectory that is cleaned up when the
  context manager exits.
* `split_points()` returns the planned time windows WITHOUT extracting
  anything, so `transcriber.py` can skip extraction for already-cached
  chunks on a resumed run.
* `extract_chunk()` extracts exactly one chunk on demand.
* Chunk files are small enough that the caller can delete them
  immediately after transcription to conserve disk space.

Compatibility
-------------
Works on any system where `ffmpeg` and `ffprobe` are on PATH:
  - Google Colab: apt-get install -y ffmpeg
  - Streamlit Cloud: packages.txt → ffmpeg
  - Linux / macOS dev machines: package-manager install
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterator

from utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tuneable constants (override via environment variables if needed)
# ---------------------------------------------------------------------------

# Duration of each audio chunk (seconds).  10 min @ 32 kbps ≈ 2.4 MB.
CHUNK_DURATION_S: int = int(os.getenv("CHUNK_DURATION_S", 600))

# Overlap between consecutive chunks (seconds).  Prevents words being
# cut at the boundary; the transcriber deduplicates the repeated text.
CHUNK_OVERLAP_S: int = int(os.getenv("CHUNK_OVERLAP_S", 30))

# ffmpeg output encoding — mono 16 kHz MP3 is ideal for speech-to-text
_BITRATE     = "32k"
_CHANNELS    = "1"       # mono
_SAMPLE_RATE = "16000"   # Hz — Whisper's native sample rate
_CHUNK_EXT   = ".mp3"


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

class AudioChunk:
    """A single extracted chunk ready for the Whisper API."""

    def __init__(self, path: Path, index: int, start_s: float, end_s: float):
        self.path     = path
        self.index    = index
        self.start_s  = start_s
        self.end_s    = end_s
        # ms aliases kept for backward compatibility with logging in transcriber.py
        self.start_ms = int(start_s * 1000)
        self.end_ms   = int(end_s   * 1000)

    def __repr__(self) -> str:
        return (
            f"AudioChunk(index={self.index}, "
            f"start={self.start_s:.1f}s, end={self.end_s:.1f}s, "
            f"path={self.path.name})"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AudioProcessor:
    """
    ffmpeg-based audio processor.

    Usage
    -----
        with AudioProcessor(path) as proc:
            for idx, (start_s, end_s) in enumerate(proc.split_points(), start=1):
                chunk = proc.extract_chunk(idx, start_s, end_s)
                transcribe(chunk)
                chunk.path.unlink(missing_ok=True)  # free disk immediately

    Or iterate all chunks in one go:
        with AudioProcessor(path) as proc:
            for chunk in proc.chunks():
                transcribe(chunk)
    """

    def __init__(self, audio_path: str | Path):
        self.audio_path = Path(audio_path)
        if not self.audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {self.audio_path}")

        self._tmpdir:      tempfile.TemporaryDirectory | None = None
        self._tmpdir_path: Path  | None = None
        self._duration_s:  float | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "AudioProcessor":
        _require_ffmpeg()   # fail fast with a clear message if not installed
        self._tmpdir      = tempfile.TemporaryDirectory(prefix="audio_chunks_")
        self._tmpdir_path = Path(self._tmpdir.name)
        self._duration_s  = _probe_duration(self.audio_path)
        logger.info(
            "Audio loaded: '%s'  %.1f min  %.1f MB",
            self.audio_path.name,
            self._duration_s / 60,
            self.audio_path.stat().st_size / 1_048_576,
        )
        return self

    def __exit__(self, *_) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            logger.debug("Temporary chunk directory cleaned up.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def duration_seconds(self) -> float:
        if self._duration_s is None:
            raise RuntimeError("Use AudioProcessor inside a 'with' block.")
        return self._duration_s

    @property
    def chunk_count(self) -> int:
        """Total number of chunks that will be produced (no extraction)."""
        return len(self.split_points())

    # ------------------------------------------------------------------
    # Public: split planning (no I/O)
    # ------------------------------------------------------------------

    def split_points(self) -> list[tuple[float, float]]:
        """
        Return the planned (start_s, end_s) windows without extracting
        any audio.  Use this to decide which chunks need transcription
        before calling extract_chunk().
        """
        if self._duration_s is None:
            raise RuntimeError("Use AudioProcessor inside a 'with' block.")
        return _calc_split_points(self._duration_s, CHUNK_DURATION_S, CHUNK_OVERLAP_S)

    # ------------------------------------------------------------------
    # Public: on-demand extraction
    # ------------------------------------------------------------------

    def extract_chunk(self, index: int, start_s: float, end_s: float) -> AudioChunk:
        """
        Extract one chunk from the source file using ffmpeg and return an
        AudioChunk pointing to the written file.

        The caller is responsible for deleting chunk.path when done.
        """
        if self._tmpdir_path is None:
            raise RuntimeError("Use AudioProcessor inside a 'with' block.")

        chunk_path = self._tmpdir_path / f"chunk_{index:04d}{_CHUNK_EXT}"
        duration_s = end_s - start_s

        logger.info(
            "Extracting chunk %d [%.1fs – %.1fs, %.0fs] → %s",
            index, start_s, end_s, duration_s, chunk_path.name,
        )
        _ffmpeg_extract(self.audio_path, start_s, duration_s, chunk_path)
        return AudioChunk(path=chunk_path, index=index, start_s=start_s, end_s=end_s)

    # ------------------------------------------------------------------
    # Public: convenience full iterator
    # ------------------------------------------------------------------

    def chunks(self) -> Iterator[AudioChunk]:
        """
        Yield all chunks in order, extracting each one lazily.
        For fine-grained resume control use split_points() + extract_chunk().
        """
        for idx, (start_s, end_s) in enumerate(self.split_points(), start=1):
            yield self.extract_chunk(idx, start_s, end_s)


# ---------------------------------------------------------------------------
# ffmpeg / ffprobe helpers
# ---------------------------------------------------------------------------

def _require_ffmpeg() -> None:
    """Raise a clear RuntimeError if ffmpeg or ffprobe are not on PATH."""
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        raise RuntimeError(
            f"Required tool(s) not found on PATH: {', '.join(missing)}.\n"
            "Install ffmpeg:\n"
            "  Linux / Colab : apt-get install -y ffmpeg\n"
            "  macOS         : brew install ffmpeg\n"
            "  Streamlit Cloud: add 'ffmpeg' to packages.txt"
        )


def _probe_duration(audio_path: Path) -> float:
    """Return audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return float(result.stdout.strip())
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffprobe failed on '{audio_path.name}':\n{exc.stderr.strip()}"
        ) from exc
    except ValueError as exc:
        raise RuntimeError(
            f"ffprobe returned unexpected output for '{audio_path.name}': "
            f"'{exc}'"
        ) from exc


def _ffmpeg_extract(
    audio_path: Path,
    start_s:    float,
    duration_s: float,
    output_path: Path,
) -> None:
    """
    Extract [start_s, start_s + duration_s) from audio_path and write it
    to output_path as mono 16 kHz MP3 @ 32 kbps.

    Uses input-side seeking (-ss before -i) which is fast even for large
    files.  Re-encoding ensures the output is a clean, self-contained file
    regardless of the source container/codec.
    """
    cmd = [
        "ffmpeg", "-y",          # -y: overwrite output without prompting
        "-ss", f"{start_s:.3f}", # input-seek (fast keyframe seek)
        "-i",  str(audio_path),
        "-t",  f"{duration_s:.3f}",  # duration to extract
        "-ac", _CHANNELS,        # mono
        "-ar", _SAMPLE_RATE,     # 16 kHz
        "-b:a", _BITRATE,        # 32 kbps
        "-vn",                   # strip any video stream
        str(output_path),
    ]
    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300,  # 5 min per chunk is very generous
        )
    except subprocess.CalledProcessError as exc:
        # Include last 600 chars of stderr for diagnosis
        stderr_tail = exc.stderr[-600:] if exc.stderr else "(no stderr)"
        raise RuntimeError(
            f"ffmpeg failed extracting chunk from '{audio_path.name}':\n{stderr_tail}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ffmpeg timed out (>300 s) extracting a chunk from '{audio_path.name}'."
        ) from exc


def _calc_split_points(
    total_s:   float,
    chunk_s:   int,
    overlap_s: int,
) -> list[tuple[float, float]]:
    """
    Return (start_s, end_s) pairs covering [0, total_s].

    Consecutive windows overlap by overlap_s seconds so no speech is lost
    at boundaries.  The last window may be shorter than chunk_s.
    """
    step_s  = chunk_s - overlap_s   # advance per window
    if step_s <= 0:
        raise ValueError(
            f"CHUNK_OVERLAP_S ({overlap_s}) must be less than "
            f"CHUNK_DURATION_S ({chunk_s})."
        )

    points: list[tuple[float, float]] = []
    cursor = 0.0

    while cursor < total_s:
        end = min(cursor + chunk_s, total_s)
        points.append((cursor, end))
        if end >= total_s:
            break
        cursor += step_s

    return points
