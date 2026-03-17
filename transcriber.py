"""
transcriber.py - Transcribe audio chunks using the OpenAI Whisper API.

Key design points
-----------------
* Uses AudioProcessor.split_points() to plan chunks WITHOUT extracting them.
  This means already-cached chunks are skipped at zero ffmpeg cost on resume.
* Each chunk is extracted just-in-time via AudioProcessor.extract_chunk(),
  transcribed, then immediately deleted to conserve disk space.
* Similarity-based deduplication (difflib.SequenceMatcher) removes the
  repeated sentences that arise from the 30-second chunk overlap.
* Audio duration is stored in the progress cache so main.py / app.py can
  feed it to CostTracker without re-reading the file.
"""

from difflib import SequenceMatcher
from pathlib import Path

from openai import OpenAI

from audio_processor import AudioChunk, AudioProcessor
from utils import ProgressStore, get_logger, read_env, retry, split_sentences

logger = get_logger(__name__)

WHISPER_LANGUAGE: str = ""    # leave empty for auto-detect; e.g. "ko" or "en"
WHISPER_MODEL:    str = "whisper-1"

# Fuzzy deduplication thresholds
_SIMILARITY_THRESHOLD:    float = 0.82
_OVERLAP_CHECK_SENTENCES: int   = 20


class Transcriber:
    """
    Orchestrates the transcription of a single audio file.

    Usage
    -----
        transcriber = Transcriber(store)
        transcript  = transcriber.transcribe(audio_path)
    """

    def __init__(self, store: ProgressStore):
        self._store  = store
        self._client = OpenAI(api_key=read_env("OPENAI_API_KEY"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_path: str | Path) -> str:
        """
        Return the full transcript for *audio_path*.

        Resumable: already-transcribed chunks are read from the progress
        cache; only remaining chunks are extracted and sent to Whisper.
        """
        audio_path = Path(audio_path)
        filename   = audio_path.name

        # Fast-path: entire transcript already cached
        if self._store.has(filename, "transcript"):
            logger.info("Using cached transcript for '%s'.", filename)
            return self._store.get(filename, "transcript")

        chunk_transcripts: list[str] = (
            self._store.get(filename, "chunk_transcripts") or []
        )

        with AudioProcessor(audio_path) as processor:
            # Persist audio duration for cost tracking (only on first run)
            if not self._store.has(filename, "audio_duration_seconds"):
                self._store.update(
                    filename,
                    "audio_duration_seconds",
                    processor.duration_seconds,
                )

            # Get planned split windows — no ffmpeg calls yet
            split_pts = processor.split_points()
            total     = len(split_pts)
            logger.info("Audio split plan: %d chunk(s).", total)

            for idx, (start_s, end_s) in enumerate(split_pts, start=1):
                # Skip chunks that were already transcribed in a previous run
                if idx - 1 < len(chunk_transcripts):
                    logger.info(
                        "Skipping chunk %d/%d [%.1fs – %.1fs] (cached).",
                        idx, total, start_s, end_s,
                    )
                    continue

                # Extract this specific chunk on demand
                logger.info(
                    "Extracting + transcribing chunk %d/%d [%.1fs – %.1fs] …",
                    idx, total, start_s, end_s,
                )
                chunk = processor.extract_chunk(idx, start_s, end_s)

                try:
                    text = self._transcribe_chunk(chunk)
                finally:
                    # Delete chunk file immediately — no need to keep it
                    try:
                        chunk.path.unlink(missing_ok=True)
                    except Exception:
                        pass

                chunk_transcripts.append(text)
                self._store.update(filename, "chunk_transcripts", chunk_transcripts)
                logger.info(
                    "Chunk %d/%d done (%d chars).", idx, total, len(text)
                )

        transcript = self._merge(chunk_transcripts)
        self._store.update(filename, "transcript", transcript)
        logger.info(
            "Full transcript: %d characters, %.0f words.",
            len(transcript),
            len(transcript.split()),
        )
        return transcript

    # ------------------------------------------------------------------
    # Internal: Whisper API call
    # ------------------------------------------------------------------

    @retry(max_attempts=3, initial_delay=2.0, backoff=2.0)
    def _transcribe_chunk(self, chunk: AudioChunk) -> str:
        """Send one chunk file to the Whisper API and return the text."""
        with open(chunk.path, "rb") as audio_file:
            kwargs: dict = {
                "model":           WHISPER_MODEL,
                "file":            audio_file,
                "response_format": "text",
            }
            if WHISPER_LANGUAGE:
                kwargs["language"] = WHISPER_LANGUAGE
            response = self._client.audio.transcriptions.create(**kwargs)

        return response.strip() if isinstance(response, str) else response.text.strip()

    # ------------------------------------------------------------------
    # Internal: similarity-based deduplication at chunk seams
    # ------------------------------------------------------------------

    @staticmethod
    def _merge(parts: list[str]) -> str:
        """
        Join chunk transcripts, removing sentences duplicated across the
        30-second overlap using fuzzy similarity matching.

        Algorithm
        ---------
        For each adjacent pair (prev, curr):
          1. Split both into sentences.
          2. Compare the last _OVERLAP_CHECK_SENTENCES of prev against
             the first _OVERLAP_CHECK_SENTENCES of curr.
          3. Any curr sentence scoring ≥ _SIMILARITY_THRESHOLD against
             any prev sentence is treated as a duplicate.
          4. All duplicates at the head of curr are stripped.
        """
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]

        merged: list[str] = [parts[0]]

        for part in parts[1:]:
            prev_sentences = split_sentences(merged[-1])
            curr_sentences = split_sentences(part)

            cut = _find_dedup_boundary(
                prev_sentences[-_OVERLAP_CHECK_SENTENCES:],
                curr_sentences,
            )
            if cut > 0:
                logger.debug(
                    "Dedup: removed %d repeated sentence(s) at seam.", cut
                )

            kept = curr_sentences[cut:]
            merged.append(" ".join(kept) if kept else part)

        return "\n\n".join(merged)


# ---------------------------------------------------------------------------
# Fuzzy similarity helpers
# ---------------------------------------------------------------------------

def _sentence_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _find_dedup_boundary(
    prev_tail:     list[str],
    curr_sentences: list[str],
) -> int:
    """
    Return the index in curr_sentences where non-duplicate content starts.

    Scans the first _OVERLAP_CHECK_SENTENCES of curr and marks sentences
    that are sufficiently similar to any sentence in prev_tail as
    duplicates.  Stops at the first non-duplicate to avoid over-trimming.
    """
    last_dup = -1
    for i, sent in enumerate(curr_sentences[:_OVERLAP_CHECK_SENTENCES]):
        if any(
            _sentence_similarity(sent, prev) >= _SIMILARITY_THRESHOLD
            for prev in prev_tail
        ):
            last_dup = i
        else:
            break   # stop at first non-dup to avoid removing real content

    return last_dup + 1   # 0 → nothing trimmed
