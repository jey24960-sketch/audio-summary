"""
transcriber.py - Transcribe audio chunks using the OpenAI Whisper API.

Upgrades over v1
----------------
* [2] Similarity-based deduplication replaces the exact-match heuristic.
      Uses difflib.SequenceMatcher (stdlib, no extra deps) to find and
      remove repeated sentences at chunk seam boundaries.
* [4] Audio duration is stored in the progress cache after transcription
      so main.py can feed it to CostTracker.
"""

from difflib import SequenceMatcher
from pathlib import Path

from openai import OpenAI

from audio_processor import AudioChunk, AudioProcessor
from utils import ProgressStore, get_logger, read_env, retry, split_sentences

logger = get_logger(__name__)

WHISPER_LANGUAGE: str = ""   # leave empty for auto-detect; e.g. "ko" or "en"
WHISPER_MODEL:    str = "whisper-1"

# Sentences from the previous chunk's tail that score above this threshold
# against a sentence in the current chunk's head are considered duplicates.
_SIMILARITY_THRESHOLD: float = 0.82

# How many sentences from each side of the seam to compare
_OVERLAP_CHECK_SENTENCES: int = 20


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
        audio_path = Path(audio_path)
        filename   = audio_path.name

        if self._store.has(filename, "transcript"):
            logger.info("Using cached transcript for '%s'.", filename)
            return self._store.get(filename, "transcript")

        chunk_transcripts: list[str] = (
            self._store.get(filename, "chunk_transcripts") or []
        )

        with AudioProcessor(audio_path) as processor:
            chunks = list(processor.chunks())
            total  = len(chunks)

            # Store audio duration for cost tracking [upgrade #4]
            if not self._store.has(filename, "audio_duration_seconds"):
                self._store.update(
                    filename,
                    "audio_duration_seconds",
                    processor.duration_seconds,
                )

            for chunk in chunks:
                if chunk.index - 1 < len(chunk_transcripts):
                    logger.info("Skipping chunk %d/%d (cached).", chunk.index, total)
                    continue

                logger.info(
                    "Transcribing chunk %d/%d [%.1fs – %.1fs] …",
                    chunk.index, total,
                    chunk.start_ms / 1000,
                    chunk.end_ms   / 1000,
                )
                text = self._transcribe_chunk(chunk)
                chunk_transcripts.append(text)
                self._store.update(filename, "chunk_transcripts", chunk_transcripts)
                logger.info("Chunk %d/%d done (%d chars).", chunk.index, total, len(text))

        transcript = self._merge(chunk_transcripts)
        self._store.update(filename, "transcript", transcript)
        logger.info(
            "Full transcript ready: %d characters, %.0f words.",
            len(transcript),
            len(transcript.split()),
        )
        return transcript

    # ------------------------------------------------------------------
    # Internal: Whisper API call
    # ------------------------------------------------------------------

    @retry(max_attempts=3, initial_delay=2.0, backoff=2.0)
    def _transcribe_chunk(self, chunk: AudioChunk) -> str:
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
    # Internal: similarity-based deduplication  [upgrade #2]
    # ------------------------------------------------------------------

    @staticmethod
    def _merge(parts: list[str]) -> str:
        """
        Merge chunk transcripts, removing sentences repeated across the
        30-second overlap boundary using fuzzy similarity matching.

        Algorithm
        ---------
        For each adjacent pair (prev, curr):
          1. Split both into sentences.
          2. Compare the last N sentences of prev against the first N
             sentences of curr using SequenceMatcher.
          3. Any curr sentence scoring ≥ threshold against any prev
             sentence is considered a duplicate.
          4. All duplicates in the head of curr are stripped; the first
             truly new sentence onwards is kept.
        """
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]

        merged: list[str] = [parts[0]]

        for part in parts[1:]:
            prev_sentences = split_sentences(merged[-1])
            curr_sentences = split_sentences(part)

            cut_index = _find_dedup_boundary(
                prev_sentences[-_OVERLAP_CHECK_SENTENCES:],
                curr_sentences,
            )

            if cut_index > 0:
                logger.debug(
                    "Deduplication: removed %d repeated sentence(s) at seam.",
                    cut_index,
                )
                kept = curr_sentences[cut_index:]
            else:
                kept = curr_sentences

            merged.append(" ".join(kept) if kept else part)

        return "\n\n".join(merged)


# ---------------------------------------------------------------------------
# Fuzzy similarity helpers
# ---------------------------------------------------------------------------

def _sentence_similarity(a: str, b: str) -> float:
    """Return SequenceMatcher ratio for two strings (case-insensitive)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _find_dedup_boundary(
    prev_tail: list[str],
    curr_sentences: list[str],
) -> int:
    """
    Return the index in *curr_sentences* where non-duplicate content starts.

    We scan the first _OVERLAP_CHECK_SENTENCES of curr_sentences and mark
    any sentence that is sufficiently similar to any sentence in prev_tail
    as a duplicate.  The returned index is one past the last duplicate in
    the contiguous duplicate prefix (duplicates scattered later in the chunk
    are intentionally kept to avoid over-trimming).
    """
    last_dup_idx = -1

    for i, curr_sent in enumerate(curr_sentences[:_OVERLAP_CHECK_SENTENCES]):
        is_dup = any(
            _sentence_similarity(curr_sent, prev_sent) >= _SIMILARITY_THRESHOLD
            for prev_sent in prev_tail
        )
        if is_dup:
            last_dup_idx = i
        else:
            # Stop at the first non-duplicate to avoid trimming real content
            # that happens to share words with the previous chunk
            break

    return last_dup_idx + 1  # 0 means nothing was trimmed
