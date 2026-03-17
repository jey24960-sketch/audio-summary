"""
transcriber.py - Transcribe audio chunks using the OpenAI Whisper API.

Design decisions
----------------
* Chunks are transcribed sequentially to stay within rate limits.
* Each successful chunk transcript is cached via ProgressStore so the job
  can be resumed after a crash without re-calling the API.
* The overlap region between adjacent chunks is trimmed from the result
  by a simple heuristic: we drop the last sentence of the *previous* chunk
  and the first sentence of the *current* chunk where they are identical or
  very similar.  For simplicity we skip this deduplication and rely on the
  summariser to handle repeated context — robust enough for lecture audio.
"""

from pathlib import Path

from openai import OpenAI

from audio_processor import AudioChunk, AudioProcessor
from utils import ProgressStore, get_logger, read_env, retry

logger = get_logger(__name__)

# Language hint passed to Whisper.  Leave empty to let the API auto-detect.
WHISPER_LANGUAGE: str = ""  # e.g. "ko" or "en"
WHISPER_MODEL: str = "whisper-1"


class Transcriber:
    """
    Orchestrates the transcription of a single audio file.

    Usage
    -----
        transcriber = Transcriber(store)
        transcript = transcriber.transcribe(audio_path)
    """

    def __init__(self, store: ProgressStore):
        self._store = store
        self._client = OpenAI(api_key=read_env("OPENAI_API_KEY"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_path: str | Path) -> str:
        """
        Return the full transcript for *audio_path*.

        If a cached transcript already exists in the progress store the file
        is not re-processed.
        """
        audio_path = Path(audio_path)
        filename = audio_path.name

        # Fast-path: complete transcript already cached
        if self._store.has(filename, "transcript"):
            logger.info("Using cached transcript for '%s'.", filename)
            return self._store.get(filename, "transcript")

        chunk_transcripts: list[str] = (
            self._store.get(filename, "chunk_transcripts") or []
        )

        with AudioProcessor(audio_path) as processor:
            chunks = list(processor.chunks())
            total = len(chunks)

            for chunk in chunks:
                # Skip already-processed chunks (resume support)
                if chunk.index - 1 < len(chunk_transcripts):
                    logger.info(
                        "Skipping chunk %d/%d (cached).", chunk.index, total
                    )
                    continue

                logger.info(
                    "Transcribing chunk %d/%d [%.1fs – %.1fs] …",
                    chunk.index, total,
                    chunk.start_ms / 1000,
                    chunk.end_ms / 1000,
                )
                text = self._transcribe_chunk(chunk)
                chunk_transcripts.append(text)

                # Persist after every chunk so we can resume on failure
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
    # Internal helpers
    # ------------------------------------------------------------------

    @retry(max_attempts=3, initial_delay=2.0, backoff=2.0)
    def _transcribe_chunk(self, chunk: AudioChunk) -> str:
        """Call the Whisper API for a single chunk file."""
        with open(chunk.path, "rb") as audio_file:
            kwargs: dict = {
                "model": WHISPER_MODEL,
                "file": audio_file,
                "response_format": "text",
            }
            if WHISPER_LANGUAGE:
                kwargs["language"] = WHISPER_LANGUAGE

            response = self._client.audio.transcriptions.create(**kwargs)

        # response_format="text" returns a plain string
        return response.strip() if isinstance(response, str) else response.text.strip()

    @staticmethod
    def _merge(parts: list[str]) -> str:
        """
        Join chunk transcripts with a newline separator.

        Because adjacent chunks share a 30-second overlap there will be some
        repeated text at the seams.  We do a lightweight deduplication: if the
        last sentence of part[i] appears verbatim at the start of part[i+1],
        we strip it from part[i+1].
        """
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]

        merged: list[str] = [parts[0]]
        for part in parts[1:]:
            prev = merged[-1]
            # Split previous part into sentences (rough split on '. ')
            prev_sentences = [s.strip() for s in prev.split(". ") if s.strip()]
            if prev_sentences:
                last_sentence = prev_sentences[-1].lower()
                # Remove leading content from `part` up to and including the overlap
                lower_part = part.lower()
                idx = lower_part.find(last_sentence)
                if idx != -1:
                    trimmed = part[idx + len(last_sentence):].lstrip(". \n")
                    part = trimmed if trimmed else part
            merged.append(part)

        return "\n\n".join(merged)
