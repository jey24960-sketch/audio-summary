"""
summarizer.py - Two-stage exam-oriented summarization via OpenAI Chat API.

Stage 1 — Chunk Summarization
    The full transcript is split into manageable token windows.
    Each window is summarised individually to fit the model's context limit.

Stage 2 — Final Synthesis
    The chunk summaries are combined into a structured, exam-ready document
    containing:
      • 핵심 개념  (Key Concepts)
      • 중요 정의  (Important Definitions)
      • 흐름 정리  (Logical Flow)
      • 시험 포인트 (Likely Exam Points)

Language policy: The summarizer detects the dominant script in the transcript
and instructs the model to respond in the same language (Korean or English).
"""

import re
from dataclasses import dataclass

from openai import OpenAI

from utils import ProgressStore, get_logger, read_env, retry

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHAT_MODEL: str = "gpt-4o"

# Approximate character budget per transcript chunk sent to Stage 1.
# gpt-4o has a 128 k token context; ~3 chars ≈ 1 token → ~12 000 chars is safe
# while leaving room for the system prompt and response.
TRANSCRIPT_CHUNK_CHARS: int = 12_000

# Approximate character budget per summary chunk sent to Stage 2.
SUMMARY_CHUNK_CHARS: int = 8_000


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SummaryResult:
    overall_summary: str       # Final combined summary (Toggle 1 content)
    key_concepts: str          # Structured key concepts  (Toggle 2 content)
    detailed_notes: str        # Detailed / logical flow  (Toggle 3 content)


# ---------------------------------------------------------------------------
# Language detection helper
# ---------------------------------------------------------------------------

def _detect_language(text: str) -> str:
    """Return 'ko' if the text is predominantly Korean, else 'en'."""
    korean_chars = len(re.findall(r"[\uAC00-\uD7A3]", text))
    return "ko" if korean_chars > 50 else "en"


def _lang_instruction(lang: str) -> str:
    if lang == "ko":
        return "반드시 한국어로 작성하세요."
    return "Write your response in English."


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_CHUNK_SYSTEM_KO = """당신은 강의 노트를 시험 준비용으로 정리하는 전문가입니다.
아래 강의 원고 일부를 읽고 핵심 내용을 **구조적으로** 요약하세요.
서술형이 아닌 개조식(bullet / 번호 목록)으로 작성하세요."""

_CHUNK_SYSTEM_EN = """You are an expert at distilling lecture transcripts into structured study notes.
Read the following excerpt and summarise the key content in **structured bullet points**.
Do NOT write in narrative prose — use numbered lists and bullets only."""

_FINAL_SYSTEM_KO = """당신은 대학 강의 내용을 시험 준비용 정리 노트로 변환하는 전문가입니다.
아래는 강의 전체의 청크별 요약입니다. 이를 통합하여 다음 네 섹션을 **반드시 포함한** 최종 정리본을 작성하세요.

## 📌 핵심 개념
- 강의에서 다룬 중요 개념을 간결하게 나열

## 📖 중요 정의
- 정의가 필요한 용어와 그 정의를 명확하게 작성

## 🔄 흐름 정리
- 내용의 논리적 흐름을 순서대로 정리

## 🎯 시험 포인트
- 시험에 나올 가능성이 높은 포인트를 중요도 순으로 나열

서술형 문단 없이 개조식으로 작성하세요."""

_FINAL_SYSTEM_EN = """You are an expert at turning lecture notes into structured exam study guides.
Below are chunk-level summaries of a full lecture. Synthesise them into a final study document
that MUST include the following four sections:

## 📌 Key Concepts
- Concise list of the core concepts covered

## 📖 Important Definitions
- Terms and their precise definitions

## 🔄 Logical Flow
- The lecture's argument / structure in sequential order

## 🎯 Likely Exam Points
- High-probability exam topics, ranked by importance

Use bullet points only — no narrative prose."""


# ---------------------------------------------------------------------------
# Summarizer class
# ---------------------------------------------------------------------------

class Summarizer:
    """
    Two-stage summarization pipeline.

    Usage
    -----
        summarizer = Summarizer(store)
        result = summarizer.summarize(filename, transcript)
    """

    def __init__(self, store: ProgressStore):
        self._store = store
        self._client = OpenAI(api_key=read_env("OPENAI_API_KEY"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize(self, filename: str, transcript: str) -> SummaryResult:
        """
        Run the two-stage pipeline and return a SummaryResult.
        Intermediate chunk summaries are cached for resume support.
        """
        lang = _detect_language(transcript)
        logger.info("Detected language: %s", lang)

        # ---- Stage 1 ------------------------------------------------
        chunk_summaries: list[str] = (
            self._store.get(filename, "chunk_summaries") or []
        )
        transcript_chunks = _split_text(transcript, TRANSCRIPT_CHUNK_CHARS)
        total = len(transcript_chunks)

        for i, chunk in enumerate(transcript_chunks):
            if i < len(chunk_summaries):
                logger.info("Skipping chunk summary %d/%d (cached).", i + 1, total)
                continue
            logger.info("Summarizing transcript chunk %d/%d …", i + 1, total)
            summary = self._summarize_chunk(chunk, lang)
            chunk_summaries.append(summary)
            self._store.update(filename, "chunk_summaries", chunk_summaries)

        # ---- Stage 2 ------------------------------------------------
        if self._store.has(filename, "final_summary_raw"):
            logger.info("Using cached final summary for '%s'.", filename)
            final_raw = self._store.get(filename, "final_summary_raw")
        else:
            logger.info("Generating final synthesis …")
            final_raw = self._synthesize(chunk_summaries, lang)
            self._store.update(filename, "final_summary_raw", final_raw)

        result = self._structure(final_raw, chunk_summaries, lang)
        logger.info("Summarization complete.")
        return result

    # ------------------------------------------------------------------
    # Internal: Stage 1
    # ------------------------------------------------------------------

    @retry(max_attempts=3, initial_delay=2.0, backoff=2.0)
    def _summarize_chunk(self, chunk: str, lang: str) -> str:
        system = _CHUNK_SYSTEM_KO if lang == "ko" else _CHUNK_SYSTEM_EN
        resp = self._client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": chunk},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Internal: Stage 2
    # ------------------------------------------------------------------

    @retry(max_attempts=3, initial_delay=2.0, backoff=2.0)
    def _synthesize(self, chunk_summaries: list[str], lang: str) -> str:
        system = _FINAL_SYSTEM_KO if lang == "ko" else _FINAL_SYSTEM_EN
        combined = _join_with_limit(chunk_summaries, SUMMARY_CHUNK_CHARS)
        resp = self._client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": combined},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Internal: structure the raw final output into three sections
    # ------------------------------------------------------------------

    def _structure(
        self,
        final_raw: str,
        chunk_summaries: list[str],
        lang: str,
    ) -> SummaryResult:
        """
        Parse the final synthesis into three distinct content blocks.

        • overall_summary  → first two sections (Key Concepts + Definitions)
        • key_concepts     → Key Concepts section only
        • detailed_notes   → Logical Flow + Exam Points
        """
        sections = _extract_sections(final_raw)

        if lang == "ko":
            key_concepts_text = sections.get("핵심 개념", "")
            definitions_text  = sections.get("중요 정의", "")
            flow_text         = sections.get("흐름 정리", "")
            exam_text         = sections.get("시험 포인트", "")
        else:
            key_concepts_text = sections.get("Key Concepts", "")
            definitions_text  = sections.get("Important Definitions", "")
            flow_text         = sections.get("Logical Flow", "")
            exam_text         = sections.get("Likely Exam Points", "")

        overall_summary = final_raw  # full synthesis goes to Toggle 1
        key_concepts    = _format_section(key_concepts_text, definitions_text, lang)
        detailed_notes  = _format_section(flow_text, exam_text, lang)

        # Fallback: if parsing failed, use chunk summaries for detailed notes
        if not detailed_notes.strip():
            detailed_notes = "\n\n---\n\n".join(chunk_summaries)

        return SummaryResult(
            overall_summary=overall_summary,
            key_concepts=key_concepts,
            detailed_notes=detailed_notes,
        )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _split_text(text: str, max_chars: int) -> list[str]:
    """
    Split *text* into chunks of at most *max_chars* characters,
    breaking only at paragraph boundaries (double newline) or sentence ends.
    """
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2  # +2 for the "\n\n"
        if current_len + para_len > max_chars and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _join_with_limit(parts: list[str], max_chars: int) -> str:
    """Join parts up to *max_chars* total, truncating gracefully."""
    result: list[str] = []
    total = 0
    for part in parts:
        if total + len(part) + 4 > max_chars:
            break
        result.append(part)
        total += len(part) + 4  # 4 for "\n\n---\n\n"
    return "\n\n---\n\n".join(result)


def _extract_sections(text: str) -> dict[str, str]:
    """
    Extract named sections from markdown-style headings (## heading).
    Returns {heading_text: body_text}.
    """
    pattern = re.compile(r"^##\s+[📌📖🔄🎯]*\s*(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        return {}

    sections: dict[str, str] = {}
    for i, match in enumerate(matches):
        heading = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[heading] = text[start:end].strip()

    return sections


def _format_section(*parts: str) -> str:
    """Concatenate non-empty parts with a blank line separator."""
    return "\n\n".join(p for p in parts if p.strip())
