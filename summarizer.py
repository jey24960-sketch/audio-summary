"""
summarizer.py - Two-stage exam-oriented summarization via OpenAI Chat API.

Upgrades over v1
----------------
* [1] Token-based chunking via tiktoken instead of character counting.
      Target: 3 000–4 000 tokens per chunk, never exceeding model limits.
* [3] High-quality exam-oriented prompts replacing the generic ones.
* [4] Token usage is captured from every API call and returned so
      main.py can feed it to CostTracker.
* [5] chunk_summaries stored on SummaryResult for NotionUploader's
      Part-N sub-sections.
"""

import re
from dataclasses import dataclass, field

import tiktoken
from openai import OpenAI

from utils import ProgressStore, get_logger, read_env, retry, split_sentences

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHAT_MODEL: str = "gpt-4o"

# Token budgets per chunk (tiktoken-exact, not character estimates)
CHUNK_TARGET_TOKENS: int = 3_500   # aim for ~3 500 tokens
CHUNK_MAX_TOKENS:    int = 4_000   # hard ceiling per transcript chunk
SUMMARY_MAX_TOKENS:  int = 6_000   # ceiling for combined summaries in Stage 2

# Reserve capacity for system prompt + expected model response
_SYSTEM_RESERVE_TOKENS: int = 1_500

# Encoder — shared across all calls; initialised once at module load
try:
    _ENCODER = tiktoken.encoding_for_model(CHAT_MODEL)
except KeyError:
    _ENCODER = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SummaryResult:
    overall_summary: str          # Final combined synthesis  (Toggle 1)
    key_concepts:    str          # Key concepts + definitions (Toggle 2)
    detailed_notes:  str          # Logical flow + exam points (Toggle 3 fallback)
    chunk_summaries: list[str] = field(default_factory=list)  # Per-chunk detail for Notion Part N


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def _detect_language(text: str) -> str:
    """Return 'ko' if the text is predominantly Korean, else 'en'."""
    korean_chars = len(re.findall(r"[\uAC00-\uD7A3]", text))
    return "ko" if korean_chars > 50 else "en"


# ---------------------------------------------------------------------------
# Prompt templates  [upgrade #3 — high-quality exam-oriented prompts]
# ---------------------------------------------------------------------------

# Stage 1: chunk-level summarization (keep brief; the real structure
# comes from Stage 2 synthesis)
_CHUNK_SYSTEM_KO = """당신은 강의 내용을 시험 대비용으로 정리하는 전문 학술 보조자입니다.
아래 강의 원고 일부를 읽고 다음 규칙에 따라 요약하세요.

규칙:
- 서술형 문장 금지 — 개조식(bullet)으로만 작성
- 핵심 키워드와 개념 중심으로 추출
- 중복·반복 내용 제거
- 교수가 강조한 표현은 그대로 보존"""

_CHUNK_SYSTEM_EN = """You are an expert academic assistant preparing lecture notes for exam study.
Read the excerpt below and summarise it according to these rules:

Rules:
- No narrative prose — use bullet points only
- Focus on key terms, definitions, and concepts
- Remove redundancy and filler
- Preserve exact wording when the lecturer emphasises a definition"""

# Stage 2: final synthesis — the high-quality exam-oriented prompt
_FINAL_SYSTEM_KO = """You are an expert academic assistant.
Your task is to convert lecture transcripts into exam-ready structured notes.
You must extract only high-value information likely to appear on exams.
Output format:
1. 핵심 개념
- 5~10 bullet points
- 핵심 키워드 중심
2. 중요 정의
- 시험에 그대로 나올 수 있는 정의
- 정확한 표현 유지
3. 흐름 정리
- 개념 전개 순서를 단계적으로 설명
4. 시험 포인트
- 출제 가능성 높은 내용
- 왜 중요한지 포함
Rules:
- Remove redundancy
- Avoid narrative explanations
- Focus on structure and clarity
- Preserve professor emphasis if detected
반드시 한국어로 작성하세요."""

_FINAL_SYSTEM_EN = """You are an expert academic assistant.
Your task is to convert lecture transcripts into exam-ready structured notes.
You must extract only high-value information likely to appear on exams.
Output format:
1. Key Concepts
- 5~10 bullet points
- Focus on core keywords
2. Important Definitions
- Definitions that could appear verbatim on the exam
- Preserve exact wording
3. Logical Flow
- Explain the progression of concepts step by step
4. Exam Points
- High-probability exam content
- Include why each point matters
Rules:
- Remove redundancy
- Avoid narrative explanations
- Focus on structure and clarity
- Preserve professor emphasis if detected"""


# ---------------------------------------------------------------------------
# Summarizer class
# ---------------------------------------------------------------------------

class Summarizer:
    """
    Two-stage summarization pipeline.

    Returns a SummaryResult.  Token usage (input + output across all calls)
    is stored in the progress cache under "summarization_tokens" for cost
    tracking by main.py.
    """

    def __init__(self, store: ProgressStore):
        self._store  = store
        self._client = OpenAI(api_key=read_env("OPENAI_API_KEY"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize(self, filename: str, transcript: str) -> SummaryResult:
        lang = _detect_language(transcript)
        logger.info("Detected language: %s", lang)

        total_input_tokens  = 0
        total_output_tokens = 0

        # ---- Stage 1: chunk summarization ---------------------------
        chunk_summaries: list[str] = (
            self._store.get(filename, "chunk_summaries") or []
        )
        # Load previously accumulated token counts (for resume)
        saved_usage = self._store.get(filename, "summarization_tokens") or {}
        total_input_tokens  = saved_usage.get("input",  0)
        total_output_tokens = saved_usage.get("output", 0)

        transcript_chunks = _split_by_tokens(
            transcript, CHUNK_TARGET_TOKENS, CHUNK_MAX_TOKENS
        )
        total = len(transcript_chunks)
        logger.info("Transcript split into %d token-based chunk(s).", total)

        for i, chunk in enumerate(transcript_chunks):
            if i < len(chunk_summaries):
                logger.info("Skipping chunk summary %d/%d (cached).", i + 1, total)
                continue

            logger.info(
                "Summarizing transcript chunk %d/%d (~%d tokens) …",
                i + 1, total, _count_tokens(chunk),
            )
            summary, usage = self._summarize_chunk(chunk, lang)
            chunk_summaries.append(summary)
            total_input_tokens  += usage["input"]
            total_output_tokens += usage["output"]

            # Persist after every chunk for resume support
            self._store.update(filename, "chunk_summaries", chunk_summaries)
            self._store.update(filename, "summarization_tokens", {
                "input":  total_input_tokens,
                "output": total_output_tokens,
            })

        # ---- Stage 2: final synthesis -------------------------------
        if self._store.has(filename, "final_summary_raw"):
            logger.info("Using cached final summary for '%s'.", filename)
            final_raw = self._store.get(filename, "final_summary_raw")
        else:
            logger.info("Generating final synthesis …")
            final_raw, usage = self._synthesize(chunk_summaries, lang)
            total_input_tokens  += usage["input"]
            total_output_tokens += usage["output"]
            self._store.update(filename, "final_summary_raw", final_raw)
            self._store.update(filename, "summarization_tokens", {
                "input":  total_input_tokens,
                "output": total_output_tokens,
            })

        logger.info(
            "Summarization complete — total tokens used: %d in / %d out.",
            total_input_tokens, total_output_tokens,
        )
        result = self._structure(final_raw, chunk_summaries, lang)
        return result

    # ------------------------------------------------------------------
    # Internal: Stage 1
    # ------------------------------------------------------------------

    @retry(max_attempts=3, initial_delay=2.0, backoff=2.0)
    def _summarize_chunk(self, chunk: str, lang: str) -> tuple[str, dict]:
        system = _CHUNK_SYSTEM_KO if lang == "ko" else _CHUNK_SYSTEM_EN
        resp   = self._client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": chunk},
            ],
            temperature=0.3,
        )
        text  = resp.choices[0].message.content.strip()
        usage = {
            "input":  resp.usage.prompt_tokens,
            "output": resp.usage.completion_tokens,
        }
        return text, usage

    # ------------------------------------------------------------------
    # Internal: Stage 2
    # ------------------------------------------------------------------

    @retry(max_attempts=3, initial_delay=2.0, backoff=2.0)
    def _synthesize(self, chunk_summaries: list[str], lang: str) -> tuple[str, dict]:
        system   = _FINAL_SYSTEM_KO if lang == "ko" else _FINAL_SYSTEM_EN
        combined = _join_by_tokens(chunk_summaries, SUMMARY_MAX_TOKENS)
        resp     = self._client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": combined},
            ],
            temperature=0.3,
        )
        text  = resp.choices[0].message.content.strip()
        usage = {
            "input":  resp.usage.prompt_tokens,
            "output": resp.usage.completion_tokens,
        }
        return text, usage

    # ------------------------------------------------------------------
    # Internal: parse final output into SummaryResult sections
    # ------------------------------------------------------------------

    def _structure(
        self,
        final_raw: str,
        chunk_summaries: list[str],
        lang: str,
    ) -> SummaryResult:
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
            exam_text         = sections.get("Exam Points", "")

        key_concepts   = _format_section(key_concepts_text, definitions_text)
        detailed_notes = _format_section(flow_text, exam_text)

        # Fallback if section parsing failed
        if not detailed_notes.strip():
            detailed_notes = "\n\n---\n\n".join(chunk_summaries)

        return SummaryResult(
            overall_summary = final_raw,
            key_concepts    = key_concepts,
            detailed_notes  = detailed_notes,
            chunk_summaries = chunk_summaries,
        )


# ---------------------------------------------------------------------------
# Token-based chunking helpers  [upgrade #1]
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def _split_by_tokens(
    text: str,
    target_tokens: int,
    max_tokens: int,
) -> list[str]:
    """
    Split *text* into chunks that respect *max_tokens*, breaking only at
    sentence boundaries.  Aims for *target_tokens* per chunk so the model
    has headroom for the system prompt and its response.
    """
    if _count_tokens(text) <= max_tokens:
        return [text]

    sentences = split_sentences(text)
    chunks:   list[str] = []
    current:  list[str] = []
    current_tok = 0

    for sentence in sentences:
        sent_tok = _count_tokens(sentence)

        # If a single sentence exceeds max_tokens, hard-split it by words
        if sent_tok > max_tokens:
            if current:
                chunks.append(" ".join(current))
                current, current_tok = [], 0
            chunks.extend(_hard_split_by_tokens(sentence, max_tokens))
            continue

        # Flush if adding this sentence would exceed the target
        if current and current_tok + sent_tok > target_tokens:
            chunks.append(" ".join(current))
            current, current_tok = [], 0

        current.append(sentence)
        current_tok += sent_tok

    if current:
        chunks.append(" ".join(current))

    return chunks


def _hard_split_by_tokens(text: str, max_tokens: int) -> list[str]:
    """Last-resort word-level split for sentences that exceed max_tokens."""
    words   = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_tok = 0

    for word in words:
        word_tok = _count_tokens(word)
        if current and current_tok + word_tok > max_tokens:
            chunks.append(" ".join(current))
            current, current_tok = [], 0
        current.append(word)
        current_tok += word_tok

    if current:
        chunks.append(" ".join(current))
    return chunks


def _join_by_tokens(parts: list[str], max_tokens: int) -> str:
    """Join summaries up to *max_tokens*, truncating gracefully."""
    SEP     = "\n\n---\n\n"
    sep_tok = _count_tokens(SEP)
    result: list[str] = []
    total = 0

    for part in parts:
        part_tok = _count_tokens(part)
        if total + part_tok + sep_tok > max_tokens:
            logger.warning(
                "Summary truncated at %d/%d parts to stay within %d tokens.",
                len(result), len(parts), max_tokens,
            )
            break
        result.append(part)
        total += part_tok + sep_tok

    return SEP.join(result)


# ---------------------------------------------------------------------------
# Section parsing and formatting helpers (unchanged from v1)
# ---------------------------------------------------------------------------

def _extract_sections(text: str) -> dict[str, str]:
    """Extract ## heading → body from the model's markdown output."""
    pattern = re.compile(r"^##\s+[📌📖🔄🎯]*\s*(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        # Try numbered headings like "1. 핵심 개념"
        pattern = re.compile(r"^\d+\.\s+(.+)$", re.MULTILINE)
        matches = list(pattern.finditer(text))
    if not matches:
        return {}

    sections: dict[str, str] = {}
    for i, match in enumerate(matches):
        heading = match.group(1).strip()
        start   = match.end()
        end     = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[heading] = text[start:end].strip()

    return sections


def _format_section(*parts: str) -> str:
    return "\n\n".join(p for p in parts if p.strip())
