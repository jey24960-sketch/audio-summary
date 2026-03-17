"""
notion_uploader.py - Upload summarised lecture notes to Notion.

Upgrades over v1
----------------
* [6] Two-phase upload: page is created with the first two toggles fully
      populated, then chunk-level detail is appended to the "상세 내용"
      toggle in batches — safely bypassing Notion's 100-block-per-request
      limit for long lectures.
* [6] "상세 내용" is broken into "Part 1 / Part 2 / …" sub-sections using
      heading_3 blocks, one per transcript chunk.
* [7] All Notion API calls wrapped with @retry.

Page structure (unchanged externally)
--------------------------------------
  Title  : <filename>  (<YYYY-MM-DD>)
  ├─ Toggle "📝 전체 요약"   → overall_summary
  ├─ Toggle "📌 핵심 개념"   → key_concepts
  └─ Toggle "📚 상세 내용"   → Part 1 … Part N  (appended post-creation)
"""

import datetime
from typing import Any

import requests

from summarizer import SummaryResult
from utils import get_logger, read_env, retry

logger = get_logger(__name__)

NOTION_API_VERSION    = "2022-06-28"
NOTION_API_BASE       = "https://api.notion.com/v1"
MAX_BLOCK_CHARS       = 1_800   # conservative Notion rich_text limit
MAX_BLOCKS_PER_BATCH  = 90      # stay under Notion's 100-block API limit


class NotionUploader:
    """
    Creates one Notion page per lecture inside a configured parent page.

    Usage
    -----
        uploader = NotionUploader()
        page_url = uploader.upload(audio_filename, summary_result)
    """

    def __init__(self):
        self._token     = read_env("NOTION_TOKEN")
        self._parent_id = read_env("NOTION_PARENT_PAGE_ID")
        self._session   = requests.Session()
        self._session.headers.update({
            "Authorization":  f"Bearer {self._token}",
            "Notion-Version": NOTION_API_VERSION,
            "Content-Type":   "application/json",
        })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upload(self, audio_filename: str, result: SummaryResult) -> str:
        """
        Two-phase upload:
          Phase 1 — Create the page with 전체 요약 + 핵심 개념 populated
                    and an empty 상세 내용 toggle shell.
          Phase 2 — Fetch the 상세 내용 block ID, then append Part N
                    sub-sections in batches of MAX_BLOCKS_PER_BATCH.
        """
        title = self._make_title(audio_filename)
        logger.info("Creating Notion page: '%s' …", title)

        # Phase 1: create page skeleton
        top_children = [
            self._toggle_block("📝 전체 요약", result.overall_summary),
            self._toggle_block("📌 핵심 개념", result.key_concepts),
            self._toggle_block("📚 상세 내용", ""),   # empty shell for Phase 2
        ]
        page    = self._create_page(title, top_children)
        page_id = page["id"]
        url     = page.get("url", "<no url>")
        logger.info("Page skeleton created: %s", url)

        # Phase 2: append chunk detail to the 상세 내용 toggle
        if result.chunk_summaries:
            blocks = self._get_page_blocks(page_id)
            # The 상세 내용 toggle is the third top-level block (index 2)
            if len(blocks) >= 3:
                detailed_block_id = blocks[2]["id"]
                self._append_detailed_chunks(detailed_block_id, result.chunk_summaries)
                logger.info(
                    "Appended %d Part section(s) to '상세 내용'.",
                    len(result.chunk_summaries),
                )
            else:
                logger.warning(
                    "Could not locate '상세 내용' toggle block — skipping Part append."
                )

        logger.info("Notion page ready: %s", url)
        return url

    # ------------------------------------------------------------------
    # Internal: page + block API calls
    # ------------------------------------------------------------------

    @retry(max_attempts=3, initial_delay=2.0, backoff=2.0, exceptions=(Exception,))
    def _create_page(self, title: str, children: list[dict]) -> dict:
        payload: dict[str, Any] = {
            "parent": {"page_id": self._parent_id},
            "properties": {
                "title": {
                    "title": [{"type": "text", "text": {"content": title}}]
                }
            },
            "children": children,
        }
        resp = self._session.post(f"{NOTION_API_BASE}/pages", json=payload)
        self._raise_for_status(resp)
        return resp.json()

    @retry(max_attempts=3, initial_delay=2.0, backoff=2.0, exceptions=(Exception,))
    def _get_page_blocks(self, page_id: str) -> list[dict]:
        """Fetch the direct children of a page to obtain block IDs."""
        resp = self._session.get(f"{NOTION_API_BASE}/blocks/{page_id}/children")
        self._raise_for_status(resp)
        return resp.json().get("results", [])

    @retry(max_attempts=3, initial_delay=2.0, backoff=2.0, exceptions=(Exception,))
    def _append_block_children(self, block_id: str, children: list[dict]) -> None:
        resp = self._session.patch(
            f"{NOTION_API_BASE}/blocks/{block_id}/children",
            json={"children": children},
        )
        self._raise_for_status(resp)

    def _append_detailed_chunks(
        self, block_id: str, chunk_summaries: list[str]
    ) -> None:
        """
        Append Part N heading + paragraph blocks for each chunk summary,
        batching to respect MAX_BLOCKS_PER_BATCH.
        """
        for i, summary in enumerate(chunk_summaries, start=1):
            part_blocks: list[dict] = []

            # "Part N" heading
            part_blocks.append({
                "object": "block",
                "type":   "heading_3",
                "heading_3": {
                    "rich_text": [self._text_element(f"Part {i}")],
                },
            })
            # Summary paragraphs
            part_blocks.extend(self._paragraphs_from_text(summary))

            # Append in batches of MAX_BLOCKS_PER_BATCH
            for start in range(0, len(part_blocks), MAX_BLOCKS_PER_BATCH):
                batch = part_blocks[start: start + MAX_BLOCKS_PER_BATCH]
                self._append_block_children(block_id, batch)

    # ------------------------------------------------------------------
    # Internal: block builders
    # ------------------------------------------------------------------

    def _toggle_block(self, heading: str, content: str) -> dict:
        """
        Return a toggle block.  Children are only embedded when content is
        non-empty — the 상세 내용 toggle is intentionally left childless
        so Phase 2 can append to it cleanly.
        """
        block: dict[str, Any] = {
            "object": "block",
            "type":   "toggle",
            "toggle": {
                "rich_text": [self._text_element(heading)],
            },
        }
        if content.strip():
            block["toggle"]["children"] = self._paragraphs_from_text(content)
        return block

    def _paragraphs_from_text(self, text: str) -> list[dict]:
        """
        Convert *text* into paragraph blocks, each ≤ MAX_BLOCK_CHARS.
        """
        chunks = _split_at_newlines(text, MAX_BLOCK_CHARS)
        blocks: list[dict] = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            rich_texts = [
                self._text_element(part)
                for part in _split_rich_text(chunk)
            ]
            blocks.append({
                "object":    "block",
                "type":      "paragraph",
                "paragraph": {"rich_text": rich_texts},
            })
        return blocks or [{
            "object":    "block",
            "type":      "paragraph",
            "paragraph": {"rich_text": [self._text_element("(내용 없음)")]},
        }]

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _text_element(text: str) -> dict:
        return {"type": "text", "text": {"content": text}}

    @staticmethod
    def _make_title(filename: str) -> str:
        stem = filename.rsplit(".", 1)[0]
        date = datetime.date.today().isoformat()
        return f"{stem}  ({date})"

    @staticmethod
    def _raise_for_status(resp: requests.Response) -> None:
        if not resp.ok:
            logger.error(
                "Notion API error %d: %s",
                resp.status_code, resp.text[:500],
            )
            resp.raise_for_status()


# ---------------------------------------------------------------------------
# Text splitting utilities (unchanged from v1)
# ---------------------------------------------------------------------------

def _split_at_newlines(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    lines   = text.splitlines(keepends=True)
    chunks: list[str] = []
    current = ""

    for line in lines:
        if len(current) + len(line) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            if len(line) > max_chars:
                for part in _hard_split(line, max_chars):
                    chunks.append(part)
            else:
                current = line
        else:
            current += line

    if current:
        chunks.append(current)
    return chunks


def _hard_split(text: str, max_chars: int) -> list[str]:
    words   = text.split(" ")
    chunks: list[str] = []
    current = ""
    for word in words:
        segment = (current + " " + word) if current else word
        if len(segment) > max_chars:
            if current:
                chunks.append(current)
            current = word
        else:
            current = segment
    if current:
        chunks.append(current)
    return chunks


def _split_rich_text(text: str, max_chars: int = 2_000) -> list[str]:
    return _split_at_newlines(text, max_chars)
