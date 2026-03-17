"""
notion_uploader.py - Upload summarised lecture notes to Notion.

Page structure
--------------
  Title  : <filename>  (<YYYY-MM-DD>)
  ├─ Toggle "전체 요약"   → overall_summary content
  ├─ Toggle "핵심 개념"   → key_concepts content
  └─ Toggle "상세 내용"   → detailed_notes content

Notion block constraints
------------------------
* A single rich_text element may not exceed 2 000 characters.
* We split content at ~1 800 characters, always breaking at a newline
  boundary so we never cut mid-sentence.
* Toggles are created as synced_block children so they render as collapsible
  sections in the Notion UI.
"""

import datetime
from typing import Any

import requests

from summarizer import SummaryResult
from utils import get_logger, read_env, retry

logger = get_logger(__name__)

NOTION_API_VERSION = "2022-06-28"
NOTION_API_BASE    = "https://api.notion.com/v1"
MAX_BLOCK_CHARS    = 1_800  # conservative limit per rich_text chunk


class NotionUploader:
    """
    Creates one Notion page per lecture inside a configured parent page.

    Usage
    -----
        uploader = NotionUploader()
        page_url = uploader.upload(audio_filename, summary_result)
    """

    def __init__(self):
        self._token      = read_env("NOTION_TOKEN")
        self._parent_id  = read_env("NOTION_PARENT_PAGE_ID")
        self._session    = requests.Session()
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
        Create a new page under the parent page and return its URL.
        """
        title    = self._make_title(audio_filename)
        children = self._build_children(result)

        logger.info("Creating Notion page: '%s' …", title)
        page = self._create_page(title, children)

        url = page.get("url", "<no url>")
        logger.info("Notion page created: %s", url)
        return url

    # ------------------------------------------------------------------
    # Internal: page creation
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

    # ------------------------------------------------------------------
    # Internal: block builders
    # ------------------------------------------------------------------

    def _build_children(self, result: SummaryResult) -> list[dict]:
        """
        Build the top-level block list: three toggle blocks.
        Each toggle's content is split into paragraph blocks respecting the
        Notion 2 000-character limit.
        """
        return [
            self._toggle_block("📝 전체 요약",  result.overall_summary),
            self._toggle_block("📌 핵심 개념",  result.key_concepts),
            self._toggle_block("📚 상세 내용",  result.detailed_notes),
        ]

    def _toggle_block(self, heading: str, content: str) -> dict:
        """
        Return a toggle block whose children are paragraph blocks.
        The heading itself is limited to 2 000 chars (always safe here).
        """
        children = self._paragraphs_from_text(content)
        return {
            "object": "block",
            "type":   "toggle",
            "toggle": {
                "rich_text": [self._text_element(heading)],
                "children":  children,
            },
        }

    def _paragraphs_from_text(self, text: str) -> list[dict]:
        """
        Convert *text* into a list of paragraph blocks, each under 1 800 chars.
        Splitting happens at newline boundaries.
        """
        chunks = _split_at_newlines(text, MAX_BLOCK_CHARS)
        blocks: list[dict] = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            # A single paragraph block can hold multiple rich_text elements;
            # each element is ≤ 2 000 chars.  We further split to be safe.
            rich_texts = [self._text_element(part) for part in _split_rich_text(chunk)]
            blocks.append({
                "object": "block",
                "type":   "paragraph",
                "paragraph": {"rich_text": rich_texts},
            })
        return blocks or [
            {
                "object": "block",
                "type":   "paragraph",
                "paragraph": {"rich_text": [self._text_element("(내용 없음)")]},
            }
        ]

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _text_element(text: str) -> dict:
        return {"type": "text", "text": {"content": text}}

    @staticmethod
    def _make_title(filename: str) -> str:
        stem = filename.rsplit(".", 1)[0]  # strip extension
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
# Text splitting utilities
# ---------------------------------------------------------------------------

def _split_at_newlines(text: str, max_chars: int) -> list[str]:
    """
    Split *text* into chunks of at most *max_chars* characters,
    breaking only at newline characters.  If a single line exceeds
    *max_chars* it is further split at word boundaries.
    """
    if len(text) <= max_chars:
        return [text]

    lines  = text.splitlines(keepends=True)
    chunks: list[str] = []
    current = ""

    for line in lines:
        if len(current) + len(line) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            # If the line itself is too long, hard-split at words
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
    """Split *text* at word boundaries when no newline exists."""
    words  = text.split(" ")
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
    """
    Ensure each segment is ≤ *max_chars* for the Notion rich_text limit.
    Uses _split_at_newlines for consistency.
    """
    return _split_at_newlines(text, max_chars)
