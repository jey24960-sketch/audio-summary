<<<<<<< claude/audio-transcription-pipeline-rDxAi
"""
app.py - Streamlit one-click web UI for the audio lecture pipeline.

Architecture
------------
The pipeline is called as a direct Python function, NOT via subprocess.
This ensures the app runs in the same interpreter and environment,
which is mandatory for reliable execution on Streamlit Cloud.

Deployment (Streamlit Cloud)
----------------------------
1. Push this repository to GitHub.
2. Go to https://share.streamlit.io → New app → select repo.
3. Set "Main file path" to: app.py
4. Under Settings → Secrets, add:
       OPENAI_API_KEY      = "sk-..."
       NOTION_TOKEN        = "secret_..."
       NOTION_PARENT_PAGE_ID = "your-32-char-hex-id"
5. Deploy.  ffmpeg is installed automatically via packages.txt.

Local usage
-----------
    cp .env.example .env   # fill in keys
    streamlit run app.py
"""

import logging
import os
import re
import shutil
import tempfile
import traceback
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Environment setup — must happen BEFORE importing pipeline modules so that
# every module's read_env() / os.getenv() call sees the right values.
# ---------------------------------------------------------------------------

def _inject_streamlit_secrets() -> None:
    """
    Copy Streamlit secrets into os.environ.

    On Streamlit Cloud, secrets are stored in st.secrets and are ALSO
    automatically available as environment variables — but only after the
    first st call.  We do this explicitly to be safe and to handle the
    case where the user overrides keys via the sidebar input fields.
    """
    try:
        for key in ("OPENAI_API_KEY", "NOTION_TOKEN", "NOTION_PARENT_PAGE_ID"):
            val = st.secrets.get(key, "")
            if val and not os.environ.get(key):
                os.environ[key] = str(val)
    except Exception:
        # st.secrets raises if no secrets file exists (local dev without secrets)
        pass


_inject_streamlit_secrets()

# Load .env for local development.  Silently skipped if python-dotenv is not
# installed or if there is no .env file (production / Streamlit Cloud).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Pipeline imports (after env is set up)
# ---------------------------------------------------------------------------
from utils import CostTracker, ProgressStore, ensure_dir, get_logger  # noqa: E402
from transcriber import Transcriber    # noqa: E402
from summarizer import Summarizer      # noqa: E402
from notion_uploader import NotionUploader  # noqa: E402

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Log capture — accumulate all pipeline log records in memory
# ---------------------------------------------------------------------------

class _MemoryLogHandler(logging.Handler):
    """Collect formatted log lines into a list for display in the UI."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


def _attach_handler() -> _MemoryLogHandler:
    handler = _MemoryLogHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
                          datefmt="%H:%M:%S")
    )
    logging.getLogger().addHandler(handler)
    return handler


def _detach_handler(handler: _MemoryLogHandler) -> None:
    logging.getLogger().removeHandler(handler)


# ---------------------------------------------------------------------------
# Core pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline(
    audio_path:  Path,
    skip_notion: bool,
    on_stage:    object,  # callable(str) — progress label callback
) -> dict:
    """
    Run transcription → summarization → Notion upload for one file.

    Parameters
    ----------
    audio_path   : path to the saved audio file (with original filename)
    skip_notion  : if True, skip the Notion upload stage
    on_stage     : callable(str) called with a human-readable stage label

    Returns
    -------
    dict with keys:
        transcript       str
        transcript_path  str
        summary          SummaryResult
        notion_url       str  (only if skip_notion is False)
    """
    output_dir      = Path("outputs")
    transcripts_dir = ensure_dir(output_dir / "transcripts")
    summaries_dir   = ensure_dir(output_dir / "summaries")
    logs_dir        = ensure_dir(output_dir / "logs")
    cache_dir       = Path(".cache")

    store        = ProgressStore(cache_dir)
    cost_tracker = CostTracker(logs_dir / "costs.json")

    filename = audio_path.name
    stem     = audio_path.stem
    out: dict = {}

    # ---- Stage 1: Transcription ----------------------------------------
    on_stage("⏳ Stage 1 / 3 — Transcribing audio (this is the longest step)…")
    transcriber = Transcriber(store)
    transcript  = transcriber.transcribe(audio_path)

    transcript_path = transcripts_dir / f"{stem}_transcript.txt"
    transcript_path.write_text(transcript, encoding="utf-8")

    duration_s = store.get(filename, "audio_duration_seconds") or 0
    cost_tracker.record_transcription(filename, duration_s)

    out["transcript"]      = transcript
    out["transcript_path"] = str(transcript_path)

    # ---- Stage 2: Summarization ----------------------------------------
    on_stage("⏳ Stage 2 / 3 — Summarizing transcript…")
    summarizer = Summarizer(store)
    result     = summarizer.summarize(filename, transcript)

    (summaries_dir / f"{stem}_overall.txt").write_text(
        result.overall_summary, encoding="utf-8"
    )
    (summaries_dir / f"{stem}_concepts.txt").write_text(
        result.key_concepts, encoding="utf-8"
    )
    (summaries_dir / f"{stem}_detailed.txt").write_text(
        result.detailed_notes, encoding="utf-8"
    )

    usage = store.get(filename, "summarization_tokens") or {}
    cost_tracker.record_summarization(
        filename,
        input_tokens  = usage.get("input",  0),
        output_tokens = usage.get("output", 0),
    )
    cost_tracker.record_total(filename)
    cost_tracker.save()
    out["summary"] = result

    # ---- Stage 3: Notion upload ----------------------------------------
    if skip_notion:
        on_stage("⏭️  Stage 3 / 3 — Notion upload skipped.")
    else:
        on_stage("⏳ Stage 3 / 3 — Uploading to Notion…")
        uploader = NotionUploader()
        url = uploader.upload(filename, result)
        store.update(filename, "notion_page_url", url)
        out["notion_url"] = url

    store.update(filename, "status", "done")
    return out


# ---------------------------------------------------------------------------
# Streamlit page layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title = "Lecture Summary Pipeline",
    page_icon  = "🎓",
    layout     = "wide",
)

st.title("🎓 Lecture Audio → Exam Notes")
st.caption(
    "Upload a lecture recording → auto-transcribe → summarise for exams → upload to Notion."
)

# ---- Sidebar: settings + optional key overrides --------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    skip_notion = st.checkbox(
        "Skip Notion upload",
        value=False,
        help="Useful for testing or if you don't have Notion configured.",
    )

    st.divider()
    st.subheader("API Key Override")
    st.caption(
        "Leave blank to use environment variables or Streamlit secrets. "
        "Values entered here are applied only for this browser session."
    )

    ov_openai    = st.text_input("OPENAI_API_KEY",          type="password", value="")
    ov_notion    = st.text_input("NOTION_TOKEN",            type="password", value="")
    ov_parent    = st.text_input("NOTION_PARENT_PAGE_ID",   value="")

    if st.button("Apply", use_container_width=True):
        if ov_openai:  os.environ["OPENAI_API_KEY"]          = ov_openai
        if ov_notion:  os.environ["NOTION_TOKEN"]            = ov_notion
        if ov_parent:  os.environ["NOTION_PARENT_PAGE_ID"]   = ov_parent
        st.success("Keys applied for this session.")

    st.divider()
    st.subheader("ℹ️ Required secrets")
    st.markdown(
        "- `OPENAI_API_KEY`\n"
        "- `NOTION_TOKEN`\n"
        "- `NOTION_PARENT_PAGE_ID`\n\n"
        "**Streamlit Cloud:** Settings → Secrets  \n"
        "**Local:** `.env` file"
    )

# ---- Main area: upload + run --------------------------------------------
uploaded = st.file_uploader(
    "Upload lecture audio",
    type=["m4a", "aac", "wav", "mp3"],
    help="Supported formats: .m4a  .aac  .wav  .mp3",
)

if uploaded is None:
    st.info("👆 Upload an audio file to get started.")
    st.stop()

# Show a playback widget so the user can verify they uploaded the right file
st.audio(uploaded)

st.markdown(f"**File:** `{uploaded.name}`  |  **Size:** {len(uploaded.getvalue()) / 1_048_576:.1f} MB")

run_btn = st.button("▶ Run Pipeline", type="primary", use_container_width=False)

if not run_btn:
    st.stop()

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

# Save the uploaded file to a temp directory under its original name.
# Using the original name matters because the ProgressStore keys on filename
# for resume support.
tmp_dir    = Path(tempfile.mkdtemp(prefix="audio_upload_"))
# Sanitise filename to avoid shell/path issues (keep extension)
safe_stem  = re.sub(r"[^\w\-]", "_", Path(uploaded.name).stem)
safe_name  = safe_stem + Path(uploaded.name).suffix.lower()
audio_path = tmp_dir / safe_name
audio_path.write_bytes(uploaded.getvalue())

log_handler = _attach_handler()

status_box  = st.empty()   # single-line status label
log_box     = st.empty()   # scrolling log area (updated during run)

def _update_status(msg: str) -> None:
    status_box.info(msg)
    log_box.code("\n".join(log_handler.lines[-40:]), language="")

try:
    with st.spinner("Pipeline running — do not close this tab…"):
        out = _run_pipeline(
            audio_path   = audio_path,
            skip_notion  = skip_notion,
            on_stage     = _update_status,
        )

    status_box.success("✅ Pipeline complete!")

    # Final log flush
    if log_handler.lines:
        with st.expander("📋 Full log", expanded=False):
            st.code("\n".join(log_handler.lines), language="")

    # ---- Results --------------------------------------------------------
    st.subheader("Results")

    tabs = st.tabs(["핵심 개념", "전체 요약", "상세 내용", "원본 텍스트"])

    with tabs[0]:
        st.markdown(out["summary"].key_concepts)

    with tabs[1]:
        st.markdown(out["summary"].overall_summary)

    with tabs[2]:
        st.markdown(out["summary"].detailed_notes)

    with tabs[3]:
        st.text_area(
            "Full transcript",
            value  = out["transcript"],
            height = 350,
            label_visibility = "collapsed",
        )

    # Notion link
    if "notion_url" in out:
        st.success(f"📝 Notion page created: {out['notion_url']}")

    # Download buttons
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            label     = "⬇️ Download transcript (.txt)",
            data      = out["transcript"],
            file_name = f"{safe_stem}_transcript.txt",
            mime      = "text/plain",
        )
    with col_b:
        st.download_button(
            label     = "⬇️ Download summary (.txt)",
            data      = out["summary"].overall_summary,
            file_name = f"{safe_stem}_summary.txt",
            mime      = "text/plain",
        )

except Exception as exc:
    status_box.error(f"❌ Pipeline failed: {exc}")
    with st.expander("Error details", expanded=True):
        st.code(traceback.format_exc())
    with st.expander("Logs"):
        st.code("\n".join(log_handler.lines))

finally:
    _detach_handler(log_handler)
    shutil.rmtree(tmp_dir, ignore_errors=True)

