# Audio Lecture Summary Pipeline

Converts lecture audio → full transcript → exam-ready structured notes → Notion page.

```
Audio (.m4a / .aac / .wav)
  │
  ├─ ffmpeg         →  10-minute MP3 chunks
  ├─ Whisper API    →  chunk transcripts  →  merged full transcript
  ├─ GPT-4o         →  two-stage summarisation (chunk → synthesis)
  └─ Notion API     →  structured lecture page
```

Supports **CLI**, **Google Colab**, and **Streamlit Cloud** (one-click web UI).

---

## Repository layout

```
audio-summary/
├── app.py                # Streamlit web UI entry point
├── main.py               # CLI entry point
├── audio_processor.py    # ffmpeg-based chunking (no pydub)
├── transcriber.py        # Whisper API transcription + fuzzy dedup
├── summarizer.py         # Two-stage GPT-4o summarisation
├── notion_uploader.py    # Notion page creation
├── utils.py              # logging, retry, ProgressStore, CostTracker
├── requirements.txt      # Python dependencies
├── packages.txt          # System packages for Streamlit Cloud (ffmpeg)
├── .env.example          # Environment variable template
└── README.md
```

---

## Prerequisites

### System dependency: ffmpeg

```bash
# Debian / Ubuntu / Google Colab
apt-get install -y ffmpeg

# macOS
brew install ffmpeg

# Streamlit Cloud — handled automatically via packages.txt (already in repo)
```

### Environment variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | [OpenAI API key](https://platform.openai.com/api-keys) |
| `NOTION_TOKEN` | [Notion integration token](https://www.notion.so/my-integrations) |
| `NOTION_PARENT_PAGE_ID` | 32-char hex ID of the Notion parent page |

---

## Option A — CLI usage (local / server)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure keys
cp .env.example .env
# Edit .env and fill in your API keys

# 3. Run on a folder of audio files
python main.py --audio-dir ./lectures

# Run on a single file
python main.py --audio-dir ./lectures --file lecture01.m4a

# Skip Notion upload (transcription + summarisation only)
python main.py --audio-dir ./lectures --skip-notion

# Custom output and cache directories
python main.py --audio-dir ./lectures --output-dir ./outputs --cache-dir .cache
```

Output structure:

```
outputs/
  transcripts/  <stem>_transcript.txt
  summaries/    <stem>_overall.txt
                <stem>_concepts.txt
                <stem>_detailed.txt
  logs/         costs.json
```

---

## Option B — Google Colab

```python
# Cell 1 — system deps
!apt-get install -y ffmpeg

# Cell 2 — Python deps
!pip install openai requests tiktoken streamlit python-dotenv

# Cell 3 — clone repo (or upload files manually)
!git clone https://github.com/<your-org>/audio-summary.git
%cd audio-summary

# Cell 4 — set API keys
import os
os.environ["OPENAI_API_KEY"]        = "sk-..."
os.environ["NOTION_TOKEN"]          = "secret_..."
os.environ["NOTION_PARENT_PAGE_ID"] = "your-page-id"

# Cell 5 — upload audio and run
# Upload your .m4a files to /content/lectures/ via the Files panel, then:
!python main.py --audio-dir /content/lectures --cache-dir /content/.cache

# Cell 5 (alternative) — run directly from Python
from pathlib import Path
from main import run_pipeline
run_pipeline(
    audio_dir   = Path("/content/lectures"),
    output_dir  = Path("/content/outputs"),
    cache_dir   = Path("/content/.cache"),
    skip_notion = False,
    target_file = None,
)
```

---

## Option C — Streamlit Cloud (one-click web UI)

### Deploy

1. Fork or push this repository to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repository.  Set **Main file path** to `app.py`.
4. Under **Settings → Secrets**, add:

```toml
OPENAI_API_KEY        = "sk-..."
NOTION_TOKEN          = "secret_..."
NOTION_PARENT_PAGE_ID = "your-32-char-hex-page-id"
```

5. Click **Deploy**.  `ffmpeg` is installed automatically via `packages.txt`.

### Use

- Open the deployed URL.
- Upload a lecture audio file (`.m4a`, `.aac`, `.wav`, `.mp3`).
- Click **▶ Run Pipeline**.
- Download the transcript or summary, or view the Notion link.

### Local Streamlit

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in keys
streamlit run app.py
```

---

## Notion page structure

Each lecture produces one child page:

```
<filename>  (YYYY-MM-DD)
├─ 📝 전체 요약   — Full exam-oriented synthesis
├─ 📌 핵심 개념   — Key concepts + important definitions
└─ 📚 상세 내용   — Part 1 / Part 2 / … per transcript chunk
```

### Setting up the Notion integration

1. Go to [notion.so/my-integrations](https://www.notion.so/my-integrations) and create an integration.
2. Copy the **Internal Integration Token** → `NOTION_TOKEN`.
3. Open the target Notion page → **…** → **Add connections** → select your integration.
4. Copy the page ID from the URL (`notion.so/<workspace>/<PAGE_ID>`) → `NOTION_PARENT_PAGE_ID`.

---

## Chunking and long audio

The pipeline handles files of any length:

- Audio is split into **10-minute chunks** with a 30-second overlap using `ffmpeg`.
- Each chunk is re-encoded as mono 16 kHz MP3 @ 32 kbps (≈ 2.4 MB / chunk).
- This is well under Whisper's 25 MB per-file limit.
- A 2-hour lecture → ~12 chunks → transcribed sequentially.
- Overlapping sentences are removed via fuzzy similarity matching (SequenceMatcher).

---

## Resume / fault tolerance

Intermediate results are cached in `.cache/<filename>.json` after each step:

| Cache key | Content |
|---|---|
| `chunk_transcripts` | Per-chunk Whisper output |
| `transcript` | Full merged transcript |
| `chunk_summaries` | Per-chunk Stage-1 summaries |
| `final_summary_raw` | Stage-2 synthesis |
| `notion_page_url` | Prevents re-upload |
| `status` | `done` or `failed` |
| `audio_duration_seconds` | Used for cost calculation |

Re-run the same command to resume from the last checkpoint.  Already-completed
files (status = `done`) are skipped automatically.  Use `--cache-dir` to point
to the existing cache on a new machine.

---

## Cost tracking

Costs are written to `outputs/logs/costs.json` after each file:

```json
{
  "lecture01.m4a": {
    "transcription": { "duration_minutes": 62.4, "cost_usd": 0.00374 },
    "summarization": { "input_tokens": 18200, "output_tokens": 1540, "cost_usd": 0.061 },
    "total_cost_usd": 0.0648,
    "processed_at": "2025-03-17T14:22:01"
  }
}
```

Pricing used: Whisper $0.006/min · GPT-4o $2.50/1M input · $10.00/1M output.

---

## Configuration knobs

Set these in `.env` or as environment variables:

| Variable | Default | Description |
|---|---|---|
| `CHUNK_DURATION_S` | `600` | Audio chunk length in seconds (10 min) |
| `CHUNK_OVERLAP_S` | `30` | Overlap between chunks in seconds |
| `LOG_LEVEL` | `INFO` | `DEBUG` for verbose ffmpeg + API output |

---

## Architecture

```
app.py / main.py
  │
  ├── audio_processor.py   ffmpeg subprocess  →  AudioChunk list
  │     _probe_duration()  ffprobe            →  duration_seconds
  │     _ffmpeg_extract()  ffmpeg             →  mono 16kHz 32kbps MP3
  │
  ├── transcriber.py       Whisper API        →  full transcript
  │     split_points()     plan chunks (no I/O)
  │     extract_chunk()    one ffmpeg call on demand
  │     _merge()           fuzzy dedup at seams
  │
  ├── summarizer.py        GPT-4o             →  SummaryResult
  │     Stage 1:  token-based chunks  →  chunk summaries (tiktoken)
  │     Stage 2:  synthesis           →  4-section exam notes
  │
  ├── notion_uploader.py   Notion API         →  page URL
  │     Phase 1:  create page skeleton
  │     Phase 2:  append Part N sub-sections
  │
  └── utils.py             logging, @retry, ProgressStore, CostTracker
```
