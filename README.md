# Audio Lecture Summary Pipeline

A production-ready pipeline that:
1. Reads audio lecture files (AAC / M4A / WAV) from a folder
2. Transcribes them with the **OpenAI Whisper API**
3. Summarises the transcript in a **two-stage pipeline** optimised for exam preparation
4. Uploads the results to **Notion** as a structured page per lecture

---

## Folder Structure

```
audio-summary/
├── main.py               # Entry point & CLI
├── audio_processor.py    # Audio loading, splitting, chunk export
├── transcriber.py        # Whisper API transcription
├── summarizer.py         # Two-stage GPT-4o summarization
├── notion_uploader.py    # Notion API page creation
├── utils.py              # Logging, retry, progress cache, helpers
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Install system dependencies

```bash
# Debian / Ubuntu / Google Colab
apt-get install -y ffmpeg

# macOS
brew install ffmpeg
```

### 2. Install Python packages

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

Required keys:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | [OpenAI API key](https://platform.openai.com/api-keys) |
| `NOTION_TOKEN` | [Notion integration token](https://www.notion.so/my-integrations) |
| `NOTION_PARENT_PAGE_ID` | 32-char hex ID of the Notion page that will contain lecture pages |

### 4. Set up the Notion integration

1. Go to https://www.notion.so/my-integrations and create a new integration.
2. Copy the **Internal Integration Token** → `NOTION_TOKEN`.
3. Open the Notion page that should be the parent for lecture pages.
4. Click **...** → **Add connections** → select your integration.
5. Copy the page ID from the URL (`notion.so/<workspace>/<PAGE_ID>`) → `NOTION_PARENT_PAGE_ID`.

---

## Usage

```bash
# Process all audio files in a folder
python main.py --audio-dir ./lectures

# Process a single file
python main.py --audio-dir ./lectures --file lecture01.m4a

# Skip Notion upload (test transcription & summarization only)
python main.py --audio-dir ./lectures --skip-notion

# Resume an interrupted run (cache is in .cache/ by default)
python main.py --audio-dir ./lectures --cache-dir .cache
```

---

## Google Colab Setup

```python
# Cell 1 - system deps
!apt-get install -y ffmpeg

# Cell 2 - Python deps
!pip install openai pydub requests python-dotenv ffmpeg-python

# Cell 3 - clone / upload the pipeline files, then:
import os
os.environ["OPENAI_API_KEY"]          = "sk-..."
os.environ["NOTION_TOKEN"]            = "secret_..."
os.environ["NOTION_PARENT_PAGE_ID"]   = "your-page-id"

# Cell 4 - run
!python main.py --audio-dir /content/lectures --cache-dir /content/.cache
```

---

## Notion Page Structure

Each lecture produces one page with this layout:

```
  <filename>  (YYYY-MM-DD)
  |-- Full exam-oriented synthesis     (전체 요약)
  |-- Key concepts + definitions       (핵심 개념)
  |-- Logical flow + exam points       (상세 내용)
```

---

## Resume / Fault Tolerance

The pipeline caches intermediate results in `.cache/<filename>.json` after each step:

| Cache key | Content |
|---|---|
| `chunk_transcripts` | Per-chunk Whisper outputs |
| `transcript` | Full merged transcript |
| `chunk_summaries` | Per-chunk Stage-1 summaries |
| `final_summary_raw` | Stage-2 synthesis |
| `notion_page_url` | Notion page URL (prevents re-upload) |
| `status` | `done` or `failed` |

Re-run the same command to resume from the last saved checkpoint.

---

## Configuration Knobs

Set these in `.env` to tune behaviour without touching code:

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | `DEBUG` for verbose output |
| `MAX_CHUNK_BYTES` | `20971520` (20 MB) | Max audio chunk size for Whisper |
| `CHUNK_OVERLAP_MS` | `30000` (30 s) | Overlap between audio chunks |

---

## Architecture

```
main.py
  |
  |-- audio_processor.py   pydub + ffmpeg  ->  list of WAV chunks
  |
  |-- transcriber.py       Whisper API     ->  full transcript
  |
  |-- summarizer.py
  |     Stage 1: transcript chunks  ->  GPT-4o chunk summaries
  |     Stage 2: chunk summaries    ->  SummaryResult (3 sections)
  |
  |-- notion_uploader.py   Notion API      ->  page URL
  |
  |-- utils.py             logging, @retry, ProgressStore, file helpers
```

---

## Error Handling

- All API calls retry up to **3 times** with exponential backoff (2s -> 4s -> 8s).
- On failure, intermediate results are saved to `.cache/` automatically.
- The failed file is marked `status: failed`; other files continue processing.
- Re-running the pipeline resumes from the last cached checkpoint.
