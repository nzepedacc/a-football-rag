# ParmaFC · Voice RAG (Audio-to-Text-to-Audio)

A **proof-of-concept** that demonstrates **Speech-to-Text (STT)** to **Text-to-Speech (TTS)** using **Streamlit**, **OpenAI Whisper**, **RAG with `file_search`**, and **OpenAI TTS**.

The app allows you to **upload an audio file** (WAV, MP3, M4A, or WEBM), transcribes it with Whisper, sends the transcription to a **RAG-powered LLM** (using a vector store of football articles), and returns a **spoken answer** using OpenAI’s TTS.

> **Note:** This version **does not support live microphone input** — you must upload a pre-recorded audio file.  
> A sample audio file is included in the repo for quick testing.

---

## Features

- **No `ffmpeg` or `pydub` required** — works with native audio formats.
- **RAG** using OpenAI **Vector Stores** + **Assistants API** (`file_search`).
- **Structured answers** (optional): Strengths, Weaknesses, EvolutionOverTime, Sources.
- **TTS response** with configurable voice (`alloy`, `echo`, etc.).
- **Autoplay** support (after user interaction).
- **Idempotent vector store setup** — safe to re-run.

---

## Demo Audio (Included)

**File:** `Is _Thierry_question.mp3`  
**Content (transcribed):**  
> *"Is Thierry Doumbia a good player?"*

Upload this file to test the full flow instantly.

---

## Prerequisites

- Python 3.9+
- OpenAI API key (with access to `whisper-1`, `tts-1`, `gpt-4o-mini`, and **Assistants API**)
- `pip` and `virtualenv` (recommended)

---
