# app.py — Streamlit UI
import os
import traceback
from pathlib import Path
from rag_core import ensure_vector_store, rag_answer
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from rag_core import ensure_vector_store, rag_answer


st.set_page_config(page_title="ParmaFC · Voice RAG", page_icon="🎙️", layout="centered")
load_dotenv()

OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
OPEN_AI_MODEL = os.getenv("OPEN_AI_MODEL", "gpt-4o-mini")

# Paths
DEFAULT_DATA_FILE = os.getenv("RAG_DATA_FILE", "data/fake_football_articles.json")
DEFAULT_STORE_NAME = os.getenv("RAG_STORE_NAME", "Some Fake Football Articles")

# TTS
TTS_MODEL = os.getenv("TTS_MODEL", "tts-1")
TTS_VOICE = os.getenv("TTS_VOICE", "alloy")

# max audio size
MAX_MB = int(os.getenv("MAX_AUDIO_MB", "25"))

# reading API KEY
if not OPEN_AI_API_KEY:
    st.error("OPEN_AI_API_KEY not found")
    st.stop()

client = OpenAI(api_key=OPEN_AI_API_KEY)

# ---------- Helpers ----------
ACCEPTED_MIME = {
    "audio/wav", "audio/x-wav",
    "audio/mpeg", "audio/mp3",
    "audio/mp4", "audio/m4a",
    "audio/webm"
}
ACCEPTED_EXT = ["wav", "mp3", "m4a", "webm"]

def transcribe_audio_passthrough(uploaded_file) -> str:
    """
    File is sent to Whisper. (mp3/wav/m4a/webm)
    """
    if uploaded_file.type not in ACCEPTED_MIME:
        st.warning(f"Type error: {uploaded_file.type}. Sube WAV/MP3/M4A/WEBM.")
        return ""

    file_bytes = uploaded_file.getvalue()
    if len(file_bytes) > MAX_MB * 1024 * 1024:
        st.warning(f"El archivo supera {MAX_MB} MB. Sube un audio más corto.")
        return ""

    tr = client.audio.transcriptions.create(
        model="whisper-1",
        file=(uploaded_file.name, file_bytes)
    )
    return (getattr(tr, "text", "") or "").strip()

def synthesize_tts(text: str) -> bytes:
    speech = client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text
        #format="mp3"
    )
    return speech.read()

# ---------- Boot Vector Store ----------
@st.cache_resource(show_spinner=True)
def boot_vector_store(store_name: str, data_file: str) -> str:
    if not Path(data_file).exists():
        raise FileNotFoundError(f"Dataset error: {data_file}")
    return ensure_vector_store(store_name=store_name, path=data_file)

# ---------- UI ----------
st.title("ParmaFC · Audio RAG")
st.caption("Upload an audio with your question and receive a spoken answer based on indexed knowledge.")

with st.expander("configuration", expanded=False):
    colA, colB = st.columns(2)
    with colA:
        store_name = st.text_input("Vector Store", value=DEFAULT_STORE_NAME)
    with colB:
        data_file = st.text_input("Dataset", value=DEFAULT_DATA_FILE)
    st.markdown(
        "Json file loaded."
    )

try:
    with st.spinner("Reading Vector Store..."):
        VECTOR_STORE_ID = boot_vector_store(store_name, data_file)
    st.success(f"Vector Store ready: `{VECTOR_STORE_ID}`")
except Exception as e:
    st.error(f"Vector Store error: {e}")
    st.code(traceback.format_exc())
    st.stop()

uploaded = st.file_uploader("Upload audio file (wav/mp3/m4a/webm)", type=ACCEPTED_EXT)
c1, c2 = st.columns(2)
with c1:
    autoplay = st.toggle("Autoplay answer", value=True)
with c2:
    show_sections = st.toggle("Answer in sections (Strengths/Weaknesses/...)", value=True)

if "turns" not in st.session_state:
    st.session_state.turns = []


if uploaded and st.button("Start", type="primary"):
    try:
        with st.spinner("Analyzing audio..."):
            user_text = transcribe_audio_passthrough(uploaded)
        if not user_text:
            st.warning("Process failed.")
            st.stop()

        st.markdown("**Transcript:**")
        st.write(user_text)

        with st.spinner("querying RAG + LLM..."):
            reply_text = rag_answer(
                query=user_text,
                vector_store_id=VECTOR_STORE_ID,
                temperature=0.2,
                #max_output_tokens=None,
                structured_sections=show_sections
            )

        if not reply_text:
            st.warning("No answer.")
        else:
            st.markdown("**Answer:**")
            st.write(reply_text)

            with st.spinner("Preparing audio..."):
                mp3_bytes = synthesize_tts(reply_text)
            st.audio(mp3_bytes, format="audio/mp3", autoplay=autoplay)

        st.session_state.turns.append({"q": user_text, "a": reply_text})

    except Exception as e:
        st.error(f"Error: {e}")
        st.code(traceback.format_exc())

# ---------- History ----------
if st.session_state.turns:
    st.divider()
    st.subheader("Logs")
    for i, turn in enumerate(reversed(st.session_state.turns), 1):
        st.markdown(f"**{i}. You:** {turn['q']}")
        st.markdown(f"**{i}. App:** {turn['a']}")

# ---------- Notes ----------
st.markdown(
    """
---
**Notes**
- This version **does not require ffmpeg** or pydub. It accepts WAV/MP3/M4A/WEBM files as-is.
"""
)
