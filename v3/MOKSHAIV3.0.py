"""
Moksh AI  v3.0  —  Hybrid Agentic RAG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
New in v3.0
  1. Cloud LLM fallback  — Google AI (Gemini 2.0 Flash) when Ollama is offline
  2. Export chat         — download current session as Markdown
  3. Better web search   — text + news combined, 5 results, shows URLs
  4. Voice input         — speak your prompt via mic (faster-whisper STT)
  5. Multiple sessions   — named chats, create/switch/delete
  6. Auto memory         — LLM silently extracts facts after each reply
  7. Polish              — response timing, code copy button, toast notifications

Install extras:
    pip install google-generativeai streamlit-mic-recorder faster-whisper

Run:
    streamlit run "e:/AI AGENT/MOKSHAIV3.0.py" --server.fileWatcherType none
"""

import json
import os
import re
import shutil
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import faiss  # type: ignore
import numpy as np
import ollama # type: ignore
import streamlit as st
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer  # type: ignore
from agent_decision import decide_retrieval_mode, RetrievalMode

# ── Optional: Google AI ───────────────────────────────────────────────────────
try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False

# ── Optional: Voice ───────────────────────────────────────────────────────────
try:
    from streamlit_mic_recorder import mic_recorder 
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

VOICE_AVAILABLE = MIC_AVAILABLE and WHISPER_AVAILABLE


# ─────────────────────────────────────────────
# 1. CONFIG & PATHS
# ─────────────────────────────────────────────
SAVE_DIR      = Path.home() / "MokshAI_Data"
VECTOR_DIR    = SAVE_DIR / "vector_store"
SESSIONS_DIR  = SAVE_DIR / "sessions"
MAX_HISTORY   = 12
SEARCH_LIMIT  = 5
GOOGLE_MODEL  = "gemini-2.0-flash"   # free tier; change to "gemma-2-9b-it" if preferred

for _p in [SAVE_DIR, VECTOR_DIR, SESSIONS_DIR]:
    _p.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 2. WHISPER STT (lazy, GPU-first)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Whisper STT…")
def load_whisper():
    if not WHISPER_AVAILABLE:
        return None
    return WhisperModel("base", device="cpu", compute_type="int8")


def transcribe_audio(audio_bytes: bytes) -> str:
    model = load_whisper()
    if not model:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp = f.name
    try:
        segments, _ = model.transcribe(tmp, beam_size=5)
        return " ".join(s.text.strip() for s in segments).strip()
    finally:
        os.unlink(tmp)


# ─────────────────────────────────────────────
# 3. EMBEDDING MODEL
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ─────────────────────────────────────────────
# 4. MOKSH BRAIN (FAISS RAG — global memory)
# ─────────────────────────────────────────────
class MokshBrain:
    DIMENSION = 384

    def __init__(self):
        self._model     = None          # lazy — loads only on first search/add
        self.index_path = str(VECTOR_DIR / "faiss_index.bin")
        self.meta_path  = VECTOR_DIR / "metadata.json"
        self._load()

    @property
    def model(self):
        if self._model is None:
            self._model = load_embedding_model()
        return self._model

    def _load(self):
        if Path(self.index_path).exists() and self.meta_path.exists():
            try:
                self.index    = faiss.read_index(self.index_path)
                self.metadata = json.loads(self.meta_path.read_text())
                return
            except Exception:
                pass
        self.index    = faiss.IndexFlatL2(self.DIMENSION)
        self.metadata = []

    def save(self):
        faiss.write_index(self.index, self.index_path)
        self.meta_path.write_text(json.dumps(self.metadata, indent=2))

    def add(self, texts):
        if not texts:
            return
        if isinstance(texts, str):
            texts = [texts]
        new_texts = [t for t in texts if t not in self.metadata]
        if not new_texts:
            return
        vecs = self.model.encode(new_texts)
        self.index.add(np.array(vecs, dtype="float32"))
        self.metadata.extend(new_texts)
        self.save()

    def search(self, query: str, top_k: int = 3) -> str:
        if self.index.ntotal == 0:
            return ""
        try:
            vec = np.array(self.model.encode([query]), dtype="float32")
            _, indices = self.index.search(vec, min(top_k, self.index.ntotal))
            results = [self.metadata[i] for i in indices[0] if i != -1]
            return "\n".join(dict.fromkeys(results))
        except Exception:
            return ""

    @property
    def total(self) -> int:
        return self.index.ntotal


# ─────────────────────────────────────────────
# 5. SESSION MANAGEMENT
# ─────────────────────────────────────────────
def list_sessions() -> list[str]:
    names = sorted(f.stem for f in SESSIONS_DIR.glob("*.json"))
    return names if names else ["main"]

def _session_path(name: str) -> Path:
    return SESSIONS_DIR / f"{name}.json"

def load_session(name: str) -> list:
    p = _session_path(name)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return []

def save_session(name: str, history: list):
    _session_path(name).write_text(json.dumps(history, indent=2))

def delete_session(name: str):
    p = _session_path(name)
    if p.exists():
        p.unlink()

def create_session(name: str):
    p = _session_path(name)
    if not p.exists():
        p.write_text("[]")


# ─────────────────────────────────────────────
# 6. TOOLS
# ─────────────────────────────────────────────

def web_search(query: str) -> tuple[str, list[dict]]:
    """Returns (context_str, sources_list). Sources have 'title' and 'url'."""
    try:
        snippets, sources = [], []
        with DDGS() as ddgs:
            text_res = list(ddgs.text(query, max_results=SEARCH_LIMIT))
            try:
                news_res = list(ddgs.news(query, max_results=2))
            except Exception:
                news_res = []

        for i, r in enumerate(text_res):
            snippets.append(f"[{i+1}] {r['title']}: {r['body'][:600]}")
            sources.append({"title": r["title"], "url": r.get("href", r.get("link", ""))})

        offset = len(text_res)
        for i, r in enumerate(news_res):
            snippets.append(f"[{offset+i+1}] [NEWS] {r['title']}: {r['body'][:400]}")
            sources.append({"title": f"[NEWS] {r['title']}", "url": r.get("url", "")})

        return "\n".join(snippets), sources
    except Exception:
        return "", []


def export_chat_md(history: list, session_name: str) -> str:
    lines = [
        f"# Moksh AI — {session_name}",
        f"> Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]
    for msg in history:
        prefix = "### You" if msg["role"] == "user" else "### Moksh"
        lines += [prefix, msg["content"], "", "---", ""]
    return "\n".join(lines)


def get_available_models(timeout: float = 3.0) -> list[str]:
    """Returns model list with a hard timeout so slow Ollama doesn't block the page."""
    result_box: list = []

    def _fetch():
        try:
            result = ollama.list()
            model_list = getattr(result, "models", None)
            if model_list is None and hasattr(result, "get"):
                model_list = result.get("models", [])
            names = []
            for m in (model_list or []):
                name = (
                    getattr(m, "model", None)
                    or getattr(m, "name", None)
                    or (m.get("model") if hasattr(m, "get") else None)
                    or (m.get("name")  if hasattr(m, "get") else None)
                )
                if name:
                    names.append(name)
            result_box.extend(names or ["llama3"])
        except Exception:
            pass

    t = threading.Thread(target=_fetch, daemon=True)
    t.start()
    t.join(timeout)
    return result_box if result_box else []


def build_messages(system: str, history: list, prompt: str) -> list:
    msgs = [{"role": "system", "content": system}]
    for msg in history[-MAX_HISTORY:]:
        msgs.append({"role": msg["role"], "content": msg["content"]})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def stream_ollama(messages: list, model: str):
    for chunk in ollama.chat(model=model, messages=messages, stream=True):
        token = chunk.message.content or ""
        if token:
            yield token


def stream_google(prompt: str, system: str, history: list, api_key: str):
    genai.configure(api_key=api_key)
    g_model = genai.GenerativeModel(GOOGLE_MODEL, system_instruction=system)
    g_history = []
    for msg in history[-MAX_HISTORY:]:
        role = "user" if msg["role"] == "user" else "model"
        g_history.append({"role": role, "parts": [msg["content"]]})
    chat = g_model.start_chat(history=g_history)
    response = chat.send_message(prompt, stream=True)
    for chunk in response:
        if chunk.text:
            yield chunk.text


def auto_extract_memory(brain: MokshBrain, user_msg: str, assistant_msg: str,
                         model: str, ollama_ok: bool, google_key: str):
    """Silently extract facts from a conversation turn and save to memory."""
    extraction_prompt = (
        f"User said: {user_msg}\n"
        f"Assistant replied: {assistant_msg[:600]}\n\n"
        "Extract factual statements about the user (preferences, skills, projects, goals). "
        "Return a JSON array of concise strings. If none, return []."
    )
    try:
        raw = ""
        if ollama_ok:
            resp = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": "Extract facts. Return ONLY a valid JSON array of strings."},
                    {"role": "user",   "content": extraction_prompt},
                ],
            )
            raw = resp.message.content or ""
        elif GOOGLE_AI_AVAILABLE and google_key:
            genai.configure(api_key=google_key)
            g = genai.GenerativeModel(
                GOOGLE_MODEL,
                system_instruction="Extract facts. Return ONLY a valid JSON array of strings.",
            )
            raw = g.generate_content(extraction_prompt).text or ""

        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            facts = json.loads(match.group())
            valid = [f for f in facts if isinstance(f, str) and len(f) > 10]
            if valid:
                brain.add(valid)
                return len(valid)
    except Exception:
        pass
    return 0


# ─────────────────────────────────────────────
# 7. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Moksh AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# 8. AETHER DARK THEME  v3  (no Google Fonts)
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Variables ── */
:root {
    --bg:          #07090f;
    --surface:     #0e1420;
    --surface-2:   #131d30;
    --border:      rgba(59,130,246,0.13);
    --border-dim:  rgba(255,255,255,0.05);
    --accent:      #3b82f6;
    --accent-2:    #60a5fa;
    --green:       #10b981;
    --green-2:     #34d399;
    --amber:       #f59e0b;
    --text:        #dde4f0;
    --text-dim:    #64748b;
    --font-sans:   -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', sans-serif;
    --font-mono:   'Consolas', 'Cascadia Code', 'Courier New', monospace;
}

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; }
#MainMenu, footer, header { visibility: hidden; }

/* ── App background ── */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    font-family: var(--font-sans) !important;
    background: var(--bg) !important;
}
.stApp {
    background:
        radial-gradient(ellipse 70% 45% at 15% 0%,   rgba(59,130,246,0.07) 0%, transparent 65%),
        radial-gradient(ellipse 55% 40% at 85% 100%,  rgba(16,185,129,0.05) 0%, transparent 60%),
        var(--bg) !important;
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1322 0%, #080e1a 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div { padding: 1.2rem 1rem !important; }
[data-testid="stSidebar"] * {
    font-family: var(--font-sans) !important;
    color: var(--text-dim) !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: rgba(96,165,250,0.5) !important;
    margin: 0 0 0.55rem !important;
}

/* ── Main block ── */
[data-testid="stMain"], .main .block-container { background: transparent !important; }
.block-container { max-width: 880px !important; padding: 0 2rem 2rem !important; }

/* ── Header ── */
.moksh-header {
    padding: 2.2rem 0 1.6rem;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-dim);
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
}
.moksh-wordmark {
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.03em;
    background: linear-gradient(120deg, #60a5fa 0%, #34d399 55%, #818cf8 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 5s linear infinite;
}
@keyframes shimmer {
    0%   { background-position: 0%   center; }
    100% { background-position: 200% center; }
}
.moksh-tagline {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-dim) !important;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 0.45rem;
    opacity: 0.6;
}
.moksh-version {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--accent) !important;
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 6px;
    padding: 3px 10px;
    align-self: flex-start;
    margin-top: 0.5rem;
}

/* ── Status badge ── */
.badge {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 4px 12px 4px 9px; border-radius: 20px;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.03em;
}
.badge-online  { background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.28); color: #34d399 !important; }
.badge-offline { background: rgba(239,68,68,0.08);  border: 1px solid rgba(239,68,68,0.28);  color: #f87171 !important; }
.badge-cloud   { background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.28); color: #fbbf24 !important; }
.badge .dot { width: 6px; height: 6px; border-radius: 50%; }
.badge-online  .dot { background: #10b981; animation: pulse-dot 2s ease infinite; }
.badge-offline .dot { background: #ef4444; }
.badge-cloud   .dot { background: #f59e0b; animation: pulse-dot 2s ease infinite; }
@keyframes pulse-dot {
    0%,100% { box-shadow: 0 0 4px currentColor; }
    50%      { box-shadow: 0 0 10px currentColor; }
}

/* ── Section labels ── */
.section-label {
    font-family: var(--font-mono);
    font-size: 0.6rem; font-weight: 700;
    letter-spacing: 0.2em; text-transform: uppercase;
    color: rgba(96,165,250,0.45) !important;
    margin: 1.1rem 0 0.45rem;
    display: flex; align-items: center; gap: 8px;
}
.section-label::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(59,130,246,0.2), transparent);
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 14px !important;
    padding: 1rem 1.2rem !important;
    margin-bottom: 0.65rem !important;
    transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="stChatMessage"]:hover {
    background: rgba(255,255,255,0.03) !important;
    border-color: var(--border) !important;
}
[data-testid="stChatMessageContent"] p {
    font-size: 0.93rem !important;
    line-height: 1.75 !important;
    color: var(--text) !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    border-color: rgba(59,130,246,0.2) !important;
    background: rgba(59,130,246,0.04) !important;
    border-left: 3px solid rgba(59,130,246,0.55) !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    border-color: rgba(16,185,129,0.15) !important;
    background: rgba(16,185,129,0.025) !important;
    border-left: 3px solid rgba(16,185,129,0.4) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    margin-top: 1rem !important;
}
[data-testid="stChatInput"] textarea {
    font-size: 0.93rem !important;
    background: transparent !important;
    color: var(--text) !important;
    caret-color: var(--accent-2) !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #2a3a52 !important; }
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(59,130,246,0.45) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.08), 0 0 24px rgba(59,130,246,0.06) !important;
}

/* ── Buttons ── */
.stButton > button {
    font-size: 0.78rem !important; font-weight: 600 !important;
    background: linear-gradient(135deg, rgba(59,130,246,0.12), rgba(16,185,129,0.06)) !important;
    border: 1px solid rgba(59,130,246,0.22) !important;
    border-radius: 8px !important;
    color: var(--accent-2) !important;
    transition: all 0.18s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(59,130,246,0.22), rgba(16,185,129,0.12)) !important;
    border-color: rgba(59,130,246,0.45) !important;
    color: #93c5fd !important;
    box-shadow: 0 0 20px rgba(59,130,246,0.12), 0 4px 12px rgba(0,0,0,0.3) !important;
    transform: translateY(-1px) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(59,130,246,0.06) 0%, rgba(16,185,129,0.03) 100%);
    border: 1px solid rgba(59,130,246,0.15);
    border-radius: 12px; padding: 0.8rem 1rem; text-align: center;
}
[data-testid="stMetricValue"] {
    font-size: 1.8rem !important; font-weight: 800 !important;
    background: linear-gradient(90deg, #60a5fa, #34d399) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--font-mono) !important;
    color: var(--text-dim) !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 8px !important;
    color: var(--text-dim) !important;
    font-size: 0.85rem !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--border) !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.1) !important;
}

/* ── Text inputs / textarea ── */
textarea, input[type="text"] {
    background: var(--surface) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-size: 0.87rem !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: var(--border) !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.08) !important;
}
label, .stCheckbox label {
    color: var(--text-dim) !important;
    font-size: 0.82rem !important;
}

/* ── Checkbox ── */
[data-testid="stCheckbox"] span { border-color: rgba(59,130,246,0.3) !important; border-radius: 4px !important; }
[data-testid="stCheckbox"] input:checked + span { background: var(--accent) !important; border-color: var(--accent) !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed rgba(59,130,246,0.2) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] * { font-size: 0.8rem !important; color: var(--text-dim) !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { font-size: 0.8rem !important; color: var(--text-dim) !important; }

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid var(--border-dim) !important; margin: 0.8rem 0 !important; }

/* ── Text ── */
p, li, .stMarkdown { color: var(--text-dim) !important; font-size: 0.88rem !important; }
code {
    font-family: var(--font-mono) !important;
    background: rgba(59,130,246,0.1) !important;
    color: var(--accent-2) !important;
    border-radius: 4px !important;
    font-size: 0.82rem !important;
    padding: 1px 6px !important;
}
strong { color: var(--text) !important; font-weight: 600 !important; }

/* ── Code blocks with copy button ── */
pre {
    position: relative !important;
    background: rgba(14,20,32,0.95) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
.copy-btn {
    position: absolute; top: 8px; right: 8px;
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.25);
    color: var(--accent-2);
    border-radius: 5px;
    padding: 2px 10px;
    font-size: 0.68rem;
    cursor: pointer;
    font-family: var(--font-mono);
    transition: all 0.15s;
}
.copy-btn:hover { background: rgba(59,130,246,0.3); }

/* ── Response meta (timing) ── */
.resp-meta {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: var(--text-dim);
    opacity: 0.45;
    margin-top: 0.4rem;
    letter-spacing: 0.06em;
}

/* ── Alerts ── */
[data-testid="stAlert"] { font-size: 0.82rem !important; border-radius: 10px !important; }

/* ── Caption ── */
.stCaption, caption { font-family: var(--font-mono) !important; font-size: 0.66rem !important; color: #1e3a5f !important; }

/* ── Welcome screen ── */
.welcome-wrap {
    text-align: center; padding: 5rem 2rem 3rem;
    border: 1px solid var(--border-dim); border-radius: 18px;
    background: linear-gradient(135deg, rgba(59,130,246,0.03) 0%, rgba(16,185,129,0.02) 100%);
    margin-top: 1.5rem;
}
.welcome-icon { font-size: 3rem; line-height: 1; margin-bottom: 1rem; }
.welcome-title { font-size: 1.05rem; font-weight: 500; color: rgba(221,228,240,0.35) !important; margin-bottom: 2rem; }
.chip-row { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; }
.chip {
    font-size: 0.78rem; padding: 6px 14px;
    border: 1px solid rgba(59,130,246,0.15); border-radius: 20px;
    color: var(--text-dim) !important; background: rgba(59,130,246,0.04);
    cursor: default; transition: all 0.2s;
}
.chip:hover { border-color: rgba(59,130,246,0.4); background: rgba(59,130,246,0.09); color: var(--accent-2) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(59,130,246,0.2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(59,130,246,0.4); }

/* ── Spinner ── */
[data-testid="stSpinner"] * { color: var(--accent-2) !important; font-size: 0.8rem !important; }

</style>

<script>
// Inject copy buttons on all code blocks
function addCopyButtons() {
    document.querySelectorAll('pre').forEach(pre => {
        if (pre.querySelector('.copy-btn')) return;
        const btn = document.createElement('button');
        btn.className = 'copy-btn';
        btn.textContent = 'copy';
        btn.onclick = () => {
            navigator.clipboard.writeText(pre.innerText.replace('copy', '').trim());
            btn.textContent = 'copied!';
            setTimeout(() => btn.textContent = 'copy', 1500);
        };
        pre.style.position = 'relative';
        pre.appendChild(btn);
    });
}
const obs = new MutationObserver(addCopyButtons);
obs.observe(document.body, { childList: true, subtree: true });
addCopyButtons();

</script>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 9. SESSION STATE INIT
# ─────────────────────────────────────────────
if "brain" not in st.session_state:
    st.session_state.brain = MokshBrain()
    if st.session_state.brain.total == 0:
        st.session_state.brain.add([
            "Mritunjay Sharma is an AI Engineer and student graduating in 2026.",
            "Mritunjay has an RTX 4060 GPU with 8GB VRAM for local model inference.",
            "The Deepfake Face Detector project achieved 94.05% accuracy on 140,000 images.",
            "Moksh AI is a Hybrid Agentic RAG assistant built by Mritunjay Sharma.",
        ])

if "session_name" not in st.session_state:
    st.session_state.session_name = list_sessions()[0]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_session(st.session_state.session_name)

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama3"

if "google_api_key" not in st.session_state:
    st.session_state.google_api_key = os.getenv("GOOGLE_API_KEY", "")

if "voice_prompt" not in st.session_state:
    st.session_state.voice_prompt = ""

if "last_resp_time" not in st.session_state:
    st.session_state.last_resp_time = 0.0


# ─────────────────────────────────────────────
# 10. HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="moksh-header">
    <div class="moksh-left">
        <div class="moksh-wordmark">Moksh AI</div>
        <div class="moksh-tagline">Hybrid Agentic RAG &nbsp;·&nbsp; Local LLM &nbsp;·&nbsp; Cloud Fallback &nbsp;·&nbsp; Voice</div>
    </div>
    <div class="moksh-version">v3.0</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 11. SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:

    st.markdown("""
    <div style="padding: 0.4rem 0 1rem;">
        <span style="font-family:'Consolas',monospace;font-size:1.1rem;font-weight:700;color:#f59e0b;">◈ MOKSH</span>
        <span style="font-family:'Consolas',monospace;font-size:0.6rem;color:#333;letter-spacing:0.15em;
                     text-transform:uppercase;display:block;margin-top:2px;">Control Panel v3</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Ollama / cloud status (3s timeout so page doesn't hang) ──
    available_models = get_available_models(timeout=3.0)
    ollama_online    = bool(available_models)

    google_ready = GOOGLE_AI_AVAILABLE and bool(st.session_state.google_api_key)

    if ollama_online:
        st.markdown('<span class="badge badge-online"><span class="dot"></span>Ollama online</span>', unsafe_allow_html=True)
    elif google_ready:
        st.markdown('<span class="badge badge-cloud"><span class="dot"></span>Cloud fallback active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-offline"><span class="dot"></span>No LLM available</span>', unsafe_allow_html=True)

    st.divider()

    # ── Sessions ──
    st.markdown('<div class="section-label">Sessions</div>', unsafe_allow_html=True)

    all_sessions = list_sessions()
    if st.session_state.session_name not in all_sessions:
        all_sessions.append(st.session_state.session_name)

    chosen_session = st.selectbox(
        "session",
        options=all_sessions,
        index=all_sessions.index(st.session_state.session_name),
        label_visibility="collapsed",
    )
    if chosen_session != st.session_state.session_name:
        st.session_state.session_name = chosen_session
        st.session_state.chat_history = load_session(chosen_session)
        st.session_state.last_sources = []
        st.rerun()

    new_session_name = st.text_input("new session name", placeholder="e.g. research", label_visibility="collapsed")
    col_ns1, col_ns2 = st.columns(2)
    with col_ns1:
        if st.button("+ New", use_container_width=True):
            name = new_session_name.strip().replace(" ", "_") or f"session_{int(time.time())}"
            create_session(name)
            st.session_state.session_name  = name
            st.session_state.chat_history  = []
            st.session_state.last_sources  = []
            st.rerun()
    with col_ns2:
        if st.button("Delete", use_container_width=True):
            if st.session_state.session_name != "main":
                delete_session(st.session_state.session_name)
                st.session_state.session_name  = "main"
                st.session_state.chat_history  = load_session("main")
                st.session_state.last_sources  = []
                st.rerun()

    st.divider()

    # ── Model selector ──
    st.markdown('<div class="section-label">Model</div>', unsafe_allow_html=True)
    if ollama_online:
        chosen_model = st.selectbox("model", options=available_models, index=0, label_visibility="collapsed")
        st.session_state.selected_model = chosen_model
    else:
        st.caption("// Ollama offline — using cloud")

    st.divider()

    # ── Google AI API key ──
    st.markdown('<div class="section-label">Cloud Fallback</div>', unsafe_allow_html=True)
    if not GOOGLE_AI_AVAILABLE:
        st.caption("// pip install google-generativeai")
    else:
        gkey_input = st.text_input(
            "Google API Key",
            value=st.session_state.google_api_key,
            type="password",
            placeholder="AIza...",
            label_visibility="collapsed",
        )
        if gkey_input != st.session_state.google_api_key:
            st.session_state.google_api_key = gkey_input
        if st.session_state.google_api_key:
            st.caption(f"// model: {GOOGLE_MODEL}")
        else:
            st.caption("// paste key to enable cloud")

    st.divider()

    # ── Controls ──
    st.markdown('<div class="section-label">Controls</div>', unsafe_allow_html=True)
    enable_web    = st.checkbox("Web search", value=True)
    enable_memory = st.checkbox("Auto memory", value=True)

    st.divider()

    # ── Vision ──
    st.markdown('<div class="section-label">Vision</div>', unsafe_allow_html=True)
    uploaded_image = st.file_uploader("img", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if uploaded_image:
        st.image(uploaded_image, use_container_width=True)
        st.caption("// using llama3.2-vision")

    st.divider()

    # ── Teach memory ──
    st.markdown('<div class="section-label">Teach Memory</div>', unsafe_allow_html=True)
    new_fact = st.text_area("fact", placeholder="e.g. My favourite framework is PyTorch.", label_visibility="collapsed", height=72)
    if st.button("+ Save fact", use_container_width=True):
        if new_fact.strip():
            st.session_state.brain.add(new_fact.strip())
            st.success("Saved.")
            st.rerun()
        else:
            st.warning("Enter a fact first.")

    st.divider()

    # ── Stats ──
    st.markdown('<div class="section-label">Stats</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Memory", st.session_state.brain.total)
    with c2:
        st.metric("Turns", len(st.session_state.chat_history))

    st.divider()

    # ── Export ──
    st.markdown('<div class="section-label">Export</div>', unsafe_allow_html=True)
    if st.session_state.chat_history:
        md_export = export_chat_md(st.session_state.chat_history, st.session_state.session_name)
        st.download_button(
            label="⬇ Download chat (.md)",
            data=md_export,
            file_name=f"moksh_{st.session_state.session_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    else:
        st.caption("// no messages to export")

    st.divider()

    # ── Actions ──
    st.markdown('<div class="section-label">Actions</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            save_session(st.session_state.session_name, [])
            st.rerun()
    with col2:
        if st.button("Reset all", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.last_sources = []
            if SAVE_DIR.exists():
                shutil.rmtree(SAVE_DIR)
            for _p in [SAVE_DIR, VECTOR_DIR, SESSIONS_DIR]:
                _p.mkdir(parents=True, exist_ok=True)
            del st.session_state.brain
            st.rerun()

    st.divider()
    st.caption(f"// {SAVE_DIR}")


# ─────────────────────────────────────────────
# 12. VOICE INPUT
# ─────────────────────────────────────────────
if VOICE_AVAILABLE:
    st.markdown('<div class="voice-hint">🎙 speak your prompt below — or type as usual</div>', unsafe_allow_html=True)
    audio = mic_recorder(start_prompt="🎙 Record", stop_prompt="⏹ Stop", key="mic_input")
    if audio and audio.get("bytes"):
        with st.spinner("// transcribing…"):
            text = transcribe_audio(audio["bytes"])
        if text:
            st.session_state.voice_prompt = text
            st.toast(f"Heard: {text[:60]}…" if len(text) > 60 else f"Heard: {text}")

# ─────────────────────────────────────────────
# 13. CHAT DISPLAY
# ─────────────────────────────────────────────
if not st.session_state.chat_history:
    st.markdown("""
    <div class="welcome-wrap">
        <div class="welcome-icon">🧘</div>
        <div class="welcome-title">What would you like to know?</div>
        <div class="chip-row">
            <span class="chip">💡 Who is Mritunjay?</span>
            <span class="chip">🌐 Latest AI news</span>
            <span class="chip">🧠 remember I love PyTorch</span>
            <span class="chip">📷 Upload an image</span>
            <span class="chip">🔍 Explain RAG systems</span>
            <span class="chip">🎙 Try voice input</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if st.session_state.last_sources:
    with st.expander(f"// {len(st.session_state.last_sources)} sources · {st.session_state.last_resp_time:.1f}s"):
        for i, src in enumerate(st.session_state.last_sources, 1):
            if isinstance(src, dict):
                url  = src.get("url", "")
                title = src.get("title", url)
                if url:
                    st.markdown(f"{i}. [{title}]({url})")
                else:
                    st.write(f"{i}. {title}")
            else:
                st.write(f"{i}. {src}")


# ───────────────────────────────────────────
# ──
# 14. CHAT INPUT & RESPONSE
# ─────────────────────────────────────────────

# Voice prompt pre-fills the chat input
initial_value = st.session_state.get("voice_prompt", "") or ""
if "voice_prompt" in st.session_state:
    del st.session_state["voice_prompt"]

prompt = st.chat_input("▸  ask anything…", key="chat_in") or (initial_value if initial_value else None)

if prompt:
    current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

    # ── remember shortcut ──
    if prompt.strip().lower().startswith("remember"):
        fact = prompt.strip()[8:].strip()
        if fact:
            st.session_state.brain.add(f"User said: {fact}")
            st.session_state.last_sources = []
            st.session_state.chat_history.append({"role": "user",      "content": prompt})
            st.session_state.chat_history.append({"role": "assistant", "content": f"Noted — I'll remember: *{fact}*"})
            save_session(st.session_state.session_name, st.session_state.chat_history)
            st.rerun()
        else:
            st.warning("Nothing to remember. Try: `remember I love PyTorch`")
            st.stop()

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        # ── Gather context (Agentic Decision Layer) ──
        if enable_web:
            decision = decide_retrieval_mode(prompt, st.session_state.brain)
            local_context = decision.memory_context
            web_context, sources = "", []
            if decision.mode in (RetrievalMode.WEB_ONLY, RetrievalMode.HYBRID):
                with st.spinner("// searching the web…"):
                    web_context, sources = web_search(prompt)
            st.caption(f"// mode: {decision.mode.value} · {decision.reasoning}")
        else:
            local_context = st.session_state.brain.search(prompt)
            web_context, sources = "", []
        st.session_state.last_sources = sources

        sys_parts = [
            f"You are Moksh, an AI Agent by Mritunjay Sharma. Date: {current_time}.",
            "Answer clearly and concisely. Say so if you don't know.",
        ]
        if local_context:
            sys_parts.append(f"PERSONAL MEMORY:\n{local_context}")
        if web_context:
            sys_parts.append(f"WEB RESULTS (cite as [1],[2]…):\n{web_context}")
        system = "\n\n".join(sys_parts)

        response_text = ""
        t0 = time.time()

        try:
            if uploaded_image:
                with st.spinner("// analysing image…"):
                    img_bytes = uploaded_image.getvalue()
                    resp = ollama.chat(
                        model="llama3.2-vision",
                        messages=[{
                            "role": "user",
                            "content": f"{system}\n\nQuestion: {prompt}",
                            "images": [img_bytes],
                        }],
                    )
                response_text = resp.message.content or ""
                st.markdown(response_text)

            else:
                history_without_last = st.session_state.chat_history[:-1]
                placeholder = st.empty()

                token_count = 0

                if ollama_online:
                    messages = build_messages(system, history_without_last, prompt)
                    for token in stream_ollama(messages, st.session_state.selected_model):
                        response_text += token
                        token_count += 1
                        if token_count % 8 == 0:
                            placeholder.markdown(response_text + "▌")
                elif google_ready:
                    for token in stream_google(
                        prompt, system, history_without_last, st.session_state.google_api_key
                    ):
                        response_text += token
                        token_count += 1
                        if token_count % 8 == 0:
                            placeholder.markdown(response_text + "▌")
                else:
                    response_text = "⚠ No LLM available. Start Ollama or add a Google API key in the sidebar."

                placeholder.markdown(response_text)

            elapsed = time.time() - t0
            st.session_state.last_resp_time = elapsed
            st.markdown(
                f'<div class="resp-meta">// {elapsed:.2f}s · {len(response_text.split())} words'
                + (" · ollama" if ollama_online else " · google cloud")
                + "</div>",
                unsafe_allow_html=True,
            )

            if sources:
                with st.expander(f"// {len(sources)} sources"):
                    for i, src in enumerate(sources, 1):
                        url   = src.get("url", "") if isinstance(src, dict) else ""
                        title = src.get("title", url) if isinstance(src, dict) else str(src)
                        if url:
                            st.markdown(f"{i}. [{title}]({url})")
                        else:
                            st.write(f"{i}. {title}")

            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
            save_session(st.session_state.session_name, st.session_state.chat_history)

            # ── Auto memory (background thread — capture values before thread starts) ──
            if enable_memory and response_text:
                _brain  = st.session_state.brain
                _model  = st.session_state.selected_model
                _gkey   = st.session_state.google_api_key
                _prompt = prompt
                _reply  = response_text

                def _run_extraction():
                    auto_extract_memory(_brain, _prompt, _reply, _model, ollama_online, _gkey)

                threading.Thread(target=_run_extraction, daemon=True).start()

        except ollama.ResponseError as e:
            st.error(f"// ollama error: {e}")
        except Exception as e:
            st.error(f"// unexpected error: {e}")
