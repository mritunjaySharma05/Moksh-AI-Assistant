import streamlit as st
import ollama
import os
import json
from pathlib import Path
from datetime import datetime
from ddgs import DDGS
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# ─────────────────────────────────────────────
# 1. CONFIG & PATHS (cross-platform, no hardcoding)
# ─────────────────────────────────────────────
SAVE_DIR    = Path.home() / "MokshAI_Data"
VECTOR_DIR  = SAVE_DIR / "vector_store"
SAVE_FILE   = SAVE_DIR / "chat_history.json"
MAX_HISTORY = 10   # rolling turns kept in Ollama context

for path in [SAVE_DIR, VECTOR_DIR]:
    path.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 2. EMBEDDING MODEL  (cached — loads once)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


# ─────────────────────────────────────────────
# 3. MOKSH BRAIN  (FAISS vector RAG)
# ─────────────────────────────────────────────
class MokshBrain:
    DIMENSION = 384

    def __init__(self):
        self.model      = load_embedding_model()
        self.index_path = str(VECTOR_DIR / "faiss_index.bin")
        self.meta_path  = VECTOR_DIR / "metadata.json"
        self._load()

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

    def add(self, texts: str | list[str]):
        if not texts:
            return
        if isinstance(texts, str):
            texts = [texts]
        # Deduplicate before adding
        new_texts = [t for t in texts if t not in self.metadata]
        if not new_texts:
            return
        embeddings = self.model.encode(new_texts)
        self.index.add(np.array(embeddings, dtype='float32'))
        self.metadata.extend(new_texts)
        self.save()

    def search(self, query: str, top_k: int = 3) -> str:
        if self.index.ntotal == 0:
            return ""
        try:
            vec = np.array(self.model.encode([query]), dtype='float32')
            distances, indices = self.index.search(vec, min(top_k, self.index.ntotal))
            results = [self.metadata[i] for i in indices[0] if i != -1]
            return "\n".join(dict.fromkeys(results))  # preserve order, deduplicate
        except Exception:
            return ""

    @property
    def total(self) -> int:
        return self.index.ntotal


# ─────────────────────────────────────────────
# 4. TOOLS
# ─────────────────────────────────────────────
def web_search(query: str, max_results: int = 3) -> tuple[str, list[str]]:
    """Returns (formatted context, list of source titles) or ("", []) on failure."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "", []
        context = "\n".join(
            f"[{i+1}] {r['title']}: {r['body'][:300]}"
            for i, r in enumerate(results)
        )
        sources = [r['title'] for r in results]
        return context, sources
    except Exception:
        return "", []


def save_chat(history: list):
    SAVE_FILE.write_text(json.dumps(history, indent=2))


def load_chat() -> list:
    if SAVE_FILE.exists():
        try:
            return json.loads(SAVE_FILE.read_text())
        except Exception:
            return []
    return []


def build_ollama_messages(system: str, history: list, prompt: str) -> list:
    """Build rolling message list: system + last MAX_HISTORY turns + current prompt."""
    messages = [{"role": "system", "content": system}]
    # Only keep last MAX_HISTORY messages for context window safety
    recent = history[-MAX_HISTORY:] if len(history) > MAX_HISTORY else history
    for msg in recent:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": prompt})
    return messages


# ─────────────────────────────────────────────
# 5. SESSION STATE INIT
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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat()

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


# ─────────────────────────────────────────────
# 6. UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="Moksh AI", page_icon="🧘", layout="wide")
st.title("🧘 Moksh AI — Hybrid Agentic RAG")
st.caption("Built by Mritunjay Sharma · llama3 + FAISS + DuckDuckGo")

# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.header("🧠 Memory")
    new_fact = st.text_area(
        "Teach Moksh a new fact:",
        placeholder="e.g. My favourite framework is PyTorch."
    )
    if st.button("➕ Learn Fact", use_container_width=True):
        if new_fact.strip():
            st.session_state.brain.add(new_fact.strip())
            st.success("Fact saved to permanent memory!")
            st.rerun()
        else:
            st.warning("Please enter a fact first.")

    st.divider()
    st.header("⚙️ Controls")
    enable_web   = st.checkbox("🌐 Enable Web Search", value=True)
    enable_voice = st.checkbox("🎙️ Voice mode (coming soon)", value=False, disabled=True)

    st.divider()
    uploaded_image = st.file_uploader(
        "📷 Upload image (uses llama3.2-vision)",
        type=['jpg', 'png', 'jpeg']
    )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            save_chat([])
            st.rerun()
    with col2:
        if st.button("💣 Reset All", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.brain = MokshBrain()
            if SAVE_DIR.exists():
                import shutil
                shutil.rmtree(SAVE_DIR)
                SAVE_DIR.mkdir(parents=True, exist_ok=True)
                VECTOR_DIR.mkdir(parents=True, exist_ok=True)
            st.rerun()

    st.divider()
    st.metric("🧬 Memory Chunks", st.session_state.brain.total)
    st.metric("💬 Chat Turns", len(st.session_state.chat_history))


# ── Chat Display ─────────────────────────────
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Show last sources if available
if st.session_state.last_sources:
    with st.expander("📚 Sources used in last response"):
        for i, src in enumerate(st.session_state.last_sources, 1):
            st.write(f"{i}. {src}")


# ── Chat Input ───────────────────────────────
if prompt := st.chat_input("Ask me anything…"):
    current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

    # Handle "remember" command (case-insensitive)
    if prompt.strip().lower().startswith("remember"):
        fact = prompt.strip()[8:].strip()  # strip "remember" prefix
        if fact:
            st.session_state.brain.add(f"User said: {fact}")
            st.session_state.last_sources = []
            st.session_state.chat_history.append({"role": "user",      "content": prompt})
            st.session_state.chat_history.append({"role": "assistant", "content": f"✅ Got it! I'll remember: *{fact}*"})
            save_chat(st.session_state.chat_history)
            st.rerun()

    # Show user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):

            # Retrieve context
            local_context = st.session_state.brain.search(prompt)
            web_context, sources = ("", [])
            if enable_web:
                web_context, sources = web_search(prompt)
            st.session_state.last_sources = sources

            # Build system prompt
            system = (
                f"You are Moksh, a professional AI Agent created by Mritunjay Sharma.\n"
                f"Current time: {current_time}\n\n"
                f"PERSONAL MEMORY (prioritise this for user-specific facts):\n"
                f"{local_context or 'None'}\n\n"
                f"WEB SEARCH RESULTS:\n"
                f"{web_context or 'None'}\n\n"
                f"Instructions:\n"
                f"- Answer clearly and concisely.\n"
                f"- For personal questions, use PERSONAL MEMORY first.\n"
                f"- If web results are relevant, cite the source number e.g. [1].\n"
                f"- If you don't know something, say so honestly."
            )

            try:
                if uploaded_image:
                    img_bytes = uploaded_image.getvalue()
                    resp = ollama.chat(
                        model='llama3.2-vision',
                        messages=[{
                            'role': 'user',
                            'content': f"{system}\n\nQuestion: {prompt}",
                            'images': [img_bytes]
                        }]
                    )
                else:
                    messages = build_ollama_messages(
                        system,
                        st.session_state.chat_history[:-1],  # exclude current user msg
                        prompt
                    )
                    resp = ollama.chat(model='llama3', messages=messages)

                response_text = resp['message']['content']
                st.markdown(response_text)

                # Show sources inline if used
                if sources:
                    with st.expander("📚 Sources"):
                        for i, src in enumerate(sources, 1):
                            st.write(f"{i}. {src}")

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response_text
                })
                save_chat(st.session_state.chat_history)

            except ollama.ResponseError as e:
                st.error(f"Ollama error: {e}. Is Ollama running? Try: `ollama serve`")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

# ── Footer ───────────────────────────────────
st.divider()
st.caption(
    f"Moksh AI v2.1 · Memory: {st.session_state.brain.total} chunks · "
    f"Chat: {len(st.session_state.chat_history)} turns · "
    f"Data: {SAVE_DIR}"
)
