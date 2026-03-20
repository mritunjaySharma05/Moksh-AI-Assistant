# 🧘 Moksh AI — Hybrid Agentic RAG Assistant

> A production-grade local AI assistant built by **Mritunjay Sharma** — powered by Llama 3, FAISS vector memory, and live web search.

---

## 🚀 What is Moksh AI?

Moksh AI is a locally-running intelligent assistant that combines:
- **LLM reasoning** via Llama 3 (Ollama)
- **Long-term vector memory** via FAISS + Sentence Transformers
- **Live web search** via DuckDuckGo (no API key needed)
- **Vision understanding** via llama3.2-vision
- **Persistent chat history** saved across sessions

It's not a simple chatbot — it's a **Hybrid Agentic RAG system** that retrieves relevant personal memory AND live web context before generating every response.

---

## 📁 Project Structure

```
moksh-ai/
│
├── v1/                        # Version 1 — Tkinter + Voice Interface
│   ├── moksh_v1.py            # Main app (Tkinter GUI + voice)
│   └── requirements.txt       # V1 dependencies
│
├── v2/                        # Version 2 — Hybrid Agentic RAG System
│   ├── moksh_v2.py            # Main app (Streamlit + FAISS + Web Search)
│   └── requirements.txt       # V2 dependencies
│
└── README.md
```

---

## 🧠 Architecture (V2)

```
User Input
    │
    ▼
┌─────────────────────────────────────┐
│           Moksh AI Engine           │
│                                     │
│  ┌─────────────┐  ┌──────────────┐  │
│  │ FAISS Brain │  │  DuckDuckGo  │  │
│  │ (Personal   │  │  Web Search  │  │
│  │  Memory)    │  │  (Live Web)  │  │
│  └──────┬──────┘  └──────┬───────┘  │
│         │                │          │
│         └────────┬───────┘          │
│                  ▼                  │
│         System Prompt Builder       │
│                  │                  │
│                  ▼                  │
│         Llama 3 (via Ollama)        │
│                  │                  │
│                  ▼                  │
│            Response + Sources       │
└─────────────────────────────────────┘
```

---

## ✨ Features

| Feature | V1 | V2 |
|---|---|---|
| LLM (Llama 3) | ✅ | ✅ |
| Voice Input | ✅ | 🔜 |
| GUI | Tkinter | Streamlit |
| Web Search | ❌ | ✅ DuckDuckGo |
| Vector Memory (FAISS) | ❌ | ✅ |
| Persistent Chat History | ❌ | ✅ |
| Vision (Image Upload) | ❌ | ✅ llama3.2-vision |
| Multi-turn Context | Basic | Rolling 10 turns |
| Source Citations | ❌ | ✅ |
| Cross-platform | ❌ | ✅ |

---

## ⚙️ Setup & Run (V2)

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### 1. Pull the required models
```bash
ollama pull llama3
ollama pull llama3.2-vision
```

### 2. Install dependencies
```bash
cd v2
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run moksh_v2.py
```

### 4. Open in browser
```
http://localhost:8501
```

---

## 💡 How to Use

- **Ask anything** — Moksh searches your memory + the web before answering
- **Teach a fact** — Use the sidebar or type `remember [fact]`
- **Upload an image** — Uses llama3.2-vision for visual understanding
- **Toggle web search** — Enable/disable live search from sidebar
- **Clear chat** — Resets conversation without deleting memory

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Llama 3, llama3.2-vision via Ollama |
| Vector Store | FAISS (IndexFlatL2) |
| Embeddings | all-MiniLM-L6-v2 (Sentence Transformers) |
| Web Search | DuckDuckGo (ddgs) |
| UI | Streamlit |
| Storage | JSON + binary FAISS index |
| Language | Python 3.10+ |

---

## 📈 V1 → V2 Improvements

- Replaced Tkinter with **Streamlit** for a modern, shareable UI
- Added **FAISS vector memory** — facts persist across sessions
- Added **live DuckDuckGo web search** with source citations
- Fixed **multi-turn context** — Ollama now receives full conversation history
- Made **cross-platform** — no hardcoded Windows paths
- Added **vision support** via llama3.2-vision
- Added proper **error handling** throughout

---

## 👤 Author

**Mritunjay Sharma** — AI & ML Engineer  
[GitHub](https://github.com/mritunjaySharma05) · [LinkedIn](https://linkedin.com/in/mritunjay-sharma05)

---

> ⭐ Star this repo if you found it useful!
