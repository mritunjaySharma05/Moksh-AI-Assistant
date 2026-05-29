# 🧘 Moksh AI v3.0 — Hybrid Agentic RAG System

> The most advanced version of Moksh AI — now with an **explicit agentic decision layer**, Google Gemini cloud fallback, voice input, multiple chat sessions, and a full evaluation suite.

---

## 🚀 What is Moksh AI?

Moksh AI is a locally-running intelligent assistant that combines:
- **LLM reasoning** via Llama 3 (Ollama) with Google Gemini 2.0 Flash as cloud fallback
- **Long-term vector memory** via FAISS + Sentence Transformers (all-MiniLM-L6-v2)
- **Live web search** via DuckDuckGo — no API key needed
- **Agentic decision layer** — intelligently decides when to use memory, web, or both
- **Vision understanding** via llama3.2-vision
- **Voice input** via Whisper STT
- **Persistent chat history** saved across multiple named sessions

It's not a simple chatbot — it's a **Hybrid Agentic RAG system** that analyses every query before retrieval, routes it through the right data sources, and generates a grounded response.

---

## 🆕 What's New in v3.0

| Feature | v2 | v3.0 |
|---|---|---|
| Agentic Decision Layer | ❌ manual toggle | ✅ algorithmic routing |
| Cloud LLM Fallback | ❌ | ✅ Google Gemini 2.0 Flash |
| Voice Input | ❌ | ✅ Whisper STT |
| Multiple Chat Sessions | ❌ | ✅ named sessions |
| Auto Memory Extraction | ❌ | ✅ background thread |
| Web Search | ✅ text only | ✅ text + news combined |
| Chat Export | ❌ | ✅ Markdown download |
| Response Timing | ❌ | ✅ per-response metrics |
| HF Spaces Ready | ❌ | ✅ app.py included |
| Evaluation Suite | ❌ | ✅ 25 questions, 3 metrics |

---

## 🧠 Architecture

```
User Input (text or voice)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│                  Agentic Decision Layer                   │
│              (agent_decision.py)                          │
│                                                           │
│  1. Detect TEMPORAL keywords  → bias toward WEB           │
│  2. Detect PERSONAL keywords  → bias toward MEMORY        │
│  3. Check FAISS L2 distance   → confidence threshold      │
│                                                           │
│  Output:  MEMORY_ONLY | WEB_ONLY | HYBRID                 │
└──────────┬──────────────────────┬────────────────────────┘
           │                      │
    ┌──────▼──────┐       ┌───────▼──────┐
    │ FAISS Brain │       │  DuckDuckGo  │
    │ IndexFlatL2 │       │  Web Search  │
    │  384-dim    │       │  5+2 results │
    └──────┬──────┘       └───────┬──────┘
           └──────────┬───────────┘
                      ▼
            System Prompt Builder
                      │
                      ▼
          ┌─────────────────────────┐
          │  LLM: Ollama (llama3)   │
          │  Fallback: Gemini 2.0   │
          └─────────────────────────┘
                      │
                      ▼
          Streamed Response + Sources
                      │
                      ▼
     Background: Auto Memory Extraction
```

---

## 🤖 Agentic Decision Logic

The core innovation of v3.0 is `agent_decision.py` — it replaces the manual checkbox with real algorithmic routing.

**Decision Table:**

| Temporal? | Personal? | FAISS Distance | → Mode |
|---|---|---|---|
| YES | NO | any | `WEB_ONLY` |
| YES | YES | < 0.50 | `HYBRID` |
| YES | YES | ≥ 0.50 | `WEB_ONLY` |
| NO | YES | < 0.50 | `MEMORY_ONLY` |
| NO | YES | 0.50 – 1.50 | `HYBRID` |
| NO | NO | < 0.50 | `MEMORY_ONLY` |
| NO | NO | 0.50 – 1.50 | `HYBRID` |
| NO | NO | > 1.50 | `WEB_ONLY` |

**Temporal keywords** (trigger web): `latest`, `today`, `news`, `current`, `2025`, `trending`...  
**Personal keywords** (trigger memory): `mritunjay`, `my`, `you`, `your`, `who built`, `deepfake`...

---

## 📊 Evaluation Results

Tested on 25 questions across 5 categories using `llama3` via Ollama.

### Moksh AI v3.0 vs Baseline RAG (no web search)

| Metric | Moksh AI v3.0 | Baseline RAG | Δ |
|---|---|---|---|
| **Overall Accuracy** | **92.0%** | 64.0% | **+28pp** |
| Avg Response Latency | 4.447s | 0.902s | +3.5s |
| Retrieval Precision | 10.7% | 10.7% | — |

### Per-Category Accuracy

| Category | Moksh AI v3.0 | Baseline RAG |
|---|---|---|
| system_identity (5 Q) | **100%** | 100% |
| technical_knowledge (9 Q) | **88.9%** | 55.6% |
| general_knowledge (5 Q) | **100%** | 40.0% |
| web_dependent (4 Q) | **100%** | 75.0% |
| reasoning (2 Q) | 50% | 50% |

> Web search adds **+28pp accuracy** at the cost of ~5× higher latency.  
> Run `python eval/evaluate.py --compare` to reproduce these results.

---

## 📁 Folder Structure

```
v3/
├── MOKSHAIV3.0.py              ← Main Streamlit application
├── agent_decision.py           ← Agentic retrieval decision module
├── app.py                      ← Hugging Face Spaces entry point
├── requirements.txt            ← All dependencies (pinned versions)
├── README.md                   ← This file
│
├── eval/
│   ├── evaluate.py             ← Evaluation script (accuracy + latency)
│   ├── baseline_rag.py         ← Simple RAG baseline for comparison
│   └── evaluation_questions.json ← 25 test questions with expected answers
│
└── docs/
    └── research_paper_info.txt ← Full technical documentation
```

---

## ⚙️ Setup & Run

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### 1. Pull models
```bash
ollama pull llama3
ollama pull llama3.2-vision   # optional, for image queries
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run
```bash
streamlit run MOKSHAIV3.0.py
```

Open at **http://localhost:8501**

### 4. Optional extras

**Voice input:**
```bash
pip install streamlit-mic-recorder faster-whisper
```

**Google Cloud fallback** (when Ollama is offline):
```bash
# Paste your key in the sidebar, or set env variable:
set GOOGLE_API_KEY=AIza...
```

---

## 💡 How to Use

| Action | How |
|---|---|
| Ask anything | Type in the chat box |
| Voice input | Click the mic button |
| Analyse an image | Upload jpg/png in sidebar → Vision |
| Teach a fact | Sidebar → Teach Memory → Save |
| Switch sessions | Sidebar → Sessions dropdown |
| Toggle web search | Sidebar → Web search checkbox |
| See decision mode | Small caption under each response |
| Export chat | Sidebar → Download chat (.md) |

---

## 🛠️ Tech Stack

| Layer | Technology | Version |
|---|---|---|
| LLM (local) | Llama 3 via Ollama | ollama 0.6.2 |
| LLM (cloud) | Gemini 2.0 Flash | google-generativeai 0.8.6 |
| Vector Store | FAISS IndexFlatL2 | faiss-cpu 1.13.2 |
| Embeddings | all-MiniLM-L6-v2 | sentence-transformers 5.4.1 |
| Web Search | DuckDuckGo DDGS | duckduckgo-search 8.1.1 |
| Voice STT | Whisper base (int8) | faster-whisper 1.2.1 |
| UI | Streamlit | 1.57.0 |
| Language | Python | 3.10+ |

---

## 🚀 Deploy to Hugging Face Spaces

1. Create a new Space → SDK: **Streamlit**
2. Upload all files from this `v3/` folder
3. Add secret: `GOOGLE_API_KEY` (Settings → Repository secrets)
4. Space auto-installs `requirements.txt` and launches `app.py`

> Ollama is unavailable on Spaces — Google Gemini 2.0 Flash is used as the LLM.

---

## 🧪 Run Evaluation

```bash
# Moksh AI only
python eval/evaluate.py

# Moksh AI vs Baseline comparison
python eval/evaluate.py --compare

# Use Google Gemini instead of Ollama
python eval/evaluate.py --google-key AIza...
```

---

## 📈 v2 → v3.0 Improvements

- Added **agentic decision layer** (`agent_decision.py`) — intelligent routing instead of a manual toggle
- Added **Google Gemini 2.0 Flash** fallback for when Ollama is offline
- Added **Whisper voice input** — speak your prompt instead of typing
- Added **multiple named sessions** — separate chat histories, shared memory
- Added **auto memory extraction** — facts silently stored after every response
- Improved **web search** — text + news combined, 7 results per query
- Added **response timing** — latency shown under every reply
- Added **chat export** — download any session as Markdown
- Added **HF Spaces** deployment support (`app.py`)
- Added **full evaluation suite** — 25 questions, 3 metrics, baseline comparison

---

## 👤 Author

**Mritunjay Sharma** — AI & ML Engineer  
[GitHub](https://github.com/mritunjaySharma05) · [LinkedIn](https://linkedin.com/in/mritunjay-sharma05)

---

> ⭐ Star this repo if you found it useful!
