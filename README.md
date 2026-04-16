---
title: LangGraph Multi-Agent App
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# LangGraph Multi-Agent App

A Streamlit app combining two LangGraph-powered agents:

- **Blog Writing Agent** — researches a topic and generates a full markdown blog post with an AI-generated image
- **Multi Utility Chatbot** — chat with PDFs, search the web, do math, and fetch live stock prices

---

## Features

### ✍️ Blog Writing Agent
- **Smart routing** — decides whether to do web research (via Tavily) before writing: `closed_book`, `hybrid`, or `open_book` mode
- **Parallel section writing** — uses LangGraph's fan-out to write all blog sections concurrently
- **Image generation** — generates one relevant diagram/image using HuggingFace FLUX.1-schnell and embeds it in the post
- **Download** — export the blog as `.md` or a `.zip` bundle (markdown + image)
- **Past blogs** — sidebar lists previously generated `.md` files for quick reload

### 🤖 Multi Utility Chatbot
- **PDF Q&A** — upload a PDF and ask questions; uses FAISS + HuggingFace embeddings for retrieval
- **Web search** — DuckDuckGo search for real-time or post-2023 information
- **Calculator** — arithmetic via a dedicated tool (never hallucinates math)
- **Stock prices** — live prices via Alpha Vantage API
- **Persistent memory** — conversation history saved per thread using SQLite checkpointing
- **Multi-thread** — create, switch between, and delete chat threads from the sidebar

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM (Blog Agent) | Groq — `llama-3.1-8b-instant` |
| LLM (RAG Chatbot) | HuggingFace — `meta-llama/Llama-3.1-8B-Instruct` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| Image Generation | HuggingFace FLUX.1-schnell |
| Web Research | Tavily |
| Web Search (chat) | DuckDuckGo |
| Frontend | Streamlit |
| Memory | SQLite (`langgraph-checkpoint-sqlite`) |

---

## Setup

### 1. Clone & create a virtual environment

```bash
git clone <repo-url>
cd "Combined (Rag + Tool Chatbot & Blog Writing agent)"
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=<your_groq_api_key>
HUGGINGFACEHUB_API_TOKEN=<your_huggingface_token>
TAVILY_API_KEY=<your_tavily_api_key>
```

| Variable | Required for | Get it from |
|---|---|---|
| `GROQ_API_KEY` | Blog Writing Agent LLM | [console.groq.com](https://console.groq.com) |
| `HUGGINGFACEHUB_API_TOKEN` | RAG Chatbot LLM + Image generation | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `TAVILY_API_KEY` | Web research in Blog Agent | [tavily.com](https://tavily.com) |

> Tavily is optional — if not set, the blog agent skips web research and runs in `closed_book` mode.

### 4. Run the app

```bash
streamlit run app.py
```

---

## Project Structure

```
.
├── app.py                      # Streamlit entry point (page navigation)
├── bwa_backend.py              # Blog Writing Agent — LangGraph graph + nodes
├── bwa_frontend.py             # Blog Writing Agent — additional frontend helpers
├── langraph_rag_backend.py     # RAG Chatbot — LangGraph graph + tools
├── pages/
│   ├── blog_writing_agent.py   # Blog Writing Agent Streamlit page
│   └── rag_with_tools.py       # RAG Chatbot Streamlit page
├── images/                     # Generated blog images saved here
├── chatbot.db                  # SQLite file for chat thread persistence
├── requirements.txt
└── .env
```

---

## How It Works

### Blog Writing Agent Flow

```
topic
  └─► router ──(needs research?)──► research (Tavily)
                                         │
                              ◄──────────┘
                              │
                         orchestrator (creates Plan with tasks)
                              │
                    ┌─────────┴─────────┐
                 worker              worker  ...  (parallel)
                    └─────────┬─────────┘
                          reducer
                    ┌─────────┴──────────────┐
               merge_content → decide_images → generate_and_place_images
                                                        │
                                                   final .md file
```

### RAG Chatbot Flow

```
user message
     └─► chat_node (LLM with tools bound)
               │
        (tool needed?)
               │
           tool_node ──► rag_tool / search / calculator / stock_price
               │
         chat_node (final answer)
```

---

## Usage

### Blog Writing Agent

1. Navigate to **Blog Writing Agent** from the home screen
2. Enter a topic in the sidebar (e.g. *"How transformers work"*)
3. Set the as-of date and click **Generate Blog**
4. Watch the live progress — routing, research, planning, writing, image generation
5. View the result across tabs: Plan, Evidence, Markdown Preview, Images, Logs
6. Download as `.md` or `.zip` bundle

### Multi Utility Chatbot

1. Navigate to **RAG With Tools** from the home screen
2. Optionally upload a PDF in the sidebar to enable document Q&A
3. Ask anything — the agent automatically picks the right tool
4. Use **New Chat** to start a fresh thread; past threads are listed in the sidebar
