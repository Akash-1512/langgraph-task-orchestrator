# langgraph-task-orchestrator
Multi-agent OKR analytics system with LangGraph, HITL, and LLM-as-Judge evaluation
# 🤖 langgraph-task-orchestrator

> Multi-agent OKR analytics system with LangGraph, HITL, and LLM-as-Judge evaluation.

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🏗️ Architecture
```
Planner → Research → Analytics → Critique → HITL Checkpoint → Output
                                    ↑              ↓ (quality gate failed)
                              hitl ← ← ← ← ← ← ←
                                ↓ (approved)
                              END
```

| Component | Demo Config | Production Config |
|-----------|------------|-------------------|
| LLM | Groq (llama-3.3-70b) | Azure OpenAI (gpt-4o) |
| Vector Store | Chroma (local) | Qdrant Cloud |
| Storage | Local filesystem | Cloudflare R2 |
| Database | Supabase | Azure Cosmos DB |
| Observability | Langfuse | Azure Monitor |

---

## 🚀 Quick Start

### 1. Clone and install
```bash
git clone https://github.com/Akash-1512/langgraph-task-orchestrator.git
cd langgraph-task-orchestrator
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Run the full stack
```bash
# Terminal 1 — FastAPI backend
uvicorn api.main:app --reload

# Terminal 2 — Streamlit UI
streamlit run ui/app.py
```

### 4. Run tests
```bash
python -m agents.test_state
python -m tests.test_llm_client
python -m tests.test_planner
python -m tests.test_research
python -m tests.test_analytics
python -m tests.test_critique
python -m tests.test_graph
```

---

## 📁 Project Structure
```
langgraph-task-orchestrator/
├── agents/          # Agent node definitions
│   ├── state.py     # AgentState TypedDict
│   ├── planner.py   # Planner agent
│   ├── research.py  # Research agent (RAG)
│   ├── analytics.py # Analytics agent
│   ├── critique.py  # Critique agent (LLM-as-Judge)
│   └── hitl.py      # HITL checkpoint node
├── graph/
│   └── agent_graph.py  # StateGraph assembly
├── core/
│   ├── llm_client.py   # Provider-agnostic LLM abstraction
│   └── retriever.py    # Vector store abstraction
├── api/
│   └── main.py      # FastAPI backend
├── ui/
│   └── app.py       # Streamlit dashboard
├── tests/           # Test suite
├── docs/            # Architecture documentation
├── .env.example     # Environment variable template
└── requirements.txt
```

---

## 🎬 Demo Video

> 📹 **[Watch 3-minute Loom demo](https://loom.com/share/placeholder)** — Full agent pipeline walkthrough with HITL approval

## ⚠️ Demo Notes

- First request to the hosted API may take ~30s (Render free tier cold start)
- Run locally via the Quick Start above for zero latency during interviews
- See `docs/architecture.md` for full production vs demo config comparison

---

## 📄 License

MIT © Akash Chaudhari
