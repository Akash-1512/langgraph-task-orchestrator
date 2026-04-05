![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![CI](https://github.com/Akash-1512/langgraph-task-orchestrator/actions/workflows/ci.yml/badge.svg)
![RAGAS](https://img.shields.io/badge/RAGAS-Faithfulness%201.0-brightgreen)
![DeepEval](https://img.shields.io/badge/DeepEval-Pass%20100%25-brightgreen)

# 🤖 Multi-Agent OKR Analytics Orchestrator

> Multi-agent OKR analytics system with LangGraph, HITL, and LLM-as-Judge evaluation.

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

## 🎯 Key Interview Talking Points

**HITL Pattern:**
> "I implemented a HITL interrupt pattern in LangGraph where the graph state is persisted to a checkpointer before pausing — the agent resumes from exactly where it stopped after human review, with zero state loss."

**Quality Gate:**
> "Every release is gated by LLM-as-Judge scores — faithfulness, coherence, and task completion. If the overall score drops below 0.75, the graph routes back to the analytics agent automatically."

**Provider Abstraction:**
> "The demo runs on Groq for cost efficiency, but the same code deploys to Azure OpenAI by changing one environment variable. Vendor lock-in is a real enterprise risk — this architecture avoids it intentionally."

---

## 🎬 Demo Video

> 🎬 **[Watch full demo video](https://youtu.be/i4-y1cYw99I)** — Architecture walkthrough + live HITL approval

## ⚠️ Demo Notes

- First request to the hosted API may take ~30s (Render free tier cold start)
- Run locally via the Quick Start above for zero latency during interviews
- See `docs/architecture.md` for full production vs demo config comparison

---

## 📄 License

MIT © Akash Chaudhari