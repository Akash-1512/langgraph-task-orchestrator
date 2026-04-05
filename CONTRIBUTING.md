# Contributing to langgraph-task-orchestrator

## Branch Strategy
main       ← releases only (v1.0.0, v2.0.0, v3.0.0)
develop    ← integration branch
feature/*  ← new features
fix/*      ← bug fixes
chore/*    ← maintenance

## Commit Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):
feat:     New agent, endpoint, or capability
fix:      Bug fix
chore:    Dependency, config, or setup change
ci:       CI/CD workflow changes
docs:     Documentation only
test:     Test files
refactor: Code restructure, no behavior change

## Pull Request Checklist

Before opening a PR:

- [ ] Branch is off `develop`, not `main`
- [ ] `--no-ff` merge to `develop`
- [ ] All tests pass locally
- [ ] DeepEval quality gate passes: `deepeval test run evaluation/test_deepeval.py`
- [ ] RAGAS gate passes: `python -m evaluation.ragas_eval`
- [ ] No secrets or API keys in code
- [ ] `.env.example` updated if new variables added
- [ ] `requirements.txt` updated if new dependencies added

## Local Setup
```bash
git clone https://github.com/Akash-1512/langgraph-task-orchestrator
cd langgraph-task-orchestrator
cp .env.example .env          # Fill in GROQ_API_KEY
uv pip install -r requirements.txt
python -m data.ingest_sec_filings     # Populate vector store
uvicorn api.main:app --reload         # Start API
streamlit run ui/app.py               # Start UI
```

## Running Tests
```bash
# Unit tests
python -m tests.test_state
python -m tests.test_llm_client
python -m tests.test_planner

# Quality gates
python -m evaluation.ragas_eval
deepeval test run evaluation/test_deepeval.py

# Real data integration
python -m tests.test_real_data
```