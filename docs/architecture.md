# System Architecture

## Agent Graph Overview

Planner → Research → Analytics → Critique → HITL Checkpoint → Output

## Production Config
- LLM: Azure OpenAI (gpt-4o)
- Vector Store: Azure AI Search
- Storage: Azure Blob Storage
- Database: Azure Cosmos DB

## Demo Config
- LLM: Groq (llama-3.3-70b-versatile)
- Vector Store: Qdrant Cloud (free tier)
- Storage: Cloudflare R2 (free tier)
- Database: Supabase (free tier)

## Architecture Diagram
_Mermaid diagram to be added in feature/03-planner-agent-node_