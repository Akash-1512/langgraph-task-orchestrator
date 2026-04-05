"""
api/main.py

FastAPI backend for the langgraph-task-orchestrator.
Exposes endpoints to run the agent graph and handle HITL approval.

Endpoints:
    POST /run     — Start a new graph run
    POST /approve — Approve or revise HITL checkpoint
    GET  /state   — Get current graph state
    GET  /ping    — Keep-alive for Render free tier
"""

import uuid
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from langgraph.types import Command
import asyncio
import json
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from graph.agent_graph import graph

app = FastAPI(
    title="langgraph-task-orchestrator",
    description="Multi-agent OKR analytics system with HITL",
    version="1.0.0"
)


from pydantic import BaseModel, validator

class RunRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None

    @validator("query")
    def query_must_be_valid(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("query cannot be empty")
        if len(v) > 2000:
            raise ValueError("query exceeds 2000 character limit")
        return v


class ApproveRequest(BaseModel):
    thread_id: str
    action: str  # "approve" or revision feedback

    @validator("thread_id")
    def thread_id_must_be_valid(cls, v):
        if not v or len(v) > 100:
            raise ValueError("invalid thread_id")
        return v

    @validator("action")
    def action_must_be_valid(cls, v):
        if not v or len(v) > 1000:
            raise ValueError("action too long or empty")
        return v
    
class RunRequest(BaseModel):
    query: str
    thread_id: str = None


class HITLRequest(BaseModel):
    thread_id: str
    action: str  # "approve" or revision feedback text


@app.get("/ping")
def ping():
    """Keep-alive endpoint for Render free tier."""
    return {"status": "alive"}


@app.post("/run")
def run_graph(request: RunRequest):
    """Start a new agent graph run. Returns when HITL interrupt is hit."""
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_input = {
        "query": request.query,
        "messages": [],
        "plan": None,
        "research_context": None,
        "retrieved_sources": None,
        "analytics_result": None,
        "critique": None,
        "hitl_status": "pending",
        "hitl_feedback": None,
        "final_output": None,
        "run_metadata": None,
        "error": None,
    }

    events = []
    interrupt_data = None

    for event in graph.stream(initial_input, config=config):
        node_name = list(event.keys())[0]
        if node_name == "__interrupt__":
            interrupt_data = event["__interrupt__"][0].value
        else:
            events.append(node_name)

    return {
        "thread_id": thread_id,
        "nodes_completed": events,
        "hitl_interrupt": interrupt_data,
        "status": "awaiting_hitl" if interrupt_data else "completed"
    }


@app.post("/approve")
def handle_hitl(request: HITLRequest):
    """Resume graph after HITL checkpoint with approve or revision."""
    config = {"configurable": {"thread_id": request.thread_id}}

    try:
        events = []
        for event in graph.stream(Command(resume=request.action), config=config):
            node_name = list(event.keys())[0]
            events.append(node_name)

        final_state = graph.get_state(config)
        return {
            "thread_id": request.thread_id,
            "hitl_status": final_state.values.get("hitl_status"),
            "final_output": final_state.values.get("final_output"),
            "nodes_completed": events,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state/{thread_id}")
def get_state(thread_id: str):
    """Get current graph state for a thread."""
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    if not state.values:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {
        "thread_id": thread_id,
        "hitl_status": state.values.get("hitl_status"),
        "analytics_result": state.values.get("analytics_result"),
        "critique": state.values.get("critique"),
        "final_output": state.values.get("final_output"),
    }

@app.websocket("/ws/run")
async def websocket_run(websocket: WebSocket):
    """
    WebSocket endpoint for real-time graph state streaming.
    Streams each node completion event as it happens.

    Usage (JavaScript):
        const ws = new WebSocket('ws://localhost:8000/ws/run');
        ws.send(JSON.stringify({query: "Analyze Apple Q1 performance"}));
        ws.onmessage = (e) => console.log(JSON.parse(e.data));
    """
    await websocket.accept()
    try:
        data = await websocket.receive_text()
        request = json.loads(data)
        query = request.get("query", "")
        thread_id = request.get("thread_id", str(uuid.uuid4()))
        config = {"configurable": {"thread_id": thread_id}}

        initial_input = {
            "query": query,
            "messages": [],
            "plan": None,
            "research_context": None,
            "retrieved_sources": None,
            "analytics_result": None,
            "critique": None,
            "hitl_status": "pending",
            "hitl_feedback": None,
            "final_output": None,
            "run_metadata": None,
            "error": None,
        }

        await websocket.send_text(json.dumps({
            "type": "start",
            "thread_id": thread_id
        }))

        for event in graph.stream(initial_input, config=config):
            node_name = list(event.keys())[0]
            if node_name == "__interrupt__":
                interrupt_data = event["__interrupt__"][0].value
                await websocket.send_text(json.dumps({
                    "type": "interrupt",
                    "node": "hitl",
                    "data": interrupt_data
                }))
            else:
                await websocket.send_text(json.dumps({
                    "type": "node_complete",
                    "node": node_name,
                }))
            await asyncio.sleep(0)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": str(e)
        }))