"""
OpenEnv Server — Strategic Negotiation Environment
FastAPI + WebSocket server exposing reset(), step(), state() endpoints.
Compatible with the OpenEnv client protocol.
"""

import json
import os
import uuid
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from env_wrapper import EnvWrapper, Observation
from tasks import ALL_TASKS, TaskConfig, get_grader

# ─────────────────────────────────────────────
# Task Registry
# ─────────────────────────────────────────────

TASK_MAP = {task.name: task for task in ALL_TASKS}


# ─────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = "task_a_easy"


class StepRequest(BaseModel):
    action_type: str  # "OFFER", "ACCEPT", "REJECT"
    price: int = 0


# ─────────────────────────────────────────────
# Session Management
# ─────────────────────────────────────────────

class SessionState:
    """Holds a single environment episode."""

    def __init__(self, task_config: TaskConfig):
        self.env = EnvWrapper(
            opp_type=task_config.opp_type,
            a_val=task_config.agent_value,
            o_val=task_config.opponent_value,
            agent_role=task_config.agent_role,
            max_rounds=task_config.max_rounds,
        )
        self.task_config = task_config
        self.done = False
        self.rewards = []
        self.steps = 0
        self.deal_made = False


# In-memory session store (keyed by session_id)
sessions: dict[str, SessionState] = {}
MAX_SESSIONS = 200


def _cleanup_sessions():
    """Evict oldest sessions when limit is exceeded."""
    while len(sessions) > MAX_SESSIONS:
        oldest_key = next(iter(sessions))
        del sessions[oldest_key]


# ─────────────────────────────────────────────
# FastAPI Application
# ─────────────────────────────────────────────

app = FastAPI(
    title="Strategic Negotiation Environment",
    description="OpenEnv-compliant negotiation simulation where AI agents learn to negotiate under uncertainty.",
    version="1.0.0",
)


# ── Health & Info ──

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/")
async def root():
    """Environment info and available tasks."""
    return {
        "status": "running",
        "environment": "negotiation-env",
        "version": "1.0.0",
        "tasks": [
            {
                "name": t.name,
                "difficulty": t.difficulty,
                "description": t.description,
                "success_threshold": t.success_threshold,
            }
            for t in ALL_TASKS
        ],
    }


# ── HTTP Endpoints ──

@app.post("/reset")
async def reset(request: ResetRequest = None):
    """Reset the environment and start a new episode."""
    if request is None:
        request = ResetRequest()

    task_name = request.task
    if task_name not in TASK_MAP:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown task: {task_name}. Available: {list(TASK_MAP.keys())}"},
        )

    task_config = TASK_MAP[task_name]
    session_id = str(uuid.uuid4())
    session = SessionState(task_config)
    obs = session.env.reset()
    sessions[session_id] = session
    _cleanup_sessions()

    return {
        "session_id": session_id,
        "observation": obs.model_dump(),
        "reward": 0.0,
        "done": False,
        "info": {
            "task": task_name,
            "difficulty": task_config.difficulty,
            "max_rounds": task_config.max_rounds,
        },
    }


@app.post("/step")
async def step(request: StepRequest, session_id: str = Query(...)):
    """Execute one step in the environment."""
    if session_id not in sessions:
        return JSONResponse(
            status_code=404,
            content={"error": "Session not found. Call POST /reset first."},
        )

    session = sessions[session_id]

    if session.done:
        obs = session.env.state()
        return {
            "observation": obs.model_dump(),
            "reward": 0.0,
            "done": True,
            "info": {"error": "Episode already ended. Call /reset for a new one."},
        }

    # Parse action
    action_type = request.action_type.upper()
    price = request.price

    if action_type == "OFFER":
        action_str = f"OFFER {price}"
    else:
        action_str = action_type
        price = 0

    # Step environment
    obs, reward, done, info = session.env.step(action_str, price)
    session.done = done
    session.rewards.append(reward)
    session.steps += 1

    if done and info.get("deal_type") in ("agent_accepted", "opponent_accepted"):
        session.deal_made = True

    # If done, compute final graded score
    if done:
        grader = get_grader(session.task_config)
        grade_result = grader.grade(session.rewards, session.steps, session.deal_made)
        info["grader_score"] = grade_result["score"]
        info["grader_success"] = grade_result["success"]
        info["threshold"] = grade_result["threshold"]

    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
async def state(session_id: str = Query(...)):
    """Get current environment state without taking an action."""
    if session_id not in sessions:
        return JSONResponse(
            status_code=404,
            content={"error": "Session not found."},
        )

    session = sessions[session_id]
    obs = session.env.state()

    return {
        "observation": obs.model_dump(),
        "done": session.done,
        "info": {
            "task": session.task_config.name,
            "steps": session.steps,
        },
    }


# ─────────────────────────────────────────────
# WebSocket Endpoint (persistent session)
# ─────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket endpoint for persistent environment sessions."""
    await ws.accept()
    session: Optional[SessionState] = None

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            msg_type = msg.get("type", "")

            if msg_type == "reset":
                task_name = msg.get("task", "task_a_easy")
                if task_name not in TASK_MAP:
                    await ws.send_json({"error": f"Unknown task: {task_name}. Available: {list(TASK_MAP.keys())}"})
                    continue

                task_config = TASK_MAP[task_name]
                session = SessionState(task_config)
                obs = session.env.reset()

                await ws.send_json({
                    "type": "reset",
                    "observation": obs.model_dump(),
                    "reward": 0.0,
                    "done": False,
                    "info": {
                        "task": task_name,
                        "difficulty": task_config.difficulty,
                        "max_rounds": task_config.max_rounds,
                    },
                })

            elif msg_type == "step":
                if session is None:
                    await ws.send_json({"error": "No active session. Send a reset message first."})
                    continue

                action = msg.get("action", {})
                action_type = action.get("action_type", "REJECT").upper()
                price = action.get("price", 0)

                if action_type == "OFFER":
                    action_str = f"OFFER {price}"
                else:
                    action_str = action_type
                    price = 0

                obs, reward, done, info = session.env.step(action_str, price)
                session.done = done
                session.rewards.append(reward)
                session.steps += 1

                if done and info.get("deal_type") in ("agent_accepted", "opponent_accepted"):
                    session.deal_made = True

                if done:
                    grader = get_grader(session.task_config)
                    grade_result = grader.grade(session.rewards, session.steps, session.deal_made)
                    info["grader_score"] = grade_result["score"]
                    info["grader_success"] = grade_result["success"]
                    info["threshold"] = grade_result["threshold"]

                await ws.send_json({
                    "type": "step",
                    "observation": obs.model_dump(),
                    "reward": reward,
                    "done": done,
                    "info": info,
                })

            elif msg_type == "state":
                if session is None:
                    await ws.send_json({"error": "No active session."})
                    continue

                obs = session.env.state()
                await ws.send_json({
                    "type": "state",
                    "observation": obs.model_dump(),
                    "done": session.done,
                    "info": {
                        "task": session.task_config.name,
                        "steps": session.steps,
                    },
                })

            else:
                await ws.send_json({"error": f"Unknown message type: {msg_type}. Use: reset, step, state"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass


# ─────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
