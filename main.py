"""
AI Agent Web Application
FastAPI backend — streaming chat with tool use, file upload, and RAG.
"""

import os
import json
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import anthropic
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from tools import ALL_TOOLS, TOOL_DEFINITIONS, handle_tool

load_dotenv()

# ── Globals ───────────────────────────────────────────────────────────────────

_client: anthropic.AsyncAnthropic | None = None

# In-memory session storage (keyed by session_id → list of messages & files)
# For production, replace with Redis / database.
_sessions: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")
    _client = anthropic.AsyncAnthropic(api_key=api_key)
    yield
    await _client.close()


app = FastAPI(title="AI Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static frontend
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str
    system_prompt: str = (
        "You are a powerful AI assistant with access to tools. "
        "You can search the web, analyze uploaded files (RAG), "
        "and read/write Google Docs and Spreadsheets. "
        "Always think step-by-step before using tools. "
        "When you use tools, explain what you're doing."
    )


class SessionResponse(BaseModel):
    session_id: str
    files: list[dict]
    message_count: int


# ── Helper ────────────────────────────────────────────────────────────────────

def _get_session(session_id: str) -> dict:
    if session_id not in _sessions:
        _sessions[session_id] = {"messages": [], "files": []}
    return _sessions[session_id]


async def _run_agent_loop(
    session: dict,
    user_message: str,
    system_prompt: str,
) -> AsyncGenerator[str, None]:
    """
    Agentic loop with streaming.
    Yields Server-Sent Events (SSE) as JSON strings.
    """
    assert _client is not None

    # Add user message to history
    session["messages"].append({"role": "user", "content": user_message})

    while True:
        # ── Stream one API call ───────────────────────────────────────────────
        full_content = []          # accumulated content blocks
        current_text = ""          # text accumulated within the current text block
        current_block_type = None  # "text" | "tool_use" | "server_tool_use" | …
        current_tool_use: dict | None = None
        input_json_acc = ""        # accumulate streamed tool input JSON

        async with _client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=4096,
            system=system_prompt,
            tools=ALL_TOOLS,
            messages=session["messages"],
            thinking={"type": "adaptive"},
        ) as stream:
            async for event in stream:
                etype = event.type

                # ── content_block_start ───────────────────────────────────────
                if etype == "content_block_start":
                    cb = event.content_block
                    current_block_type = cb.type
                    if cb.type == "text":
                        current_text = ""
                    elif cb.type == "tool_use":
                        current_tool_use = {
                            "type": "tool_use",
                            "id": cb.id,
                            "name": cb.name,
                            "input": {},
                        }
                        input_json_acc = ""
                        yield _sse({"type": "tool_start", "name": cb.name, "id": cb.id})

                # ── content_block_delta ───────────────────────────────────────
                elif etype == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        current_text += delta.text
                        yield _sse({"type": "text", "content": delta.text})
                    elif delta.type == "input_json_delta":
                        input_json_acc += delta.partial_json

                # ── content_block_stop ────────────────────────────────────────
                elif etype == "content_block_stop":
                    if current_block_type == "text" and current_text:
                        full_content.append({"type": "text", "text": current_text})
                        current_text = ""
                    elif current_block_type == "tool_use" and current_tool_use:
                        if input_json_acc:
                            try:
                                current_tool_use["input"] = json.loads(input_json_acc)
                            except json.JSONDecodeError:
                                current_tool_use["input"] = {}
                        full_content.append(current_tool_use)
                        current_tool_use = None
                        input_json_acc = ""
                    current_block_type = None

                # ── message_delta (stop_reason) ───────────────────────────────
                elif etype == "message_delta":
                    stop_reason = getattr(event.delta, "stop_reason", None)
                    if stop_reason:
                        yield _sse({"type": "stop_reason", "reason": stop_reason})

            # Retrieve the final message for stop_reason check
            final_message = await stream.get_final_message()

        # ── Append assistant turn to history ──────────────────────────────────
        session["messages"].append(
            {"role": "assistant", "content": full_content or [{"type": "text", "text": ""}]}
        )

        # ── Check whether we're done ──────────────────────────────────────────
        stop_reason = final_message.stop_reason
        if stop_reason == "end_turn":
            break

        if stop_reason == "pause_turn":
            # Server-side tools need continuation — loop without user input
            continue

        if stop_reason != "tool_use":
            break

        # ── Execute user-defined tools ────────────────────────────────────────
        tool_use_blocks = [b for b in full_content if b.get("type") == "tool_use"]
        if not tool_use_blocks:
            break

        tool_results = []
        for block in tool_use_blocks:
            tool_name = block["name"]
            tool_input = block["input"]
            tool_id = block["id"]

            yield _sse({"type": "tool_executing", "name": tool_name, "input": tool_input})

            result_text = await handle_tool(
                tool_name, tool_input, session["files"], _client
            )

            yield _sse({"type": "tool_result", "name": tool_name, "result": result_text[:500]})

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result_text,
                }
            )

        # Add tool results as a user message and loop
        session["messages"].append({"role": "user", "content": tool_results})

    yield _sse({"type": "done"})


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")


@app.post("/api/session")
async def create_session():
    import uuid
    sid = str(uuid.uuid4())[:8]
    _get_session(sid)
    return {"session_id": sid}


@app.get("/api/session/{session_id}")
async def get_session(session_id: str) -> SessionResponse:
    session = _get_session(session_id)
    return SessionResponse(
        session_id=session_id,
        files=session["files"],
        message_count=len(session["messages"]),
    )


@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    if session_id in _sessions:
        _sessions[session_id] = {"messages": [], "files": []}
    return {"status": "cleared"}


@app.post("/api/upload/{session_id}")
async def upload_file(
    session_id: str,
    file: UploadFile = File(...),
):
    """Upload a file to the Anthropic Files API and register it in the session."""
    assert _client is not None
    session = _get_session(session_id)

    content = await file.read()
    mime = file.content_type or "application/octet-stream"

    # Upload to Anthropic Files API
    uploaded = await _client.beta.files.upload(
        file=(file.filename or "upload", content, mime),
    )

    file_info = {
        "file_id": uploaded.id,
        "filename": file.filename or "upload",
        "mime_type": mime,
        "size": len(content),
    }
    session["files"].append(file_info)

    return {
        "status": "ok",
        "file_id": uploaded.id,
        "filename": file.filename,
        "size": len(content),
    }


@app.post("/api/chat/{session_id}")
async def chat(session_id: str, req: ChatRequest):
    """Stream an agentic chat response as SSE."""
    if req.session_id != session_id:
        raise HTTPException(400, "session_id mismatch")

    session = _get_session(session_id)

    return StreamingResponse(
        _run_agent_loop(session, req.message, req.system_prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/tools")
async def list_tools():
    return {"tools": [t["name"] for t in TOOL_DEFINITIONS] + ["web_search"]}
