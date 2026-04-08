"""
Custom tool implementations for the AI Agent.
Each tool is defined as a dict (for the Claude API) and an async handler function.
"""

import os
import json
import io
import base64
from typing import Any
import anthropic

# ── Google API helpers ────────────────────────────────────────────────────────

def _get_google_creds():
    """Build Google credentials from the service-account JSON in the env."""
    try:
        from google.oauth2 import service_account
        sa_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
        if sa_file and os.path.exists(sa_file):
            scopes = [
                "https://www.googleapis.com/auth/documents",
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ]
            return service_account.Credentials.from_service_account_file(
                sa_file, scopes=scopes
            )
    except Exception:
        pass
    return None


def _docs_service():
    from googleapiclient.discovery import build
    creds = _get_google_creds()
    if not creds:
        raise RuntimeError("Google credentials not configured. Set GOOGLE_SERVICE_ACCOUNT_FILE.")
    return build("docs", "v1", credentials=creds)


def _sheets_service():
    from googleapiclient.discovery import build
    creds = _get_google_creds()
    if not creds:
        raise RuntimeError("Google credentials not configured. Set GOOGLE_SERVICE_ACCOUNT_FILE.")
    return build("sheets", "v4", credentials=creds)


def _drive_service():
    from googleapiclient.discovery import build
    creds = _get_google_creds()
    if not creds:
        raise RuntimeError("Google credentials not configured. Set GOOGLE_SERVICE_ACCOUNT_FILE.")
    return build("drive", "v3", credentials=creds)


# ── Tool definitions (passed to Claude) ──────────────────────────────────────

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "read_google_doc",
        "description": (
            "Read the full text content of a Google Document. "
            "Returns the document title and plain text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "The Google Docs document ID (from the URL).",
                }
            },
            "required": ["document_id"],
        },
    },
    {
        "name": "write_google_doc",
        "description": (
            "Insert text at the end of a Google Document, or create a brand-new document "
            "when no document_id is supplied."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Existing document ID. Omit to create a new document.",
                },
                "title": {
                    "type": "string",
                    "description": "Title for the new document (only used when creating).",
                },
                "content": {
                    "type": "string",
                    "description": "Text to insert / use as the document body.",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "read_google_sheet",
        "description": "Read cell values from a Google Spreadsheet range.",
        "input_schema": {
            "type": "object",
            "properties": {
                "spreadsheet_id": {
                    "type": "string",
                    "description": "The Google Sheets spreadsheet ID.",
                },
                "range": {
                    "type": "string",
                    "description": "A1 notation range, e.g. 'Sheet1!A1:D10'.",
                },
            },
            "required": ["spreadsheet_id", "range"],
        },
    },
    {
        "name": "write_google_sheet",
        "description": "Write values to a Google Spreadsheet range.",
        "input_schema": {
            "type": "object",
            "properties": {
                "spreadsheet_id": {
                    "type": "string",
                    "description": "The Google Sheets spreadsheet ID.",
                },
                "range": {
                    "type": "string",
                    "description": "A1 notation range to write into, e.g. 'Sheet1!A1'.",
                },
                "values": {
                    "type": "array",
                    "description": "2-D array of values (rows × columns).",
                    "items": {"type": "array"},
                },
            },
            "required": ["spreadsheet_id", "range", "values"],
        },
    },
    {
        "name": "analyze_uploaded_file",
        "description": (
            "Analyze a previously uploaded file (PDF, CSV, DOCX, TXT, image, …) "
            "using its file_id returned by the /upload endpoint. "
            "Claude will read the file contents and answer the given question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "The Anthropic Files API file ID (e.g. 'file_01...').",
                },
                "question": {
                    "type": "string",
                    "description": "What you want to know about the file.",
                },
            },
            "required": ["file_id", "question"],
        },
    },
    {
        "name": "search_uploaded_files",
        "description": (
            "Search across all uploaded files in the current session for content "
            "relevant to a query (RAG-style retrieval). "
            "Returns a summary of the most relevant passages."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search / question to answer from uploaded files.",
                },
                "file_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific file IDs to search. Leave empty to search all session files.",
                },
            },
            "required": ["query"],
        },
    },
]

# Built-in server-side tool (web search)
WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search"}

ALL_TOOLS = [WEB_SEARCH_TOOL] + TOOL_DEFINITIONS


# ── Tool handlers ─────────────────────────────────────────────────────────────

async def handle_tool(
    tool_name: str,
    tool_input: dict,
    session_files: list[dict],  # list of {file_id, filename, mime_type}
    client: anthropic.AsyncAnthropic,
) -> str:
    """Dispatch a tool call and return a string result."""
    try:
        if tool_name == "read_google_doc":
            return await _read_google_doc(tool_input["document_id"])
        elif tool_name == "write_google_doc":
            return await _write_google_doc(tool_input)
        elif tool_name == "read_google_sheet":
            return await _read_google_sheet(tool_input["spreadsheet_id"], tool_input["range"])
        elif tool_name == "write_google_sheet":
            return await _write_google_sheet(tool_input)
        elif tool_name == "analyze_uploaded_file":
            return await _analyze_file(tool_input["file_id"], tool_input["question"], client)
        elif tool_name == "search_uploaded_files":
            ids = tool_input.get("file_ids") or [f["file_id"] for f in session_files]
            return await _search_files(tool_input["query"], ids, client)
        else:
            return f"Unknown tool: {tool_name}"
    except Exception as exc:
        return f"Tool error ({tool_name}): {exc}"


# ── Individual implementations ────────────────────────────────────────────────

async def _read_google_doc(document_id: str) -> str:
    service = _docs_service()
    doc = service.documents().get(documentId=document_id).execute()
    title = doc.get("title", "")
    text_parts: list[str] = []
    for element in doc.get("body", {}).get("content", []):
        para = element.get("paragraph")
        if para:
            for run in para.get("elements", []):
                text_run = run.get("textRun")
                if text_run:
                    text_parts.append(text_run.get("content", ""))
    return f"Title: {title}\n\n{''.join(text_parts)}"


async def _write_google_doc(inp: dict) -> str:
    content: str = inp["content"]
    document_id: str | None = inp.get("document_id")
    title: str = inp.get("title", "New Document")

    docs = _docs_service()
    if not document_id:
        # Create new document
        doc = docs.documents().create(body={"title": title}).execute()
        document_id = doc["documentId"]

    # Read current end index
    doc = docs.documents().get(documentId=document_id).execute()
    end_index = doc["body"]["content"][-1]["endIndex"] - 1

    docs.documents().batchUpdate(
        documentId=document_id,
        body={
            "requests": [
                {
                    "insertText": {
                        "location": {"index": end_index},
                        "text": content,
                    }
                }
            ]
        },
    ).execute()
    return f"Document updated. ID: {document_id}"


async def _read_google_sheet(spreadsheet_id: str, range_: str) -> str:
    service = _sheets_service()
    result = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range=range_)
        .execute()
    )
    values = result.get("values", [])
    if not values:
        return "No data found in that range."
    rows = ["\t".join(str(c) for c in row) for row in values]
    return "\n".join(rows)


async def _write_google_sheet(inp: dict) -> str:
    service = _sheets_service()
    service.spreadsheets().values().update(
        spreadsheetId=inp["spreadsheet_id"],
        range=inp["range"],
        valueInputOption="USER_ENTERED",
        body={"values": inp["values"]},
    ).execute()
    return f"Sheet updated: {inp['range']}"


async def _analyze_file(file_id: str, question: str, client: anthropic.AsyncAnthropic) -> str:
    """Ask Claude directly about a specific file using the Files API."""
    response = await client.beta.messages.create(
        model="claude-opus-4-6",
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {"type": "file", "file_id": file_id},
                    },
                    {"type": "text", "text": question},
                ],
            }
        ],
        betas=["files-api-2025-04-14"],
    )
    return next(
        (b.text for b in response.content if b.type == "text"),
        "No answer produced.",
    )


async def _search_files(query: str, file_ids: list[str], client: anthropic.AsyncAnthropic) -> str:
    """RAG: search across multiple uploaded files."""
    if not file_ids:
        return "No files uploaded in this session."

    content_blocks: list[dict] = []
    for fid in file_ids[:5]:  # cap at 5 files to stay within context
        content_blocks.append(
            {"type": "document", "source": {"type": "file", "file_id": fid}}
        )
    content_blocks.append(
        {
            "type": "text",
            "text": (
                f"Based on the documents above, answer this query concisely:\n\n{query}\n\n"
                "Cite which document each piece of information comes from."
            ),
        }
    )

    response = await client.beta.messages.create(
        model="claude-opus-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": content_blocks}],
        betas=["files-api-2025-04-14"],
    )
    return next(
        (b.text for b in response.content if b.type == "text"),
        "No relevant content found.",
    )
