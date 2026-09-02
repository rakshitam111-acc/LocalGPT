"""SQLite database persistence layer for LocalGPT conversations and messages."""

import json
import os
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

DB_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DB_DIR, "localgpt.db")
DEFAULT_SYSTEM_PROMPT = "You are LocalGPT, a helpful, intelligent, and concise AI assistant running 100% locally."


def get_connection() -> sqlite3.Connection:
    """Create and return a database connection, ensuring directories exist."""
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(os.path.join(DB_DIR, "documents"), exist_ok=True)
    os.makedirs(os.path.join(DB_DIR, "conversations"), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables if they do not exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            system_prompt TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            sources_json TEXT,
            xray_json TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()


def create_conversation(title: str = "New Chat", system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """Create a new conversation session and return its unique ID."""
    init_db()
    conv_id = str(uuid.uuid4())
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO conversations (id, title, system_prompt, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (conv_id, title, system_prompt, now, now),
    )
    conn.commit()
    conn.close()
    return conv_id


def get_conversations() -> List[Dict[str, Any]]:
    """Retrieve all conversations sorted by updated_at descending."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, system_prompt, created_at, updated_at FROM conversations ORDER BY updated_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_conversation(conv_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single conversation by ID."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, system_prompt, created_at, updated_at FROM conversations WHERE id = ?", (conv_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def rename_conversation(conv_id: str, new_title: str):
    """Rename a conversation."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
        (new_title, datetime.now().isoformat(), conv_id),
    )
    conn.commit()
    conn.close()


def update_system_prompt(conv_id: str, system_prompt: str):
    """Update system prompt for a conversation."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE conversations SET system_prompt = ?, updated_at = ? WHERE id = ?",
        (system_prompt, datetime.now().isoformat(), conv_id),
    )
    conn.commit()
    conn.close()


def delete_conversation(conv_id: str):
    """Delete a conversation and all its messages."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
    cursor.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    conn.commit()
    conn.close()


def add_message(
    conv_id: str,
    role: str,
    content: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    xray_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Add a message to a conversation."""
    init_db()
    msg_id = str(uuid.uuid4())
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    sources_json = json.dumps(sources) if sources else None
    xray_json = json.dumps(xray_data) if xray_data else None

    cursor.execute(
        """
        INSERT INTO messages (id, conversation_id, role, content, timestamp, sources_json, xray_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (msg_id, conv_id, role, content, now, sources_json, xray_json),
    )

    # Update conversation's updated_at timestamp
    cursor.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))

    conn.commit()
    conn.close()
    return msg_id


def get_messages(conv_id: str) -> List[Dict[str, Any]]:
    """Retrieve all messages for a conversation ordered chronologically."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, conversation_id, role, content, timestamp, sources_json, xray_json FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC",
        (conv_id,),
    )
    rows = cursor.fetchall()
    conn.close()

    result = []
    for r in rows:
        item = dict(r)
        item["sources"] = json.loads(item["sources_json"]) if item["sources_json"] else []
        item["xray"] = json.loads(item["xray_json"]) if item["xray_json"] else None
        result.append(item)
    return result


def delete_messages_from(conv_id: str, msg_id: str):
    """Delete a specific message and all subsequent messages in the conversation (useful for edit/regenerate)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp FROM messages WHERE id = ? AND conversation_id = ?", (msg_id, conv_id))
    row = cursor.fetchone()
    if row:
        target_time = row["timestamp"]
        cursor.execute(
            "DELETE FROM messages WHERE conversation_id = ? AND timestamp >= ?",
            (conv_id, target_time),
        )
        conn.commit()
    conn.close()
