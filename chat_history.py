"""
chat_history.py
---------------
SQLite persistence layer for NutriTrack AI.

Tables
------
  chat_sessions   – one row per conversation session
  chat_messages   – individual messages per session
  food_log        – logged meals (one row per meal entry, keyed by date)

All functions accept/return plain Python dicts so the Streamlit layer
never imports sqlite3 directly.
"""

import sqlite3
import json
from datetime import datetime, date
from pathlib import Path

DB_PATH = Path(__file__).parent / "nutritrack.db"


# ── helpers ──────────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    return con


def init_db() -> None:
    """Create tables if they do not already exist. Call once at app startup."""
    with _conn() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id          TEXT PRIMARY KEY,
                title       TEXT NOT NULL DEFAULT 'New chat',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                preview     TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                role        TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content     TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS food_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                log_date     TEXT NOT NULL,          -- ISO date YYYY-MM-DD
                logged_at    TEXT NOT NULL,           -- ISO datetime
                label        TEXT NOT NULL,
                meal_type    TEXT DEFAULT 'meal',
                food_results TEXT NOT NULL DEFAULT '[]',   -- JSON
                calories     REAL DEFAULT 0,
                protein      REAL DEFAULT 0,
                carbs        REAL DEFAULT 0,
                fat          REAL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON chat_messages(session_id);

            CREATE INDEX IF NOT EXISTS idx_food_log_date
                ON food_log(log_date);
            """
        )


# ── chat sessions ─────────────────────────────────────────────────────────────

def create_session(session_id: str, title: str = "New chat") -> dict:
    now = datetime.now().isoformat()
    with _conn() as con:
        con.execute(
            "INSERT INTO chat_sessions (id, title, created_at, updated_at) VALUES (?,?,?,?)",
            (session_id, title, now, now),
        )
    return get_session(session_id)


def get_session(session_id: str) -> dict | None:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM chat_sessions WHERE id = ?", (session_id,)
        ).fetchone()
    return dict(row) if row else None


def list_sessions() -> list[dict]:
    """Return all sessions, newest first."""
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM chat_sessions ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def update_session_meta(session_id: str, title: str, preview: str = "") -> None:
    with _conn() as con:
        con.execute(
            "UPDATE chat_sessions SET title=?, preview=?, updated_at=? WHERE id=?",
            (title, preview, datetime.now().isoformat(), session_id),
        )


def delete_session(session_id: str) -> None:
    with _conn() as con:
        con.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))


# ── chat messages ─────────────────────────────────────────────────────────────

def add_message(session_id: str, role: str, content: str) -> dict:
    now = datetime.now().isoformat()
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO chat_messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
            (session_id, role, content, now),
        )
        msg_id = cur.lastrowid
        con.execute(
            "UPDATE chat_sessions SET updated_at=? WHERE id=?",
            (now, session_id),
        )
    return {"id": msg_id, "session_id": session_id, "role": role, "content": content, "created_at": now}


def get_messages(session_id: str) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM chat_messages WHERE session_id=? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ── food log ──────────────────────────────────────────────────────────────────

def log_meal(
    label: str,
    food_results: list,
    totals: dict,
    meal_type: str = "meal",
    log_date: str | None = None,
) -> dict:
    """
    Persist a logged meal entry.

    Parameters
    ----------
    label        : human-readable meal description / filename
    food_results : raw list of food dicts from pipeline
    totals       : {'calories': float, 'protein': float, 'carbs': float, 'fat': float}
    meal_type    : 'breakfast' | 'lunch' | 'dinner' | 'snack' | 'meal'
    log_date     : ISO date string; defaults to today
    """
    if log_date is None:
        log_date = date.today().isoformat()
    now = datetime.now().isoformat()
    with _conn() as con:
        cur = con.execute(
            """
            INSERT INTO food_log
                (log_date, logged_at, label, meal_type, food_results,
                 calories, protein, carbs, fat)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                log_date,
                now,
                label,
                meal_type,
                json.dumps(food_results),
                totals.get("calories", 0),
                totals.get("protein", 0),
                totals.get("carbs", 0),
                totals.get("fat", 0),
            ),
        )
        entry_id = cur.lastrowid
    return get_meal_entry(entry_id)


def get_meal_entry(entry_id: int) -> dict | None:
    with _conn() as con:
        row = con.execute("SELECT * FROM food_log WHERE id=?", (entry_id,)).fetchone()
    if not row:
        return None
    entry = dict(row)
    entry["food_results"] = json.loads(entry["food_results"])
    return entry


def get_meals_for_date(log_date: str) -> list[dict]:
    """Return all meals logged on a given ISO date, ordered by time."""
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM food_log WHERE log_date=? ORDER BY logged_at ASC",
            (log_date,),
        ).fetchall()
    entries = []
    for r in rows:
        entry = dict(r)
        entry["food_results"] = json.loads(entry["food_results"])
        entries.append(entry)
    return entries


def get_logged_dates() -> list[str]:
    """Return all dates that have at least one entry, newest first."""
    with _conn() as con:
        rows = con.execute(
            "SELECT DISTINCT log_date FROM food_log ORDER BY log_date DESC"
        ).fetchall()
    return [r["log_date"] for r in rows]


def delete_meal_entry(entry_id: int) -> None:
    with _conn() as con:
        con.execute("DELETE FROM food_log WHERE id=?", (entry_id,))


def get_daily_totals_db(log_date: str) -> dict:
    """Aggregate macros for a given date directly from the DB."""
    with _conn() as con:
        row = con.execute(
            """
            SELECT
                ROUND(SUM(calories))         AS calories,
                ROUND(SUM(protein) * 10)/10  AS protein,
                ROUND(SUM(carbs)   * 10)/10  AS carbs,
                ROUND(SUM(fat)     * 10)/10  AS fat
            FROM food_log
            WHERE log_date = ?
            """,
            (log_date,),
        ).fetchone()
    return dict(row) if row else {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
