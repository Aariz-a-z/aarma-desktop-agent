"""SQLite access layer with a forward-only migration ladder.

Phase 1 deliberately keeps the schema small (specification section 28 warns
against over-engineering it).  Only tables that are actually written to in this
phase exist:

``schema_version``   -- migration bookkeeping.
``kv_store``         -- small runtime values that are not user settings.
``conversation``     -- the transcript, so the dashboard survives a restart.
``action_history``   -- audit trail: what ran, at what risk, and whether it worked.
``permission_audit`` -- every allow/deny decision, for the security review.

Tables for memories, reminders and calendar sync are added by later migrations
when the features that use them are actually built.

Connections are per-thread.  The GUI thread and the engine's worker both touch
the database, and a single ``sqlite3.Connection`` is not safe to share.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, List, Mapping, Optional, Sequence

from utils.logger import get_logger
from utils.paths import get_paths

logger = get_logger("database")

SCHEMA_VERSION = 3

_MIGRATIONS: dict[int, tuple[str, ...]] = {
    1: (
        """
        CREATE TABLE IF NOT EXISTS kv_store (
            key         TEXT PRIMARY KEY,
            value       TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS conversation (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            role        TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
            text        TEXT NOT NULL,
            language    TEXT NOT NULL DEFAULT 'en'
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_conversation_session ON conversation (session_id, id)",
        """
        CREATE TABLE IF NOT EXISTS action_history (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at   TEXT NOT NULL,
            session_id   TEXT NOT NULL,
            action       TEXT NOT NULL,
            parameters   TEXT NOT NULL DEFAULT '{}',
            risk         TEXT NOT NULL,
            source       TEXT NOT NULL,
            allowed      INTEGER NOT NULL,
            succeeded    INTEGER NOT NULL,
            message      TEXT NOT NULL DEFAULT '',
            duration_ms  INTEGER NOT NULL DEFAULT 0
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_action_history_time ON action_history (created_at)",
        """
        CREATE TABLE IF NOT EXISTS permission_audit (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at  TEXT NOT NULL,
            permission  TEXT NOT NULL,
            action      TEXT NOT NULL,
            decision    TEXT NOT NULL,
            reason      TEXT NOT NULL DEFAULT ''
        )
        """,
    ),
    2: (
        """
        CREATE TABLE IF NOT EXISTS reminders (
            id           TEXT PRIMARY KEY,
            created_at   TEXT NOT NULL,
            due_at       TEXT NOT NULL,
            title        TEXT NOT NULL,
            message      TEXT NOT NULL DEFAULT '',
            recurrence   TEXT NOT NULL DEFAULT '',
            state        TEXT NOT NULL DEFAULT 'pending'
                         CHECK (state IN ('pending', 'done', 'cancelled')),
            notified_at  TEXT NOT NULL DEFAULT ''
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders (state, due_at)",
        """
        CREATE TABLE IF NOT EXISTS calendar_events (
            id           TEXT PRIMARY KEY,
            created_at   TEXT NOT NULL,
            starts_at    TEXT NOT NULL,
            ends_at      TEXT NOT NULL,
            title        TEXT NOT NULL,
            location     TEXT NOT NULL DEFAULT '',
            notes        TEXT NOT NULL DEFAULT '',
            provider     TEXT NOT NULL DEFAULT 'local',
            state        TEXT NOT NULL DEFAULT 'confirmed'
                         CHECK (state IN ('confirmed', 'cancelled'))
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_events_start ON calendar_events (state, starts_at)",
    ),
    # Migration 3 (Phase 3) supports the reminder scheduler and email drafts.
    #
    # ``outcome`` records *how* a reminder finished, which ``state`` alone
    # cannot: a reminder that rang, one that rang late, and one the scheduler
    # found hours stale and deliberately did not ring are all "done", and
    # collapsing them would make a missed reminder indistinguishable from a
    # delivered one. Added as a column rather than a new ``state`` value
    # because SQLite cannot alter a CHECK constraint without rebuilding the
    # table, and rebuilding a table that holds live user data to gain a label
    # is a poor trade.
    3: (
        "ALTER TABLE reminders ADD COLUMN outcome TEXT NOT NULL DEFAULT ''",
        """
        CREATE TABLE IF NOT EXISTS email_drafts (
            id           TEXT PRIMARY KEY,
            created_at   TEXT NOT NULL,
            recipients   TEXT NOT NULL DEFAULT '',
            subject      TEXT NOT NULL DEFAULT '',
            body         TEXT NOT NULL DEFAULT '',
            provider     TEXT NOT NULL DEFAULT 'local-drafts',
            state        TEXT NOT NULL DEFAULT 'draft'
                         CHECK (state IN ('draft', 'sent', 'discarded')),
            -- Empty unless a provider *confirmed* delivery. Nothing in this
            -- build can set it, which is the point: there is no transport.
            sent_at      TEXT NOT NULL DEFAULT ''
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_drafts_state ON email_drafts (state, created_at)",
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ActionHistoryEntry:
    """One row of the audit trail, shaped for the dashboard's activity list."""

    created_at: str
    action: str
    risk: str
    source: str
    allowed: bool
    succeeded: bool
    message: str


class Database:
    """Thread-safe SQLite wrapper.

    Every public method opens (or reuses) the calling thread's connection, so
    the class can be handed to the GUI and to background workers alike.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or get_paths().database_file
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._initialised = False

    @property
    def path(self) -> Path:
        return self._path

    # ---------------------------------------------------------- connections

    def _connect(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self._path, timeout=10.0)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA foreign_keys = ON")
            self._local.connection = connection
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run statements in a transaction, rolling back on any exception."""
        connection = self._connect()
        try:
            with connection:
                yield connection
        except sqlite3.Error:
            logger.exception("Database transaction failed")
            raise

    def close(self) -> None:
        """Close this thread's connection, if it has one."""
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None

    # ------------------------------------------------------------ migrations

    def initialise(self) -> int:
        """Create or upgrade the schema.  Returns the resulting version."""
        with self._init_lock:
            connection = self._connect()
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_version ("
                " version INTEGER NOT NULL, applied_at TEXT NOT NULL)"
            )
            connection.commit()
            current = self.schema_version()
            for version in sorted(_MIGRATIONS):
                if version <= current:
                    continue
                logger.info("Applying database migration", extra={"context": {"version": version}})
                with connection:
                    for statement in _MIGRATIONS[version]:
                        connection.execute(statement)
                    connection.execute(
                        "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                        (version, _utc_now()),
                    )
                current = version
            self._initialised = True
            logger.info("Database ready",
                        extra={"context": {"path": str(self._path), "schema_version": current}})
            return current

    def schema_version(self) -> int:
        row = self._connect().execute(
            "SELECT MAX(version) AS version FROM schema_version"
        ).fetchone()
        return int(row["version"]) if row and row["version"] is not None else 0

    # ------------------------------------------------------------- reminders
    #
    # Times are stored as ISO-8601 strings *with* an offset.  SQLite has no
    # datetime type, and a naive string here would silently become "some time,
    # in whichever timezone the reader assumes" -- which for a reminder is the
    # difference between 6 PM and the middle of the night.

    def add_reminder(self, *, reminder_id: str, title: str, due_at: str,
                     message: str = "", recurrence: str = "") -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO reminders (id, created_at, due_at, title, message,"
                " recurrence, state) VALUES (?, ?, ?, ?, ?, ?, 'pending')",
                (reminder_id, _utc_now(), due_at, title, message, recurrence),
            )

    def reminders(self, *, state: Optional[str] = "pending",
                  limit: int = 100) -> List[sqlite3.Row]:
        query = "SELECT * FROM reminders"
        params: list = []
        if state:
            query += " WHERE state = ?"
            params.append(state)
        query += " ORDER BY due_at ASC LIMIT ?"
        params.append(int(limit))
        return list(self._connect().execute(query, params).fetchall())

    def reminder(self, reminder_id: str) -> Optional[sqlite3.Row]:
        return self._connect().execute(
            "SELECT * FROM reminders WHERE id = ?", (reminder_id,)).fetchone()

    def set_reminder_state(self, reminder_id: str, state: str) -> bool:
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE reminders SET state = ? WHERE id = ? AND state = 'pending'",
                (state, reminder_id))
        return cursor.rowcount > 0

    def mark_reminder_notified(self, reminder_id: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE reminders SET notified_at = ? WHERE id = ?",
                (_utc_now(), reminder_id))

    def finish_reminder(self, reminder_id: str, *, outcome: str,
                        state: str = "done") -> bool:
        """Close a reminder out, recording how it ended.

        Guarded on ``state = 'pending'`` so a reminder the user cancelled a
        moment before the scheduler reached it is not resurrected as "fired".
        """
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE reminders SET state = ?, outcome = ?, notified_at = ?"
                " WHERE id = ? AND state = 'pending'",
                (state, outcome, _utc_now(), reminder_id))
        return cursor.rowcount > 0

    def reminders_by_outcome(self, outcome: str,
                             limit: int = 50) -> List[sqlite3.Row]:
        return list(self._connect().execute(
            "SELECT * FROM reminders WHERE outcome = ? ORDER BY due_at DESC"
            " LIMIT ?", (outcome, int(limit))).fetchall())

    # ----------------------------------------------------------- email drafts

    def add_email_draft(self, *, draft_id: str, recipients: str, subject: str,
                        body: str, provider: str = "local-drafts") -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO email_drafts (id, created_at, recipients, subject,"
                " body, provider, state) VALUES (?, ?, ?, ?, ?, ?, 'draft')",
                (draft_id, _utc_now(), recipients, subject, body, provider))

    def email_drafts(self, *, state: str = "draft",
                     limit: int = 20) -> List[sqlite3.Row]:
        return list(self._connect().execute(
            "SELECT * FROM email_drafts WHERE state = ? ORDER BY created_at DESC"
            " LIMIT ?", (state, int(limit))).fetchall())

    def email_draft(self, draft_id: str) -> Optional[sqlite3.Row]:
        return self._connect().execute(
            "SELECT * FROM email_drafts WHERE id = ?", (draft_id,)).fetchone()

    def discard_email_draft(self, draft_id: str) -> bool:
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE email_drafts SET state = 'discarded'"
                " WHERE id = ? AND state = 'draft'", (draft_id,))
        return cursor.rowcount > 0

    # -------------------------------------------------------------- calendar

    def add_event(self, *, event_id: str, title: str, starts_at: str,
                  ends_at: str, location: str = "", notes: str = "",
                  provider: str = "local") -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO calendar_events (id, created_at, starts_at, ends_at,"
                " title, location, notes, provider, state)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'confirmed')",
                (event_id, _utc_now(), starts_at, ends_at, title, location,
                 notes, provider),
            )

    def events(self, *, since: Optional[str] = None, until: Optional[str] = None,
               limit: int = 100) -> List[sqlite3.Row]:
        query = "SELECT * FROM calendar_events WHERE state = 'confirmed'"
        params: list = []
        if since:
            query += " AND ends_at >= ?"
            params.append(since)
        if until:
            query += " AND starts_at <= ?"
            params.append(until)
        query += " ORDER BY starts_at ASC LIMIT ?"
        params.append(int(limit))
        return list(self._connect().execute(query, params).fetchall())

    def event(self, event_id: str) -> Optional[sqlite3.Row]:
        return self._connect().execute(
            "SELECT * FROM calendar_events WHERE id = ?", (event_id,)).fetchone()

    def cancel_event(self, event_id: str) -> bool:
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE calendar_events SET state = 'cancelled'"
                " WHERE id = ? AND state = 'confirmed'", (event_id,))
        return cursor.rowcount > 0

    # --------------------------------------------------------------- kv_store

    def set_value(self, key: str, value: Any) -> None:
        payload = json.dumps(value, default=str)
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO kv_store (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                "updated_at = excluded.updated_at",
                (key, payload, _utc_now()),
            )

    def get_value(self, key: str, default: Any = None) -> Any:
        row = self._connect().execute(
            "SELECT value FROM kv_store WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except ValueError:
            return default

    # ----------------------------------------------------------- conversation

    def add_message(self, session_id: str, role: str, text: str, language: str = "en") -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO conversation (session_id, created_at, role, text, language) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, _utc_now(), role, text, language),
            )

    def recent_messages(self, limit: int = 50,
                        session_id: Optional[str] = None) -> List[sqlite3.Row]:
        if session_id:
            rows = self._connect().execute(
                "SELECT * FROM conversation WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        else:
            rows = self._connect().execute(
                "SELECT * FROM conversation ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return list(reversed(rows))

    def clear_conversation(self) -> int:
        with self.transaction() as connection:
            cursor = connection.execute("DELETE FROM conversation")
            return cursor.rowcount

    # --------------------------------------------------------- action history

    def record_action(self, *, session_id: str, action: str, parameters: Mapping[str, Any],
                      risk: str, source: str, allowed: bool, succeeded: bool,
                      message: str = "", duration_ms: int = 0) -> None:
        """Append one row to the audit trail.

        Parameters are stored redacted -- see :func:`utils.logger.redact` -- so
        an action that ever carries a secret cannot leak it into the database.
        """
        from utils.logger import redact

        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO action_history (created_at, session_id, action, parameters, risk,"
                " source, allowed, succeeded, message, duration_ms)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (_utc_now(), session_id, action,
                 json.dumps(redact(dict(parameters)), default=str),
                 risk, source, int(allowed), int(succeeded), message[:500], int(duration_ms)),
            )

    def recent_actions(self, limit: int = 20) -> List[ActionHistoryEntry]:
        rows = self._connect().execute(
            "SELECT created_at, action, risk, source, allowed, succeeded, message"
            " FROM action_history ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            ActionHistoryEntry(
                created_at=row["created_at"],
                action=row["action"],
                risk=row["risk"],
                source=row["source"],
                allowed=bool(row["allowed"]),
                succeeded=bool(row["succeeded"]),
                message=row["message"],
            )
            for row in rows
        ]

    # -------------------------------------------------------- permission audit

    def record_permission_decision(self, *, permission: str, action: str,
                                   decision: str, reason: str = "") -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO permission_audit (created_at, permission, action, decision, reason)"
                " VALUES (?, ?, ?, ?, ?)",
                (_utc_now(), permission, action, decision, reason[:300]),
            )

    def recent_permission_decisions(self, limit: int = 50) -> List[sqlite3.Row]:
        return list(self._connect().execute(
            "SELECT * FROM permission_audit ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall())

    # ------------------------------------------------------------------ stats

    def counts(self) -> dict[str, int]:
        """Row counts per table, used by the About screen and the tests."""
        tables: Sequence[str] = ("conversation", "action_history", "permission_audit", "kv_store")
        connection = self._connect()
        return {
            table: int(connection.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])
            for table in tables
        }
