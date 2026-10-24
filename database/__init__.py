"""Local SQLite persistence.

Everything AARMA remembers between runs that is not a *setting* lives here:
the conversation transcript, the audit trail of executed actions and the
permission decisions behind them.  Settings themselves stay in
``config.json`` so a user can read and repair them without a database tool.
"""

from database.db import Database, SCHEMA_VERSION

__all__ = ["Database", "SCHEMA_VERSION"]
