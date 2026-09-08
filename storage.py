"""MongoDB storage for interview sessions, profiles, and generated documents.

The database is where the kiosk's records live, but it is deliberately not something the
kiosk depends on to function: an interview that cannot be written to Mongo still finishes,
still prints, and is still saved to disk under data/sessions. Losing the database costs
you the query layer, not the person standing at the machine.

Two properties make that safe:

* Every operation is wrapped and returns a value instead of raising.
* A failure opens a circuit breaker for RETRY_AFTER seconds. Without it, each write would
  wait out the driver's server-selection timeout, and a stopped mongod would add that
  delay to every single turn of every interview.

Collections
    sessions    one document per interview: language, phase, transcript, profile, delivery
    documents   one per generated application from the document assistant
"""
import os
import time
from typing import Any

MONGO_URL = os.getenv("MONGO_URL", "mongodb://127.0.0.1:27017")
MONGO_DB = os.getenv("MONGO_DB", "tepma")
# Short on purpose: this runs inside a request, and the fallback is always available.
CONNECT_TIMEOUT_MS = int(os.getenv("MONGO_TIMEOUT_MS", "800"))
RETRY_AFTER = 30.0

_client: Any = None
_unavailable_until = 0.0
_warned = False


def _database(force: bool = False):
    """The database handle, or None while Mongo is unreachable.

    force skips the circuit breaker. Writes never do - the whole point is that they do not
    pay the connection timeout - but a human asking "is the database up?" wants the answer
    now, not up to RETRY_AFTER seconds after it came back.
    """
    global _client, _unavailable_until, _warned

    if not force and time.monotonic() < _unavailable_until:
        return None
    try:
        if _client is None:
            from pymongo import MongoClient

            _client = MongoClient(
                MONGO_URL,
                serverSelectionTimeoutMS=CONNECT_TIMEOUT_MS,
                connectTimeoutMS=CONNECT_TIMEOUT_MS,
                socketTimeoutMS=CONNECT_TIMEOUT_MS * 5,
                appname="tepma",
            )
        _client.admin.command("ping")
        _warned = False
        return _client[MONGO_DB]
    except Exception as exc:
        _unavailable_until = time.monotonic() + RETRY_AFTER
        _client = None
        if not _warned:
            print(f"storage: MongoDB unavailable ({type(exc).__name__}); "
                  f"records stay on disk only. Retrying in {RETRY_AFTER:.0f}s.")
            _warned = True
        return None


def available() -> bool:
    return _database() is not None


def status() -> dict:
    """For the status endpoint: is the database connected, and what is in it?"""
    database = _database(force=True)
    if database is None:
        return {"connected": False, "url": MONGO_URL, "database": MONGO_DB}
    try:
        return {
            "connected": True,
            "url": MONGO_URL,
            "database": MONGO_DB,
            "sessions": database.sessions.estimated_document_count(),
            "documents": database.documents.estimated_document_count(),
        }
    except Exception:
        return {"connected": False, "url": MONGO_URL, "database": MONGO_DB}


def save_session(session_id: str, **fields) -> bool:
    """Create or update one interview record. Returns whether it was written."""
    database = _database()
    if database is None or not session_id:
        return False
    try:
        database.sessions.update_one(
            {"_id": session_id},
            {"$set": {**fields, "updated_at": time.time()},
             "$setOnInsert": {"created_at": time.time()}},
            upsert=True,
        )
        return True
    except Exception as exc:
        print(f"storage: could not save session {session_id}: {exc}")
        return False


def save_document(session_id: str, **fields) -> bool:
    """Create or update one generated-document record."""
    database = _database()
    if database is None or not session_id:
        return False
    try:
        database.documents.update_one(
            {"_id": session_id},
            {"$set": {**fields, "updated_at": time.time()},
             "$setOnInsert": {"created_at": time.time()}},
            upsert=True,
        )
        return True
    except Exception as exc:
        print(f"storage: could not save document {session_id}: {exc}")
        return False


def get_session(session_id: str) -> dict | None:
    database = _database()
    if database is None:
        return None
    try:
        return database.sessions.find_one({"_id": session_id})
    except Exception:
        return None


def recent_sessions(limit: int = 20) -> list[dict]:
    """Newest interviews first - what a dashboard or an operator would ask for."""
    database = _database()
    if database is None:
        return []
    try:
        return list(database.sessions.find(
            {}, {"transcript": 0}).sort("created_at", -1).limit(limit))
    except Exception:
        return []


def ensure_indexes() -> bool:
    """Called once at startup. Indexes the fields anything would actually query on."""
    database = _database()
    if database is None:
        return False
    try:
        database.sessions.create_index("created_at")
        database.sessions.create_index("language")
        database.sessions.create_index("profile.email")
        database.documents.create_index("created_at")
        return True
    except Exception as exc:
        print(f"storage: could not create indexes: {exc}")
        return False
