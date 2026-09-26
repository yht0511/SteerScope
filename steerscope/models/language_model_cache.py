"""Incremental SQLite storage for language-model completions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import fcntl
import hashlib
from itertools import islice
import logging
import os
from pathlib import Path
import pickle
import sqlite3
import threading


logger = logging.getLogger(__name__)


class LanguageModelCacheError(RuntimeError):
    """Raised when a persistent language-model cache cannot be used safely."""


class SQLiteLanguageModelCache:
    """Process-safe SQLite completion cache; WAL databases must reside on a local filesystem."""

    SCHEMA_VERSION = "1"
    HASH_ALGORITHM = "sha256"
    _QUERY_BATCH_SIZE = 500
    _MIGRATION_BATCH_SIZE = 1_000
    _REQUIRED_TABLES = frozenset({"cache_entries", "cache_metadata"})

    def __init__(
        self,
        path: str | Path,
        *,
        legacy_pickle_path: str | Path | None = None,
        busy_timeout_ms: int = 30_000,
    ) -> None:
        self.path = Path(path)
        self.legacy_pickle_path = (
            None if legacy_pickle_path is None else Path(legacy_pickle_path)
        )
        self.busy_timeout_ms = int(busy_timeout_ms)
        if self.busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms cannot be negative")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None
        try:
            self._connection = sqlite3.connect(
                self.path,
                timeout=self.busy_timeout_ms / 1_000,
                check_same_thread=False,
            )
            # Serialize WAL negotiation and first-time schema creation.
            initialization_lock = self.path.with_name(
                self.path.name + ".initialization.lock"
            )
            with initialization_lock.open("a+b") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    self._configure_connection()
                    self._validate_or_initialize_schema()
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            self._migrate_legacy_pickle()
        except Exception:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            raise

    @staticmethod
    def key_digest(cache_key: str) -> bytes:
        """Return the storage identity for an existing string cache key."""
        if not isinstance(cache_key, str):
            raise TypeError("Language-model cache keys must be strings")
        return hashlib.sha256(cache_key.encode("utf-8")).digest()

    def get_many(self, cache_keys: Iterable[str]) -> dict[str, str]:
        """Return cached completions keyed by the caller's original strings."""
        keys = list(dict.fromkeys(cache_keys))
        if not keys:
            return {}
        digest_to_key: dict[bytes, str] = {}
        for key in keys:
            digest = self.key_digest(key)
            previous = digest_to_key.setdefault(digest, key)
            if previous != key:
                raise LanguageModelCacheError(
                    "SHA-256 collision between language-model cache keys"
                )

        found: dict[str, str] = {}
        with self._lock:
            connection = self._require_open()
            digests = list(digest_to_key)
            for offset in range(0, len(digests), self._QUERY_BATCH_SIZE):
                batch = digests[offset:offset + self._QUERY_BATCH_SIZE]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    "SELECT key_hash, completion FROM cache_entries "
                    f"WHERE key_hash IN ({placeholders})",
                    batch,
                )
                for digest, completion in rows:
                    key = digest_to_key[bytes(digest)]
                    found[key] = str(completion)
        return found

    def upsert_many(self, entries: Mapping[str, str]) -> None:
        """Insert or replace a batch of completions in one transaction."""
        if not entries:
            return
        rows = []
        digest_to_key: dict[bytes, str] = {}
        for cache_key, completion in entries.items():
            digest = self.key_digest(cache_key)
            if not isinstance(completion, str):
                raise TypeError("Language-model cache completions must be strings")
            previous = digest_to_key.setdefault(digest, cache_key)
            if previous != cache_key:
                raise LanguageModelCacheError(
                    "SHA-256 collision between language-model cache keys"
                )
            rows.append((digest, completion))

        with self._lock:
            connection = self._require_open()
            with connection:
                connection.executemany(
                    "INSERT INTO cache_entries(key_hash, completion) VALUES (?, ?) "
                    "ON CONFLICT(key_hash) DO UPDATE SET "
                    "completion = excluded.completion",
                    rows,
                )

    def flush(self) -> None:
        """Commit any outstanding transaction without checkpointing the WAL."""
        with self._lock:
            self._require_open().commit()

    def close(self) -> None:
        """Commit and close the connection.  Closing twice is harmless."""
        with self._lock:
            if self._connection is None:
                return
            try:
                self._connection.commit()
            finally:
                self._connection.close()
                self._connection = None

    def _require_open(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("Language-model cache is closed")
        return self._connection

    def _configure_connection(self) -> None:
        connection = self._require_open()
        try:
            connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
            mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if str(mode).lower() != "wal":
                raise LanguageModelCacheError(
                    f"SQLite refused WAL mode for cache {self.path}: {mode!r}"
                )
            connection.execute("PRAGMA synchronous=NORMAL")
            # Read the schema without an O(database size) quick_check on every open.
            connection.execute("PRAGMA schema_version").fetchone()
        except sqlite3.DatabaseError as error:
            raise LanguageModelCacheError(
                f"Unreadable or corrupt SQLite language-model cache {self.path}: "
                f"{error}"
            ) from error

    def _validate_or_initialize_schema(self) -> None:
        connection = self._require_open()
        try:
            # Serialize first-time initialization across processes.  A plain
            # SELECT followed by CREATE has a race where two fresh processes
            # can both observe an empty database.
            connection.execute("BEGIN IMMEDIATE")
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            if not tables:
                connection.execute(
                    "CREATE TABLE cache_entries ("
                    "key_hash BLOB PRIMARY KEY NOT NULL, "
                    "completion TEXT NOT NULL"
                    ") WITHOUT ROWID"
                )
                connection.execute(
                    "CREATE TABLE cache_metadata ("
                    "name TEXT PRIMARY KEY NOT NULL, "
                    "value TEXT NOT NULL"
                    ") WITHOUT ROWID"
                )
                connection.executemany(
                    "INSERT INTO cache_metadata(name, value) VALUES (?, ?)",
                    (
                        ("schema_version", self.SCHEMA_VERSION),
                        ("key_hash_algorithm", self.HASH_ALGORITHM),
                    ),
                )
                connection.commit()
                return

            if tables != self._REQUIRED_TABLES:
                raise LanguageModelCacheError(
                    f"Incompatible SQLite language-model cache schema at "
                    f"{self.path}: tables={sorted(tables)}"
                )
            self._validate_table_columns(
                "cache_entries",
                (("key_hash", "BLOB", 1), ("completion", "TEXT", 0)),
            )
            self._validate_table_columns(
                "cache_metadata",
                (("name", "TEXT", 1), ("value", "TEXT", 0)),
            )
            metadata = dict(connection.execute(
                "SELECT name, value FROM cache_metadata "
                "WHERE name IN ('schema_version', 'key_hash_algorithm')"
            ))
            connection.commit()
        except sqlite3.DatabaseError as error:
            if connection.in_transaction:
                connection.rollback()
            raise LanguageModelCacheError(
                f"Unreadable or incompatible SQLite language-model cache "
                f"{self.path}: {error}"
            ) from error
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise

        if metadata.get("schema_version") != self.SCHEMA_VERSION:
            raise LanguageModelCacheError(
                f"Unsupported language-model cache schema version at "
                f"{self.path}: {metadata.get('schema_version')!r}"
            )
        if metadata.get("key_hash_algorithm") != self.HASH_ALGORITHM:
            raise LanguageModelCacheError(
                f"Unsupported language-model cache key hash at {self.path}: "
                f"{metadata.get('key_hash_algorithm')!r}"
            )

    def _validate_table_columns(self, table: str, expected) -> None:
        connection = self._require_open()
        columns = tuple(
            (str(row[1]), str(row[2]).upper(), int(row[5]))
            for row in connection.execute(f"PRAGMA table_info({table})")
        )
        if columns != tuple(expected):
            raise LanguageModelCacheError(
                f"Incompatible columns for {table} in cache {self.path}: "
                f"{columns!r}"
            )

    def _migrate_legacy_pickle(self) -> None:
        legacy_path = self.legacy_pickle_path
        if legacy_path is None or not legacy_path.is_file():
            return
        lock_path = self.path.with_name(self.path.name + ".migration.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                self._migrate_legacy_pickle_locked(legacy_path)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _migrate_legacy_pickle_locked(self, legacy_path: Path) -> None:
        try:
            with legacy_path.open("rb") as legacy_file:
                stat = os.fstat(legacy_file.fileno())
                fingerprint = {
                    "legacy_pickle_path": str(legacy_path.resolve()),
                    "legacy_pickle_size": str(stat.st_size),
                    "legacy_pickle_mtime_ns": str(stat.st_mtime_ns),
                    "legacy_import_complete": "1",
                }
                if self._metadata_matches(fingerprint):
                    return
                cached = pickle.load(legacy_file)
        except (
            OSError,
            EOFError,
            pickle.PickleError,
            AttributeError,
            ImportError,
            IndexError,
            TypeError,
            ValueError,
        ) as error:
            logger.warning(
                "Ignoring unreadable legacy language-model cache %s: %s",
                legacy_path,
                error,
            )
            return

        if not isinstance(cached, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in cached.items()
        ):
            logger.warning(
                "Ignoring invalid legacy language-model cache %s: expected "
                "dict[str, str]",
                legacy_path,
            )
            return

        connection = self._require_open()
        try:
            items = iter(cached.items())
            while True:
                batch = list(islice(items, self._MIGRATION_BATCH_SIZE))
                if not batch:
                    break
                rows = [
                    (self.key_digest(cache_key), completion)
                    for cache_key, completion in batch
                ]
                with connection:
                    connection.executemany(
                        "INSERT OR IGNORE INTO "
                        "cache_entries(key_hash, completion) VALUES (?, ?)",
                        rows,
                    )
            with connection:
                connection.executemany(
                    "INSERT INTO cache_metadata(name, value) VALUES (?, ?) "
                    "ON CONFLICT(name) DO UPDATE SET value = excluded.value",
                    fingerprint.items(),
                )
        except sqlite3.DatabaseError as error:
            raise LanguageModelCacheError(
                f"Failed to import legacy language-model cache {legacy_path} "
                f"into {self.path}: {error}"
            ) from error

    def _metadata_matches(self, expected: Mapping[str, str]) -> bool:
        connection = self._require_open()
        placeholders = ",".join("?" for _ in expected)
        actual = dict(connection.execute(
            "SELECT name, value FROM cache_metadata "
            f"WHERE name IN ({placeholders})",
            tuple(expected),
        ))
        return actual == dict(expected)

    def __enter__(self) -> "SQLiteLanguageModelCache":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
