import concurrent.futures
import hashlib
import logging
import os
from pathlib import Path
import pickle
import sqlite3
from unittest.mock import patch

import pytest

from steerscope.models.language_model_cache import (
    LanguageModelCacheError,
    SQLiteLanguageModelCache,
)


def test_sqlite_cache_round_trip_overwrite_and_hash_only_storage(tmp_path):
    path = tmp_path / "judge.sqlite3"
    cache = SQLiteLanguageModelCache(path)
    cache.upsert_many({"large prompt key": "first", "other": "second"})
    assert cache.get_many(["missing", "other", "large prompt key"]) == {
        "other": "second",
        "large prompt key": "first",
    }

    cache.upsert_many({"large prompt key": "refreshed"})
    cache.flush()
    cache.close()
    cache.close()

    restored = SQLiteLanguageModelCache(path)
    assert restored.get_many(["large prompt key"]) == {
        "large prompt key": "refreshed"
    }
    row = restored._connection.execute(
        "SELECT key_hash, completion FROM cache_entries "
        "WHERE completion = 'refreshed'"
    ).fetchone()
    assert bytes(row[0]) == hashlib.sha256(b"large prompt key").digest()
    assert len(row[0]) == 32
    restored.close()


def test_sqlite_cache_configures_wal_normal_and_busy_timeout(tmp_path):
    cache = SQLiteLanguageModelCache(
        tmp_path / "judge.sqlite3", busy_timeout_ms=12_345
    )

    assert cache._connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert cache._connection.execute("PRAGMA synchronous").fetchone()[0] == 1
    assert cache._connection.execute("PRAGMA busy_timeout").fetchone()[0] == 12_345
    cache.close()


def test_get_many_chunks_large_queries_and_deduplicates_input(tmp_path):
    cache = SQLiteLanguageModelCache(tmp_path / "judge.sqlite3")
    entries = {f"key-{index}": f"value-{index}" for index in range(1_205)}
    cache.upsert_many(entries)

    requested = list(entries) + ["key-0", "absent"]
    assert cache.get_many(requested) == entries
    cache.close()


def test_cache_rejects_invalid_key_and_completion_types(tmp_path):
    cache = SQLiteLanguageModelCache(tmp_path / "judge.sqlite3")

    with pytest.raises(TypeError, match="keys must be strings"):
        cache.get_many([1])
    with pytest.raises(TypeError, match="completions must be strings"):
        cache.upsert_many({"prompt": 1})
    cache.close()


def test_cache_is_thread_safe_with_one_shared_connection(tmp_path):
    cache = SQLiteLanguageModelCache(tmp_path / "judge.sqlite3")

    def write_partition(partition):
        cache.upsert_many({
            f"key-{partition}-{index}": f"value-{partition}-{index}"
            for index in range(100)
        })

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write_partition, range(8)))

    expected_keys = [
        f"key-{partition}-{index}"
        for partition in range(8)
        for index in range(100)
    ]
    assert len(cache.get_many(expected_keys)) == 800
    cache.close()


def test_independent_connections_initialize_and_write_without_lost_updates(
    tmp_path,
):
    path = tmp_path / "judge.sqlite3"

    def open_and_write(partition):
        cache = SQLiteLanguageModelCache(path)
        try:
            cache.upsert_many({
                f"key-{partition}-{index}": f"value-{partition}-{index}"
                for index in range(100)
            })
        finally:
            cache.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(open_and_write, range(4)))

    restored = SQLiteLanguageModelCache(path)
    expected_keys = [
        f"key-{partition}-{index}"
        for partition in range(4)
        for index in range(100)
    ]
    assert len(restored.get_many(expected_keys)) == 400
    restored.close()


def test_legacy_pickle_migration_is_idempotent_and_never_overwrites_sqlite(
    tmp_path,
):
    legacy = tmp_path / "judge_cache.pkl"
    path = tmp_path / "judge.sqlite3"
    with legacy.open("wb") as file:
        pickle.dump({"old-key": "legacy-value"}, file)

    cache = SQLiteLanguageModelCache(path, legacy_pickle_path=legacy)
    assert cache.get_many(["old-key"]) == {"old-key": "legacy-value"}
    cache.upsert_many({"old-key": "refreshed-value"})
    cache.close()
    assert legacy.is_file()

    # A changed legacy file is imported again, but INSERT OR IGNORE preserves
    # a newer SQLite value while adding keys that were not imported before.
    with legacy.open("wb") as file:
        pickle.dump({
            "old-key": "stale-legacy-value",
            "new-key": "new-legacy-value",
        }, file)
    stat = legacy.stat()
    os.utime(legacy, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))

    migrated = SQLiteLanguageModelCache(path, legacy_pickle_path=legacy)
    assert migrated.get_many(["old-key", "new-key"]) == {
        "old-key": "refreshed-value",
        "new-key": "new-legacy-value",
    }
    migrated.close()

    # An unchanged pickle is recognized before pickle.load is called.
    with patch(
        "steerscope.models.language_model_cache.pickle.load",
        side_effect=AssertionError("legacy pickle was loaded twice"),
    ):
        unchanged = SQLiteLanguageModelCache(path, legacy_pickle_path=legacy)
    unchanged.close()


@pytest.mark.parametrize(
    "payload, warning",
    [
        (b"not a pickle", "unreadable legacy"),
        (pickle.dumps({"prompt": 123}), "invalid legacy"),
    ],
)
def test_corrupt_or_invalid_legacy_pickle_is_warned_and_ignored(
    tmp_path, caplog, payload, warning
):
    legacy = tmp_path / "judge_cache.pkl"
    legacy.write_bytes(payload)

    with caplog.at_level(logging.WARNING):
        cache = SQLiteLanguageModelCache(
            tmp_path / "judge.sqlite3", legacy_pickle_path=legacy
        )

    assert warning in caplog.text
    assert cache.get_many(["prompt"]) == {}
    assert legacy.read_bytes() == payload
    cache.close()


def test_corrupt_sqlite_database_fails_closed_without_replacing_it(tmp_path):
    path = tmp_path / "judge.sqlite3"
    corrupt = b"this is not a sqlite database"
    path.write_bytes(corrupt)

    with pytest.raises(LanguageModelCacheError, match="corrupt SQLite"):
        SQLiteLanguageModelCache(path)

    assert path.read_bytes() == corrupt


def test_incompatible_tables_and_schema_version_fail_closed(tmp_path):
    incompatible_path = tmp_path / "other.sqlite3"
    connection = sqlite3.connect(incompatible_path)
    connection.execute("CREATE TABLE unrelated(value TEXT)")
    connection.commit()
    connection.close()

    with pytest.raises(LanguageModelCacheError, match="tables=.*unrelated"):
        SQLiteLanguageModelCache(incompatible_path)
    connection = sqlite3.connect(incompatible_path)
    assert connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall() == [("unrelated",)]
    connection.close()

    version_path = tmp_path / "version.sqlite3"
    cache = SQLiteLanguageModelCache(version_path)
    cache.close()
    connection = sqlite3.connect(version_path)
    connection.execute(
        "UPDATE cache_metadata SET value='999' WHERE name='schema_version'"
    )
    connection.commit()
    connection.close()

    with pytest.raises(LanguageModelCacheError, match="schema version"):
        SQLiteLanguageModelCache(version_path)


def test_operations_after_close_fail(tmp_path):
    cache = SQLiteLanguageModelCache(tmp_path / "judge.sqlite3")
    cache.close()

    with pytest.raises(RuntimeError, match="cache is closed"):
        cache.get_many(["prompt"])
    with pytest.raises(RuntimeError, match="cache is closed"):
        cache.flush()

