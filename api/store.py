"""SQLite persistence for job offers and uploaded candidates."""
import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS offers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    category TEXT,
    text TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    source TEXT,
    seconds REAL,
    resume TEXT,
    lines TEXT,
    prediction TEXT,
    created_at REAL NOT NULL
);
"""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.files = path.parent / "documents"
        self.files.mkdir(exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    def _all(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    def _write(self, sql, args=()):
        with self.lock:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return cur.lastrowid

    def seed_offers(self, offers: list[dict]):
        if not self._all("SELECT id FROM offers LIMIT 1"):
            for o in offers:
                title = o["text"].split(".")[0]
                self.add_offer(title, o["text"], o["category"])

    def add_offer(self, title: str, text: str, category: str | None = None) -> int:
        return self._write("INSERT INTO offers (title, category, text, created_at) VALUES (?, ?, ?, ?)",
                           (title, category, text, time.time()))

    def offers(self):
        return self._all("SELECT * FROM offers ORDER BY id")

    def offer(self, offer_id: int):
        rows = self._all("SELECT * FROM offers WHERE id = ?", (offer_id,))
        return rows[0] if rows else None

    def delete_offer(self, offer_id: int):
        self._write("DELETE FROM offers WHERE id = ?", (offer_id,))

    def new_candidate(self, filename: str) -> int:
        return self._write("INSERT INTO candidates (filename, status, created_at) VALUES (?, 'processing', ?)",
                           (filename, time.time()))

    def finish_candidate(self, cid: int, parsed: dict, prediction: dict, image: bytes):
        (self.files / f"{cid}.png").write_bytes(image)
        self._write("UPDATE candidates SET status='ready', source=?, seconds=?, resume=?, lines=?, prediction=? "
                    "WHERE id=?", (parsed["source"], parsed["seconds"], parsed["resume"].model_dump_json(),
                                   json.dumps(parsed["lines"]), json.dumps(prediction), cid))

    def fail_candidate(self, cid: int, error: str):
        self._write("UPDATE candidates SET status='error', error=? WHERE id=?", (error, cid))

    def update_resume(self, cid: int, resume_json: str, prediction: dict):
        self._write("UPDATE candidates SET resume=?, prediction=? WHERE id=?",
                    (resume_json, json.dumps(prediction), cid))

    def candidates(self, with_lines: bool = False):
        rows = self._all("SELECT * FROM candidates ORDER BY id DESC")
        return [self._decode(r, with_lines) for r in rows]

    def candidate(self, cid: int):
        rows = self._all("SELECT * FROM candidates WHERE id = ?", (cid,))
        return self._decode(rows[0], True) if rows else None

    def delete_candidate(self, cid: int):
        self._write("DELETE FROM candidates WHERE id = ?", (cid,))
        (self.files / f"{cid}.png").unlink(missing_ok=True)

    @staticmethod
    def _decode(row: dict, with_lines: bool):
        for key in ("resume", "prediction") + (("lines",) if with_lines else ()):
            row[key] = json.loads(row[key]) if row.get(key) else None
        if not with_lines:
            row.pop("lines", None)
        return row
