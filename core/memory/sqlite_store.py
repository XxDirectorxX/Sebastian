# core/memory/sqlite_store.py

import sqlite3

class SQLiteMemory:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS memory (key TEXT PRIMARY KEY, value TEXT)")
        self.conn.commit()

    def save_fact(self, key, value):
        self.conn.execute("REPLACE INTO memory (key, value) VALUES (?, ?)", (key, value))
        self.conn.commit()

    def load_fact(self, key):
        cursor = self.conn.execute("SELECT value FROM memory WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row[0] if row else None
