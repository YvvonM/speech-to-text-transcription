import sqlite3
import json
from datetime import datetime 

class ResearchDatabase:
    def __init__(self, db_path = "Research.db"):
        self.db_path = db_path
        self._setup()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _setup(self):
        with self._connect() as conn:
            conn.execute(""" CREATE TABLE IF NOT EXISTS research_data(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            search_queries TEXT NOT NULL,
            structure TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)

    def insert_research_entry(self, query:str, search_queries: dict, structure: dict):
        with self._connect()as conn:
            conn.execute('''
            INSERT INTO research_data(query, search_queries, structure)
            VALUES(?,?,?)''',
           (query, json.dumps(search_queries), json.dumps(structure)))

    def fetch_all_entries(self):
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM research_data"
            )
            return cursor.fetchall()

