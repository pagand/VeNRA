import os
import sqlite3
from fastapi.testclient import TestClient
from venra.main import app

client = TestClient(app)

def test_db_initialization():
    # Trigger startup event
    with TestClient(app) as tc:
        assert os.path.exists("venra.db")
        
        # Verify tables exist
        conn = sqlite3.connect("venra.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        assert "trace" in tables
        assert "chatsession" in tables
        conn.close()
