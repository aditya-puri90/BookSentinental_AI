import json
import sqlite3
import os

CONFIG_FILE = 'slots.json'
DB_FILE = 'database.db'

def load_config(path=CONFIG_FILE):
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)

def save_config(data, path=CONFIG_FILE):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def get_db_connection(db_path=DB_FILE):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path=DB_FILE):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    # Create books table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS books (
            slot_id TEXT PRIMARY KEY,
            book_name TEXT,
            author TEXT,
            category TEXT,
            status TEXT DEFAULT 'Present'
        )
    ''')
    
    # Create events table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            slot_id TEXT,
            action TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (slot_id) REFERENCES books (slot_id)
        )
    ''')

    # Create detected_books table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detected_books (
            id INTEGER PRIMARY KEY,
            name TEXT,
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {db_path}")

if __name__ == "__main__":
    init_db()
