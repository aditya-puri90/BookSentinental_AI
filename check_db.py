import sqlite3
import pandas as pd
from utils import DB_FILE

def check_db():
    try:
        conn = sqlite3.connect(DB_FILE)
        query = "SELECT * FROM detected_books"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            print("No books detected yet.")
        else:
            print("Detected Books in Database:")
            print(df)
            
    except Exception as e:
        print(f"Error reading database: {e}")
        print("Make sure 'database.db' exists and 'detected_books' table is created.")

if __name__ == "__main__":
    check_db()
