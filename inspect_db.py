
import sqlite3

def inspect_db():
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        if not tables:
            print("No tables found in database.")
            return

        for table in tables:
            print(f"\n--- Table: {table} ---")
            try:
                # Get column names
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [info[1] for info in cursor.fetchall()]
                print(f"Columns: {columns}")
                
                # Get data
                cursor.execute(f"SELECT * FROM {table}")
                rows = cursor.fetchall()
                
                if not rows:
                    print("No records found.")
                else:
                    for row in rows:
                        print(row)
            except Exception as e:
                print(f"Error querying {table}: {e}")
                
        conn.close()
    except Exception as e:
        print(f"Database error: {e}")

if __name__ == "__main__":
    inspect_db()
