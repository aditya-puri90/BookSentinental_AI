import sqlite3

def check_displacements():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT * FROM book_displacements ORDER BY id DESC LIMIT 5")
        rows = cursor.fetchall()
        
        if not rows:
            print("No displacements found yet.")
        else:
            print("Recent Displacements:")
            for row in rows:
                print(row)
    except sqlite3.OperationalError as e:
        print(f"Error querying database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_displacements()
