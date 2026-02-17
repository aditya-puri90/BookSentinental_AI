import sqlite3
import os

def clear_detected_books():
    db_path = 'database.db'
    if not os.path.exists(db_path):
        print("Database not found.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Clear detected_books table
        cursor.execute("DELETE FROM detected_books")
        # Also clear inventory since it depends on names
        cursor.execute("DELETE FROM book_inventory")
        
        conn.commit()
        print("Successfully cleared 'detected_books' and 'book_inventory'.")
        print("You can now run the monitor to re-scan books with improved OCR.")
    except Exception as e:
        print(f"Error clearing database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    choice = input("This will delete all identified books and inventory. Continue? (y/n): ")
    if choice.lower() == 'y':
        clear_detected_books()
    else:
        print("Operation cancelled.")
