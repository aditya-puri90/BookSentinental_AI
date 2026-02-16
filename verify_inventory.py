import sqlite3
import os
import time
from utils import init_db, get_db_connection

# Mocking BookStateManager dependencies
class MockBookStateManager:
    def __init__(self):
        self.conn = get_db_connection('test_inventory.db')
        self.cursor = self.conn.cursor()

    def update_inventory(self, book_name, change):
        if not book_name: return
        
        try:
            # Check current quantity
            self.cursor.execute("SELECT quantity FROM book_inventory WHERE book_name=?", (book_name,))
            row = self.cursor.fetchone()
            
            current_qty = 0
            if row:
                current_qty = row[0]
            
            new_qty = max(0, current_qty + change)
            
            if row:
                self.cursor.execute("UPDATE book_inventory SET quantity=?, last_updated=CURRENT_TIMESTAMP WHERE book_name=?", (new_qty, book_name))
            else:
                self.cursor.execute("INSERT INTO book_inventory (book_name, quantity) VALUES (?, ?)", (book_name, new_qty))
                
            self.conn.commit()
            print(f"[INVENTORY] Updated '{book_name}' Quantity: {current_qty} -> {new_qty}")
        except Exception as e:
            print(f"DB Error (Inventory): {e}")

    def close(self):
        if hasattr(self, 'conn'):
            self.conn.close()

def verify_inventory():
    DB_FILE = 'test_inventory.db'
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    
    init_db(DB_FILE)
    
    manager = MockBookStateManager()
    
    print("--- Test 1: New Detection (Harry Potter) ---")
    manager.update_inventory("Harry Potter", 1)
    
    conn = get_db_connection(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT quantity FROM book_inventory WHERE book_name='Harry Potter'")
    qty = cursor.fetchone()[0]
    assert qty == 1, f"Expected 1, got {qty}"
    print("PASS")

    print("--- Test 2: Second Detection (Harry Potter) ---")
    manager.update_inventory("Harry Potter", 1)
    cursor.execute("SELECT quantity FROM book_inventory WHERE book_name='Harry Potter'")
    qty = cursor.fetchone()[0]
    assert qty == 2, f"Expected 2, got {qty}"
    print("PASS")

    print("--- Test 3: Detection of Another Book (The Hobbit) ---")
    manager.update_inventory("The Hobbit", 1)
    cursor.execute("SELECT quantity FROM book_inventory WHERE book_name='The Hobbit'")
    qty = cursor.fetchone()[0]
    assert qty == 1, f"Expected 1, got {qty}"
    print("PASS")

    print("--- Test 4: Removal of One Copy (Harry Potter) ---")
    manager.update_inventory("Harry Potter", -1)
    cursor.execute("SELECT quantity FROM book_inventory WHERE book_name='Harry Potter'")
    qty = cursor.fetchone()[0]
    assert qty == 1, f"Expected 1, got {qty}"
    print("PASS")

    print("--- Test 5: Removal to Zero (The Hobbit) ---")
    manager.update_inventory("The Hobbit", -1)
    cursor.execute("SELECT quantity FROM book_inventory WHERE book_name='The Hobbit'")
    qty = cursor.fetchone()[0]
    assert qty == 0, f"Expected 0, got {qty}"
    print("PASS")

    print("--- Test 6: Removal below Zero (Safety Check) ---")
    manager.update_inventory("The Hobbit", -1)
    cursor.execute("SELECT quantity FROM book_inventory WHERE book_name='The Hobbit'")
    qty = cursor.fetchone()[0]
    assert qty == 0, f"Expected 0, got {qty}"
    print("PASS")
    
    manager.close()
    conn.close()
    if os.path.exists(DB_FILE):
        try:
            os.remove(DB_FILE)
        except:
            pass
    print("\nAll Inventory Tests Passed!")

if __name__ == "__main__":
    verify_inventory()
