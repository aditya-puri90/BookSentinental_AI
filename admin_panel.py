import tkinter as tk
from tkinter import ttk, messagebox
import sqlite3
from utils import load_config, get_db_connection

class AdminPanel:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Library Admin Panel")
        self.root.geometry("600x400")

        self.slots = load_config()
        self.slot_ids = list(self.slots.keys())
        
        if not self.slot_ids:
            messagebox.showwarning("Warning", "No slots found in slots.json. Please run annotate_slots.py first.")
            # self.root.destroy() # Don't destroy, let them verify config
            
        self.create_widgets()
        self.load_data()

    def create_widgets(self):
        # Slot Selection
        tk.Label(self.root, text="Select Slot ID:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.slot_combo = ttk.Combobox(self.root, values=self.slot_ids)
        self.slot_combo.grid(row=0, column=1, padx=10, pady=10)
        self.slot_combo.bind("<<ComboboxSelected>>", self.load_slot_details)

        # Book Metadata Fields
        tk.Label(self.root, text="Book Name:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.book_name_entry = tk.Entry(self.root, width=40)
        self.book_name_entry.grid(row=1, column=1, padx=10, pady=5)

        tk.Label(self.root, text="Author:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.author_entry = tk.Entry(self.root, width=40)
        self.author_entry.grid(row=2, column=1, padx=10, pady=5)

        tk.Label(self.root, text="Category:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.category_entry = tk.Entry(self.root, width=40)
        self.category_entry.grid(row=3, column=1, padx=10, pady=5)

        # Buttons
        self.save_btn = tk.Button(self.root, text="Save Mapping", command=self.save_mapping, bg="green", fg="white")
        self.save_btn.grid(row=4, column=0, columnspan=2, pady=20)
        
        self.refresh_btn = tk.Button(self.root, text="Refresh Slots", command=self.refresh_slots)
        self.refresh_btn.grid(row=5, column=0, columnspan=2)

    def refresh_slots(self):
        self.slots = load_config()
        self.slot_ids = list(self.slots.keys())
        self.slot_combo['values'] = self.slot_ids
        if not self.slot_ids:
             messagebox.showwarning("Warning", "No slots found. Run annotate_slots.py first.")

    def load_slot_details(self, event=None):
        slot_id = self.slot_combo.get()
        if not slot_id:
            return

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT book_name, author, category FROM books WHERE slot_id=?", (slot_id,))
        row = cursor.fetchone()
        conn.close()

        self.book_name_entry.delete(0, tk.END)
        self.author_entry.delete(0, tk.END)
        self.category_entry.delete(0, tk.END)

        if row:
            self.book_name_entry.insert(0, row['book_name'])
            # Check if author exists before inserting (handling potential None or missing col implementation)
            if row['author']: self.author_entry.insert(0, row['author'])
            if row['category']: self.category_entry.insert(0, row['category'])

    def save_mapping(self):
        slot_id = self.slot_combo.get()
        book_name = self.book_name_entry.get()
        author = self.author_entry.get()
        category = self.category_entry.get()

        if not slot_id or not book_name:
            messagebox.showerror("Error", "Slot ID and Book Name are required.")
            return

        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Upsert logic (INSERT OR REPLACE)
        cursor.execute('''
            INSERT OR REPLACE INTO books (slot_id, book_name, author, category, status)
            VALUES (?, ?, ?, ?, 'Present') 
        ''', (slot_id, book_name, author, category))
        
        conn.commit()
        conn.close()
        messagebox.showinfo("Success", f"Mapped {book_name} to {slot_id}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AdminPanel(root)
    root.mainloop()
