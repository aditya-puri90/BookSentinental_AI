
import sqlite3
import csv
import os

def export_to_csv():
    db_path = 'database.db'
    output_file = 'book_records.csv'
    
    if not os.path.exists(db_path):
        print(f"Error: Database file {db_path} not found.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Load book name mapping
    book_mapping = {}
    if os.path.exists('book_names.json'):
        import json
        try:
            with open('book_names.json', 'r') as f:
                book_mapping = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load book_names.json: {e}")

    # Query prioritizing detected_books and book_inventory since books table might be empty
    query = '''
        SELECT 
            db.name as book_name, 
            db.id as book_id, 
            COALESCE(bi.quantity, 0) as quantity, 
            CASE WHEN COALESCE(bi.quantity, 0) > 0 THEN 'Present' ELSE 'Absent' END as status, 
            COALESCE(bi.last_updated, db.last_seen) as timestamp
        FROM 
            detected_books db
        LEFT JOIN 
            book_inventory bi ON db.name = bi.book_name
    '''
    
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Process rows to apply mapping
        processed_rows = []
        for row in rows:
            book_name, book_id, quantity, status, timestamp = row
            
            # Apply mapping if available
            str_id = str(book_id)
            if str_id in book_mapping:
                book_name = book_mapping[str_id]
            
            processed_rows.append((book_name, book_id, quantity, status, timestamp))
            
        rows = processed_rows
        
        if not rows:
            print("No records found in detected_books either.")
        
        # Define the header
        header = ['Book Name', 'Book ID', 'Quantity', 'Status', 'Timestamp']
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
            
        msg = f"Successfully exported {len(rows)} records to {output_file}"
        print(msg)
        return msg
        
    except Exception as e:
        error_msg = f"Error exporting data: {e}"
        print(error_msg)
        return error_msg
    finally:
        conn.close()

if __name__ == "__main__":
    export_to_csv()
