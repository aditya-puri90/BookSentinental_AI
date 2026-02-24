import cv2
import json
import time
import argparse
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from utils import load_config, get_db_connection, init_db
from ocr_utils import extract_text_from_image, KNOWN_BOOKS, _KNOWN_BOOKS_LOWER

# Configuration
CONFIG_FILE = 'slots.json'
TARGET_CLASS = 'book'
GRACE_PERIOD = 2.0  # Seconds before marking as Absent
IOU_THRESHOLD = 0.3 # Threshold for re-identifying a lost book
FRAME_SKIP = 5      # Run inference every N frames
MIN_CONFIRMATION_FRAMES = 10 # Frames to persist before adding to DB
MAX_REOCR_ATTEMPTS = 3  # Max re-OCR attempts for low-quality names
REOCR_FRAME_INTERVAL = 15  # Frames between re-OCR attempts

def _is_good_ocr_name(name):
    """Check if an OCR name is good quality (matches known book or is long enough)."""
    if not name or name == "Unknown" or len(name.strip()) < 3:
        return False
    # Check if it matches a known book
    if name.lower() in _KNOWN_BOOKS_LOWER:
        return True
    # If not in known books, at least require a reasonable length
    alpha_chars = sum(c.isalpha() for c in name)
    return alpha_chars >= 5

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    Box format: (x1, y1, x2, y2)
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    inter_width = max(0, xi_max - xi_min)
    inter_height = max(0, yi_max - yi_min)
    inter_area = inter_width * inter_height

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0
    return inter_area / union_area

class IDTracker:
    def __init__(self, start_id=1):
        # Maps current YOLO track_id to our persistent_id
        # { yolo_track_id: persistent_id }
        self.active_tracks = {}
        
        # Stores the last known state of a persistent_id
        # { persistent_id: {'box': (x1, y1, x2, y2), 'last_seen': time} }
        self.persistent_states = {}
        
        self.next_persistent_id = start_id

    def update(self, detections):
        """
        detections: list of (x1, y1, x2, y2, conf, cls, yolo_track_id)
        Returns: list of (x1, y1, x2, y2, conf, cls, persistent_id)
        """
        now = time.time()
        current_yolo_ids = set()
        resolved_detections = []
        
        # 1. Process active tracks or new associations
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            box = (x1, y1, x2, y2)
            yolo_id = det[6]
            
            if yolo_id is None:
                continue
                
            current_yolo_ids.add(yolo_id)
            
            if yolo_id in self.active_tracks:
                p_id = self.active_tracks[yolo_id]
                self.persistent_states[p_id] = {'box': box, 'last_seen': now}
                resolved_detections.append(det[:6] + (p_id,))
            else:
                best_iou = 0
                match_p_id = None
                
                active_p_ids = set(self.active_tracks.values())
                
                for p_id, state in self.persistent_states.items():
                    if p_id in active_p_ids:
                        continue 
                        
                    iou = calculate_iou(box, state['box'])
                    if iou > best_iou and iou > IOU_THRESHOLD:
                        best_iou = iou
                        match_p_id = p_id
                        
                if match_p_id is not None:
                    self.active_tracks[yolo_id] = match_p_id
                    self.persistent_states[match_p_id] = {'box': box, 'last_seen': now}
                    resolved_detections.append(det[:6] + (match_p_id,))
                    print(f"[TRACK] Remapped YOLO ID {yolo_id} -> Persistent ID {match_p_id} (IoU: {best_iou:.2f})")
                else:
                    p_id = self.next_persistent_id
                    self.next_persistent_id += 1
                    self.active_tracks[yolo_id] = p_id
                    self.persistent_states[p_id] = {'box': box, 'last_seen': now}
                    resolved_detections.append(det[:6] + (p_id,))
                    print(f"[TRACK] New Persistent ID {p_id} (from YOLO ID {yolo_id})")

        # 2. Cleanup active matches
        for yolo_id in list(self.active_tracks.keys()):
            if yolo_id not in current_yolo_ids:
                del self.active_tracks[yolo_id]

        return resolved_detections

class BookStateManager:
    def __init__(self):
        # { persistent_id: { 'status': 'Present'/'Absent', 'last_seen': float, 'timestamp': str, 'ocr_name': str } }
        self.states = {}
        # { persistent_id: count }
        self.pending_books = {}
        # { persistent_id: { 'attempts': int, 'last_attempt_frame': int } }
        self.reocr_tracker = {}
        init_db() # Ensure DB exists
        self.conn = get_db_connection()
        self.cursor = self.conn.cursor()
        self.frame_count = 0

    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()

    def update(self, resolved_detections, frame):
        now = time.time()
        self.frame_count += 1
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        current_persistent_ids = {det[6] for det in resolved_detections}
        
        # Create a map for quick access to boxes
        p_id_to_box = {det[6]: det[:4] for det in resolved_detections}

        for p_id in current_persistent_ids:
            if p_id not in self.states:
                # Check pending status
                if p_id not in self.pending_books:
                    self.pending_books[p_id] = 0
                
                self.pending_books[p_id] += 1
                
                # Only add if confirmed
                if self.pending_books[p_id] >= MIN_CONFIRMATION_FRAMES:
                    # New Book Confirmed
                    x1, y1, x2, y2 = p_id_to_box[p_id]
                    
                    # Ensure coordinates are within frame
                    h, w, _ = frame.shape
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(w, int(x2))
                    y2 = min(h, int(y2))
                    
                    # Run OCR
                    book_img = frame[y1:y2, x1:x2]
                    ocr_text = "Unknown"
                    if book_img.size > 0:
                        print(f"Running OCR on Book {p_id}...")
                        ocr_text = extract_text_from_image(book_img)
                        print(f"OCR Result for Book {p_id}: {ocr_text}")
                    
                    self.states[p_id] = {
                        'status': 'Present',
                        'last_seen': now,
                        'timestamp': current_time_str,
                        'ocr_name': ocr_text
                    }
                    
                    # Save to DB
                    try:
                        self.cursor.execute('''
                            INSERT INTO detected_books (id, name, first_seen, last_seen)
                            VALUES (?, ?, ?, ?)
                        ''', (int(p_id), ocr_text, current_time_str, current_time_str))
                        self.conn.commit()
                    except Exception as e:
                        print(f"DB Error: {e}")

                    print(f"[NEW] Book {p_id} ({ocr_text}) detected at {current_time_str}")
                    
                    # Update Inventory (Increment)
                    self.update_inventory(ocr_text, 1)
                    
                    # Remove from pending
                    del self.pending_books[p_id]
                    
                    # Schedule re-OCR if name is low quality
                    if not _is_good_ocr_name(ocr_text):
                        self.reocr_tracker[p_id] = {
                            'attempts': 0,
                            'last_attempt_frame': self.frame_count
                        }
            else:
                self.states[p_id]['last_seen'] = now
                
                if self.states[p_id]['status'] == 'Absent':
                    self.states[p_id]['status'] = 'Present'
                    self.states[p_id]['timestamp'] = current_time_str
                    print(f"[RETURN] Book {p_id} marked PRESENT at {current_time_str}")
                    
                    # Update Inventory (Increment)
                    self.update_inventory(self.states[p_id].get('ocr_name'), 1)
                
                # Re-OCR logic: retry on present books with low-quality names
                if p_id in self.reocr_tracker and p_id in p_id_to_box:
                    tracker = self.reocr_tracker[p_id]
                    frames_since = self.frame_count - tracker['last_attempt_frame']
                    if tracker['attempts'] < MAX_REOCR_ATTEMPTS and frames_since >= REOCR_FRAME_INTERVAL:
                        x1, y1, x2, y2 = p_id_to_box[p_id]
                        fh, fw, _ = frame.shape
                        x1c = max(0, int(x1))
                        y1c = max(0, int(y1))
                        x2c = min(fw, int(x2))
                        y2c = min(fh, int(y2))
                        book_img = frame[y1c:y2c, x1c:x2c]
                        if book_img.size > 0:
                            tracker['attempts'] += 1
                            tracker['last_attempt_frame'] = self.frame_count
                            print(f"[RE-OCR] Attempt {tracker['attempts']}/{MAX_REOCR_ATTEMPTS} for Book {p_id}")
                            new_text = extract_text_from_image(book_img)
                            if new_text and _is_good_ocr_name(new_text):
                                old_name = self.states[p_id].get('ocr_name', '')
                                self.states[p_id]['ocr_name'] = new_text
                                print(f"[RE-OCR] Improved Book {p_id}: '{old_name}' -> '{new_text}'")
                                # Update DB
                                try:
                                    self.cursor.execute('UPDATE detected_books SET name=? WHERE id=?',
                                                        (new_text, int(p_id)))
                                    self.conn.commit()
                                except Exception as e:
                                    print(f"DB Error (Re-OCR update): {e}")
                                # Update inventory
                                if old_name and old_name != new_text:
                                    self.update_inventory(old_name, -1)
                                    self.update_inventory(new_text, 1)
                                # Stop re-OCR for this book
                                del self.reocr_tracker[p_id]
                            elif tracker['attempts'] >= MAX_REOCR_ATTEMPTS:
                                print(f"[RE-OCR] Max attempts reached for Book {p_id}, keeping '{self.states[p_id].get('ocr_name', '')}'")
                                del self.reocr_tracker[p_id]

        for p_id, state in self.states.items():
            if p_id not in current_persistent_ids:
                if state['status'] == 'Present':
                    if now - state['last_seen'] > GRACE_PERIOD:
                        state['status'] = 'Absent'
                        state['timestamp'] = current_time_str
                        
                        # Log displacement
                        book_name = state.get('ocr_name', 'Unknown')
                        try:
                            self.cursor.execute('''
                                INSERT INTO book_displacements (book_id, book_name, displacement_time)
                                VALUES (?, ?, ?)
                            ''', (p_id, book_name, current_time_str))
                            self.conn.commit()
                        except Exception as e:
                            print(f"DB Error (Displacement): {e}")

                        # Update Inventory (Decrement)
                        self.update_inventory(book_name, -1)

                        print(f"[GONE] Book {p_id} marked ABSENT at {current_time_str}")

        # Cleanup pending books that are no longer detected
        for p_id in list(self.pending_books.keys()):
            if p_id not in current_persistent_ids:
                # Decrease confidence or remove immediately? 
                # Removing immediately is safer to avoid carrying over stale counts
                del self.pending_books[p_id]

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

    def get_state(self, p_id):
        return self.states.get(p_id, {})

def get_books_in_slot(detections, slot_coords):
    sx, sy, sw, sh = slot_coords
    sx2, sy2 = sx + sw, sy + sh
    
    books_in_slot = []
    for det in detections:
        dx1, dy1, dx2, dy2 = det[:4]
        cx = (dx1 + dx2) / 2
        cy = (dy1 + dy2) / 2
        
        if sx <= cx <= sx2 and sy <= cy <= sy2:
            p_id = det[6]
            if p_id is not None:
                books_in_slot.append(p_id)
            
    return books_in_slot

class SmartLibraryMonitor:
    def __init__(self, model_path='yolov8n.pt', config_file='slots.json'):
        self.config_file = config_file
        self.slots = load_config(config_file)
        if not self.slots:
            print(f"Warning: No slots found in {config_file}. Run annotate_slots.py first.")
            
        print("Loading YOLOv8 model...")
        self.model = YOLO(model_path)
        print("Model loaded.")
        
        # Initialize DB connection to find the starting ID
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(id) FROM detected_books")
        max_id = cursor.fetchone()[0]
        conn.close()

        start_id = (max_id + 1) if max_id is not None else 1
        print(f"Starting ID tracking from ID: {start_id}")

        self.id_tracker = IDTracker(start_id=start_id)
        self.state_manager = BookStateManager()
        self.frame_count = 0
        self.resolved_detections = []

    def process_frame(self, frame):
        self.frame_count += 1
        
        # Only run inference every FRAME_SKIP frames
        if self.frame_count % FRAME_SKIP == 0:
            # YOLO Tracking
            results = self.model.track(frame, persist=True, verbose=False)
            
            raw_detections = []
            for r in results:
                boxes = r.boxes
                if boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy().astype(int)
                else:
                    track_ids = [None] * len(boxes)
                    
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    if cls_name == TARGET_CLASS:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        yolo_id = track_ids[i]
                        if yolo_id is not None:
                             raw_detections.append((x1, y1, x2, y2, conf, cls_id, yolo_id))

            # Remap IDs
            self.resolved_detections = self.id_tracker.update(raw_detections)
            
            # Update Status
            self.state_manager.update(self.resolved_detections, frame)
        
        # Visualization
        debug_frame = frame.copy()
        
        for (x1, y1, x2, y2, conf, _, p_id) in self.resolved_detections:
             state = self.state_manager.get_state(p_id)
             status = state.get('status', 'Unknown')
             
             color = (255, 0, 0)
             cv2.rectangle(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
             
             label = f"ID:{p_id}"
             status_label = f"({status})"
             
             ocr_name = state.get('ocr_name', '')
             if ocr_name:
                 label += f" | {ocr_name}"
                 
             cv2.putText(debug_frame, label, (int(x1), int(y1)-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
             cv2.putText(debug_frame, status_label, (int(x1), int(y1)-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        for slot_id, (x, y, w, h) in self.slots.items():
            if w <= 0 or h <= 0: continue
            
            books_in_slot_ids = get_books_in_slot(self.resolved_detections, (x, y, w, h))
            count = len(books_in_slot_ids)
            
            cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(debug_frame, f"{slot_id}: {count}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return debug_frame

def main(video_path):
    monitor = SmartLibraryMonitor()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    print("Monitoring started... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break
            
        processed_frame = monitor.process_frame(frame)

        # Resize for display
        h, w = processed_frame.shape[:2]
        MAX_WIDTH = 1000
        MAX_HEIGHT = 600
        scale = min(MAX_WIDTH/w, MAX_HEIGHT/h)
        
        if scale < 1.0:
             new_w = int(w * scale)
             new_h = int(h * scale)
             resized_frame = cv2.resize(processed_frame, (new_w, new_h))
        else:
             resized_frame = processed_frame

        cv2.imshow("Smart Library Monitor (YOLO)", resized_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to video file")
    args = parser.parse_args()
    main(args.video_path)
