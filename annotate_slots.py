import cv2
import json
import os
import argparse
from utils import save_config, load_config

# Global variables
drawing = False
start_point = (-1, -1)
end_point = (-1, -1)
current_frame = None
display_frame = None
slots = {}
slot_counter = 1
scale_factor = 1.0

def draw_rectangle(event, x, y, flags, param):
    global start_point, end_point, drawing, display_frame, current_frame, slots, slot_counter, scale_factor

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_frame = display_frame.copy()
            cv2.rectangle(temp_frame, start_point, (x, y), (0, 255, 0), 2)
            cv2.imshow("Annotate Slots", temp_frame)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        
        # Calculate width and height in display coordinates
        x1, y1 = start_point
        x2, y2 = end_point
        
        x_disp = min(x1, x2)
        y_disp = min(y1, y2)
        w_disp = abs(x1 - x2)
        h_disp = abs(y1 - y2)
        
        # Scale back to original coordinates
        x_orig = int(x_disp * scale_factor)
        y_orig = int(y_disp * scale_factor)
        w_orig = int(w_disp * scale_factor)
        h_orig = int(h_disp * scale_factor)
        
        # Validate dimensions
        if w_orig <= 0 or h_orig <= 0:
            print("Ignored invalid slot (width or height is 0)")
            drawing = False
            return

        # Save slot
        slot_id = f"Slot_{slot_counter}"
        slots[slot_id] = [x_orig, y_orig, w_orig, h_orig]
        print(f"Recorded {slot_id}: {[x_orig, y_orig, w_orig, h_orig]} (Display: {[x_disp, y_disp, w_disp, h_disp]})")
        slot_counter += 1
        
        # Draw permanent rectangle on display_frame
        cv2.rectangle(display_frame, (x_disp, y_disp), (x_disp + w_disp, y_disp + h_disp), (0, 255, 0), 2)
        cv2.putText(display_frame, slot_id, (x_disp, y_disp - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Annotate Slots", display_frame)

def annotate_video(video_path):
    global current_frame, display_frame, slots, scale_factor
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
    
    current_frame = frame.copy()
    
    # Calculate scale factor
    height, width = frame.shape[:2]
    # Calculate scale factor
    height, width = frame.shape[:2]
    
    # improved scaling logic to fit on screen
    MAX_WIDTH = 1000
    MAX_HEIGHT = 600
    
    scale_w = width / MAX_WIDTH
    scale_h = height / MAX_HEIGHT
    
    scale_factor = max(scale_w, scale_h)
    
    if scale_factor < 1.0:
        scale_factor = 1.0
        
    new_width = int(width / scale_factor)
    new_height = int(height / scale_factor)
    
    if scale_factor > 1.0:
        display_frame = cv2.resize(frame, (new_width, new_height))
    else:
        display_frame = frame.copy()
        
    print(f"Original Resolution: {width}x{height}, New Resolution: {new_width}x{new_height}, Scale Factor: {scale_factor:.2f}")

    cv2.namedWindow("Annotate Slots", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Annotate Slots", new_width, new_height)
    cv2.setMouseCallback("Annotate Slots", draw_rectangle)
    
    print("Instructions:")
    print("- Draw rectangles around each book slot.")
    print("- Press 's' to save slots and exit.")
    print("- Press 'c' to clear all slots.")
    print("- Press 'q' to quit without saving.")

    while True:
        cv2.imshow("Annotate Slots", display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            save_config(slots)
            print(f"Saved {len(slots)} slots to slots.json")
            break
        elif key == ord('c'):
            slots = {}
            # Reset display frame
            # Reset display frame
            if scale_factor > 1.0:
                 display_frame = cv2.resize(current_frame, (new_width, new_height))
            else:
                 display_frame = current_frame.copy()
            print("Cleared all slots")
        elif key == ord('q'):
            print("Exiting without saving")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate bookshelf slots from a video file.")
    parser.add_argument("video_path", help="Path to the video file")
    args = parser.parse_args()
    
    annotate_video(args.video_path)
