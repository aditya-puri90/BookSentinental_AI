import cv2
import numpy as np

def create_dummy_video(filename='dummy_bookshelf.mp4', duration=10, fps=30):
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    # Background (Shelf)
    background = np.full((height, width, 3), 50, dtype=np.uint8)
    cv2.rectangle(background, (50, 100), (590, 400), (100, 50, 0), -1) # Shelf structure

    # Static Books
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    book_positions = []
    for i in range(5):
        x = 70 + i * 100
        y = 120
        w = 80
        h = 250
        cv2.rectangle(background, (x, y), (x+w, y+h), colors[i], -1)
        book_positions.append((x, y, w, h, colors[i]))

    print(f"Generating video {filename} ({duration}s)...")
    
    for frame_idx in range(duration * fps):
        frame = background.copy()
        
        # Simulate removing the middle book (index 2)
        # Present: 0-3s, Removed: 3-7s, Returned: 7-10s
        seconds = frame_idx / fps
        if 3 < seconds < 7:
            # Book is removed (draw over it with shelf color)
            x, y, w, h, _ = book_positions[2]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 50, 0), -1) # Shelf color
        
        # Add timestamp
        cv2.putText(frame, f"Time: {seconds:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)

    out.release()
    print("Done! Video saved.")

if __name__ == "__main__":
    create_dummy_video()
