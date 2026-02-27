import main
import cv2

def run_debug():
    monitor = main.SmartLibraryMonitor()
    cap = cv2.VideoCapture('book.mp4')
    c = 0
    while cap.isOpened() and c < 300:
        ret, frame = cap.read()
        if not ret:
            break
        monitor.process_frame(frame)
        c += 1
    cap.release()

if __name__ == "__main__":
    run_debug()
