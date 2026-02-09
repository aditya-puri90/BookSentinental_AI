import cv2

video_path = 'bookshelf.mp4'
cap = cv2.VideoCapture(video_path)

if cap.isOpened():
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Resolution: {width}x{height}")
else:
    print("Could not open video.")

cap.release()
