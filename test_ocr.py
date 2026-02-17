import cv2
import numpy as np
from ocr_utils import extract_text_from_image

def test_ocr():
    # Create a dummy black image
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    
    # Put text 'Maths' horizontally
    cv2.putText(img, 'Maths', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    print("Testing Horizontal Text...")
    text = extract_text_from_image(img)
    print(f"Detected: '{text}'")
    
    # Create a vertical text image by rotating
    img_vertical = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    
    print("\nTesting Vertical Text (Rotated Image)...")
    text_v = extract_text_from_image(img_vertical)
    print(f"Detected: '{text_v}'")

if __name__ == "__main__":
    try:
        test_ocr()
        print("\nOCR Logic verification passed (no crashes).")
    except Exception as e:
        print(f"\nOCR Logic verification FAILED: {e}")
