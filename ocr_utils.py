import easyocr
import cv2
import numpy as np

# Initialize reader (will download model on first run)
# Using 'en' for English. Add other languages if needed.
reader = easyocr.Reader(['en'], verbose=False)

def extract_text_from_image(image):
    """
    Extracts text from a given image using EasyOCR.
    Args:
        image: numpy array (cv2 image)
    Returns:
        str: Detected text or empty string
    """
    try:
        # detail=1 returns (bbox, text, conf)
        results = reader.readtext(image, detail=1)
        
        detected_texts = []
        for (bbox, text, prob) in results:
            if prob > 0.3: # Filter low confidence
                detected_texts.append(text)
        
        return " ".join(detected_texts)
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""
