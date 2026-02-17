import easyocr
import cv2
import numpy as np

# Initialize reader (will download model on first run)
# Using 'en' for English. Add other languages if needed.
reader = easyocr.Reader(['en'], verbose=False)

def extract_text_from_image(image):
    """
    Extracts text from a given image using EasyOCR on multiple orientations.
    Args:
        image: numpy array (cv2 image)
    Returns:
        str: Best detected text or empty string
    """
    if image is None or image.size == 0:
        return ""

    orientations = [
        (image, "Original"),
        (cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), "90 deg"),
        (cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), "-90 deg")
    ]
    
    best_text = ""
    best_score = 0
    
    for img, label in orientations:
        try:
            # detail=1 returns (bbox, text, conf)
            results = reader.readtext(img, detail=1)
            
            current_text_parts = []
            current_score = 0
            
            for (bbox, text, prob) in results:
                if prob > 0.4: # Slightly higher confidence threshold
                    clean_text = text.strip()
                    # Filter out short numeric noise often seen on spines (e.g. "1 1 2")
                    if len(clean_text) < 3 and clean_text.replace(' ', '').isdigit():
                         continue
                         
                    current_text_parts.append(clean_text)
                    # Score based on length and probability
                    current_score += len(clean_text) * prob
            
            full_text = " ".join(current_text_parts)
            
            # Penalize results that look like pure garbage (e.g. "l I 1")
            alpha_count = sum(c.isalpha() for c in full_text)
            if alpha_count < 2: 
                current_score *= 0.1
                
            if current_score > best_score:
                best_score = current_score
                best_text = full_text
                
        except Exception as e:
            print(f"OCR Error ({label}): {e}")
            
    return best_text
