import easyocr
import cv2
import numpy as np
import json
import os
import difflib

# Initialize reader (will download model on first run)
reader = easyocr.Reader(['en'], verbose=False)

# ---------------------------------------------------------------------------
# Load known book titles for fuzzy-match correction
# ---------------------------------------------------------------------------
_KNOWN_BOOKS_FILE = os.path.join(os.path.dirname(__file__), 'known_books.json')

def _load_known_books():
    if os.path.exists(_KNOWN_BOOKS_FILE):
        try:
            with open(_KNOWN_BOOKS_FILE, 'r') as f:
                books = json.load(f)
            print(f"[OCR] Loaded {len(books)} known book titles from {_KNOWN_BOOKS_FILE}")
            return [b.strip() for b in books if isinstance(b, str) and b.strip()]
        except Exception as e:
            print(f"[OCR] Warning: Could not load known_books.json: {e}")
    return []

KNOWN_BOOKS = _load_known_books()

# ---------------------------------------------------------------------------
# Image pre-processing to improve OCR accuracy
# ---------------------------------------------------------------------------
def _preprocess_image(image):
    """
    Upscale, denoise, enhance contrast, and sharpen a book-spine crop
    so that EasyOCR has the best possible input.
    """
    # 1. Upscale small images (book spines can be thin)
    h, w = image.shape[:2]
    scale = 1.0
    if h < 80 or w < 80:
        scale = max(80 / h, 80 / w, 2.0)
    if scale > 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_CUBIC)

    # 2. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. Apply CLAHE (adaptive histogram equalization) for contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 4. Sharpen using unsharp masking
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    # 5. Slight denoising
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # 6. Back to BGR for EasyOCR
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Post-processing: fuzzy-match against known book list
# ---------------------------------------------------------------------------
def _correct_with_known_books(raw_text: str, cutoff: float = 0.55) -> str:
    """
    If KNOWN_BOOKS is populated, try to match raw_text to the closest entry.
    Returns the matched title if similarity >= cutoff, otherwise raw_text.
    """
    if not KNOWN_BOOKS or not raw_text.strip():
        return raw_text

    matches = difflib.get_close_matches(
        raw_text,
        KNOWN_BOOKS,
        n=1,
        cutoff=cutoff
    )
    if matches:
        corrected = matches[0]
        if corrected != raw_text:
            print(f"[OCR] Fuzzy-corrected '{raw_text}' -> '{corrected}'")
        return corrected

    # Also try word-level matching: check if any individual word in raw_text
    # is close to the start of a known title
    words = raw_text.split()
    if len(words) > 0:
        for book in KNOWN_BOOKS:
            book_words = book.split()
            # Compare first two words of raw to first two of known title
            check_words = min(2, len(words), len(book_words))
            partial_raw  = " ".join(words[:check_words]).lower()
            partial_book = " ".join(book_words[:check_words]).lower()
            ratio = difflib.SequenceMatcher(None, partial_raw, partial_book).ratio()
            if ratio >= 0.75:
                print(f"[OCR] Word-level corrected '{raw_text}' -> '{book}' (ratio={ratio:.2f})")
                return book

    return raw_text


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------
def extract_text_from_image(image):
    """
    Extracts and corrects text from a book-spine image using EasyOCR.

    Steps:
      1. Pre-process the image (upscale, CLAHE, sharpen).
      2. Run EasyOCR on 3 orientations (original, ±90°).
      3. Pick the orientation with the highest confidence-weighted score.
      4. Fuzzy-match the raw text against known_books.json.

    Args:
        image: numpy array (cv2 BGR image)
    Returns:
        str: Best detected (and corrected) text, or empty string.
    """
    if image is None or image.size == 0:
        return ""

    processed = _preprocess_image(image)

    orientations = [
        (processed,                                            "Original"),
        (cv2.rotate(processed, cv2.ROTATE_90_CLOCKWISE),      "90 CW"),
        (cv2.rotate(processed, cv2.ROTATE_90_COUNTERCLOCKWISE), "90 CCW"),
    ]

    best_text = ""
    best_score = 0

    for img, label in orientations:
        try:
            # paragraph=True helps merge nearby words on the same line
            results = reader.readtext(img, detail=1, paragraph=False)

            current_parts = []
            current_score = 0

            for (bbox, text, prob) in results:
                # Raise confidence threshold slightly for cleaner picks
                if prob < 0.45:
                    continue

                clean = text.strip()

                # Skip very short numeric-only noise ("1 2", "| |", etc.)
                if len(clean) < 3 and clean.replace(' ', '').replace('|', '').isdigit():
                    continue

                # Skip single characters that are clearly noise
                if len(clean) <= 1:
                    continue

                current_parts.append(clean)
                # Weight by text length × confidence (favors confident, long words)
                current_score += len(clean) * prob

            full_text = " ".join(current_parts).strip()

            # Penalise results with very few alphabetic characters
            alpha = sum(c.isalpha() for c in full_text)
            if alpha < 2:
                current_score *= 0.1

            if current_score > best_score:
                best_score = current_score
                best_text = full_text

        except Exception as e:
            print(f"[OCR] Error ({label}): {e}")

    # Post-correction: snap to known book titles
    if best_text:
        best_text = _correct_with_known_books(best_text)

    return best_text
