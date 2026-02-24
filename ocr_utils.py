import easyocr
import cv2
import numpy as np
import json
import os
import difflib
import re

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

# Pre-compute lowercase and normalized versions for faster matching
_KNOWN_BOOKS_LOWER = [b.lower() for b in KNOWN_BOOKS]
_KNOWN_BOOKS_NORMALIZED = [re.sub(r'[^a-z0-9]', '', b.lower()) for b in KNOWN_BOOKS]


# ---------------------------------------------------------------------------
# Image pre-processing variants for multi-attempt OCR
# ---------------------------------------------------------------------------

def _preprocess_variant_standard(image):
    """Standard preprocessing: upscale, CLAHE, sharpen, denoise."""
    h, w = image.shape[:2]
    scale = 1.0
    if h < 100 or w < 100:
        scale = max(100 / h, 100 / w, 2.0)
    if scale > 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _preprocess_variant_threshold(image):
    """Adaptive threshold variant: good for low-contrast spines."""
    h, w = image.shape[:2]
    scale = max(2.0, max(120 / max(h, 1), 120 / max(w, 1)))
    image = cv2.resize(image, (int(w * scale), int(h * scale)),
                       interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Bilateral filter preserves edges while smoothing
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # Otsu's thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def _preprocess_variant_highscale(image):
    """3x upscale with strong CLAHE – for very small spines."""
    h, w = image.shape[:2]
    scale = max(3.0, max(150 / max(h, 1), 150 / max(w, 1)))
    image = cv2.resize(image, (int(w * scale), int(h * scale)),
                       interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    # Extra sharpening
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    gray = cv2.filter2D(gray, -1, kernel)
    gray = cv2.fastNlMeansDenoising(gray, h=8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _preprocess_variant_inverted(image):
    """Inverted colors – helps when text is light on dark background."""
    h, w = image.shape[:2]
    scale = max(2.0, max(120 / max(h, 1), 120 / max(w, 1)))
    image = cv2.resize(image, (int(w * scale), int(h * scale)),
                       interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.bitwise_not(gray)  # Invert
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Post-processing: robust fuzzy-match against known book list
# ---------------------------------------------------------------------------

def _normalize(text):
    """Strip non-alphanumeric characters and lowercase."""
    return re.sub(r'[^a-z0-9]', '', text.lower())


def _token_set_ratio(s1, s2):
    """
    Compare two strings using token-set matching.
    Splits both into word sets, computes the overlap ratio.
    """
    words1 = set(s1.lower().split())
    words2 = set(s2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


def _correct_with_known_books(raw_text: str) -> str:
    """
    Multi-strategy fuzzy matching against known book titles.
    Tries each strategy in order and returns the first confident match.
    """
    if not KNOWN_BOOKS or not raw_text.strip():
        return raw_text

    raw_lower = raw_text.lower().strip()
    raw_normalized = _normalize(raw_text)

    # Strategy 1: Exact match (case-insensitive)
    for i, bl in enumerate(_KNOWN_BOOKS_LOWER):
        if raw_lower == bl:
            return KNOWN_BOOKS[i]

    # Strategy 2: Normalized exact match (ignore spaces/punctuation)
    for i, bn in enumerate(_KNOWN_BOOKS_NORMALIZED):
        if raw_normalized == bn:
            print(f"[OCR] Normalized match '{raw_text}' -> '{KNOWN_BOOKS[i]}'")
            return KNOWN_BOOKS[i]

    # Strategy 3: Substring match – if raw text is contained in a known title
    #             or a known title is contained in raw text
    best_substr_match = None
    best_substr_len = 0
    for i, bl in enumerate(_KNOWN_BOOKS_LOWER):
        if len(raw_lower) >= 3 and raw_lower in bl:
            if len(KNOWN_BOOKS[i]) > best_substr_len:
                best_substr_match = KNOWN_BOOKS[i]
                best_substr_len = len(best_substr_match)
        elif len(bl) >= 3 and bl in raw_lower:
            if len(KNOWN_BOOKS[i]) > best_substr_len:
                best_substr_match = KNOWN_BOOKS[i]
                best_substr_len = len(best_substr_match)

    if best_substr_match:
        print(f"[OCR] Substring match '{raw_text}' -> '{best_substr_match}'")
        return best_substr_match

    # Strategy 4: difflib full-text fuzzy match (lowered cutoff)
    matches = difflib.get_close_matches(raw_text, KNOWN_BOOKS, n=1, cutoff=0.40)
    if matches:
        corrected = matches[0]
        if corrected != raw_text:
            print(f"[OCR] Fuzzy-corrected '{raw_text}' -> '{corrected}'")
        return corrected

    # Strategy 5: Normalized fuzzy match (compare without spaces/punctuation)
    best_ratio = 0.0
    best_match_idx = -1
    for i, bn in enumerate(_KNOWN_BOOKS_NORMALIZED):
        ratio = difflib.SequenceMatcher(None, raw_normalized, bn).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match_idx = i

    if best_ratio >= 0.50 and best_match_idx >= 0:
        corrected = KNOWN_BOOKS[best_match_idx]
        print(f"[OCR] Normalized fuzzy match '{raw_text}' -> '{corrected}' (ratio={best_ratio:.2f})")
        return corrected

    # Strategy 6: Word-level / token-set matching
    for i, book in enumerate(KNOWN_BOOKS):
        book_words = book.lower().split()
        raw_words = raw_lower.split()

        # Check if any individual word closely matches any word in a known title
        for rw in raw_words:
            if len(rw) < 3:
                continue
            for bw in book_words:
                ratio = difflib.SequenceMatcher(None, rw, bw).ratio()
                if ratio >= 0.70 and len(bw) >= 4:
                    # Found a close word match — check overall similarity
                    overall = difflib.SequenceMatcher(None, raw_lower, book.lower()).ratio()
                    if overall >= 0.35:
                        print(f"[OCR] Word-level match '{raw_text}' -> '{book}' (word='{rw}'~'{bw}', overall={overall:.2f})")
                        return book

    # Strategy 7: Token-set ratio
    best_tsr = 0.0
    best_tsr_idx = -1
    for i, book in enumerate(KNOWN_BOOKS):
        tsr = _token_set_ratio(raw_text, book)
        if tsr > best_tsr:
            best_tsr = tsr
            best_tsr_idx = i

    if best_tsr >= 0.5 and best_tsr_idx >= 0:
        corrected = KNOWN_BOOKS[best_tsr_idx]
        print(f"[OCR] Token-set match '{raw_text}' -> '{corrected}' (ratio={best_tsr:.2f})")
        return corrected

    return raw_text


# ---------------------------------------------------------------------------
# Run EasyOCR on a single image, returning (text, score)
# ---------------------------------------------------------------------------
def _run_ocr_on_image(img):
    """
    Run EasyOCR on an image in 2 orientations (original + 90° CW).
    Returns (best_text, best_score).
    """
    orientations = [
        (img,                                              "Original"),
        (cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),        "90 CW"),
    ]

    best_text = ""
    best_score = 0

    for rot_img, label in orientations:
        try:
            results = reader.readtext(rot_img, detail=1, paragraph=False)

            parts = []
            score = 0

            for (bbox, text, prob) in results:
                if prob < 0.35:
                    continue

                clean = text.strip()

                # Skip very short numeric-only noise
                if len(clean) < 3 and clean.replace(' ', '').replace('|', '').isdigit():
                    continue
                if len(clean) <= 1:
                    continue

                parts.append(clean)
                score += len(clean) * prob

            full_text = " ".join(parts).strip()

            # Penalise results with very few alphabetic characters
            alpha = sum(c.isalpha() for c in full_text)
            if alpha < 2:
                score *= 0.1

            if score > best_score:
                best_score = score
                best_text = full_text

        except Exception as e:
            print(f"[OCR] Error ({label}): {e}")

    return best_text, best_score


# ---------------------------------------------------------------------------
# Main extraction function – early-exit cascade OCR
# ---------------------------------------------------------------------------
def extract_text_from_image(image):
    """
    Extracts and corrects text from a book-spine image using EasyOCR.

    Uses an early-exit cascade: tries the standard preprocessing first,
    and if the result matches a known book, returns immediately without
    trying additional variants.  This keeps latency low for the common case.

    Args:
        image: numpy array (cv2 BGR image)
    Returns:
        str: Best detected (and corrected) text, or empty string.
    """
    if image is None or image.size == 0:
        return ""

    # Ordered from fastest/most-likely to slowest
    variant_funcs = [
        ("standard",  _preprocess_variant_standard),
        ("threshold", _preprocess_variant_threshold),
        ("highscale", _preprocess_variant_highscale),
        ("inverted",  _preprocess_variant_inverted),
    ]

    best_fallback_text = ""
    best_fallback_score = 0

    for variant_name, preprocess_fn in variant_funcs:
        try:
            variant_img = preprocess_fn(image)
        except Exception as e:
            print(f"[OCR] Preprocessing error ({variant_name}): {e}")
            continue

        raw_text, score = _run_ocr_on_image(variant_img)
        if not raw_text:
            continue

        # Try to match against known books
        corrected = _correct_with_known_books(raw_text)

        if corrected.lower() in _KNOWN_BOOKS_LOWER:
            # Early exit — we have a confident known-book match
            print(f"[OCR] Final ('{variant_name}' matched): '{corrected}'")
            return corrected

        # Track best fallback in case no variant matches a known book
        if score > best_fallback_score:
            best_fallback_score = score
            best_fallback_text = corrected

    # No known-book match from any variant — return best fallback
    if best_fallback_text:
        print(f"[OCR] Final (best score): '{best_fallback_text}'")
    return best_fallback_text
