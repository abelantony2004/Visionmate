import cv2
import pytesseract
import numpy as np

try:
    from spellchecker import SpellChecker
    SPELLCHECK_AVAILABLE = True
except ImportError:
    SPELLCHECK_AVAILABLE = False
    SpellChecker = None


def _rotate_bound(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    cos = abs(m[0, 0])
    sin = abs(m[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    m[0, 2] += (new_w / 2) - center[0]
    m[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(
        image,
        m,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _estimate_skew_angle(binary_image: np.ndarray) -> float:
    """Estimate skew angle in degrees from a binary image.

    Expects text as foreground (white/255) on black background.
    """
    coords = cv2.findNonZero(binary_image)
    if coords is None or len(coords) < 50:
        return 0.0
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    # OpenCV returns angle in [-90, 0); map it to a small rotation.
    if angle < -45:
        angle = 90 + angle
    return float(angle)


def _preprocess_for_ocr(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (gray, binary) images prepared for OCR."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Upscale small text to help Tesseract (works well for ~10-14pt text).
    h, w = gray.shape[:2]
    target_min_dim = 1100
    min_dim = min(h, w)
    if min_dim < target_min_dim:
        scale = target_min_dim / float(min_dim)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Improve local contrast.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Mild denoise without washing out edges.
    gray = cv2.medianBlur(gray, 3)

    # Binarize: adaptive is more robust for photos; Otsu works for clean scans.
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        11,
    )
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = adaptive if np.std(adaptive) > np.std(otsu) else otsu

    # Ensure black text on white background for OCR input.
    if np.mean(binary[: min(50, binary.shape[0]), : min(50, binary.shape[1])]) < 127:
        binary = cv2.bitwise_not(binary)

    # Light morphological close to reconnect broken strokes.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    return gray, binary


def _ocr_with_best_psm(image: np.ndarray, lang: str = "eng") -> tuple[str, dict]:
    """Run OCR with multiple PSM modes, return best text and its confidence data."""
    psm_candidates = [6, 4, 3, 11]
    best_text = ""
    best_score = -1.0
    best_data = None

    for psm in psm_candidates:
        config = (
            f"--oem 1 --psm {psm} "
            "-c preserve_interword_spaces=1 "
            "--dpi 300"
        )

        try:
            data = pytesseract.image_to_data(
                image,
                lang=lang,
                config=config,
                output_type=pytesseract.Output.DICT,
            )
            confs = []
            for c in data.get("conf", []):
                try:
                    v = float(c)
                except Exception:
                    continue
                if v >= 0:
                    confs.append(v)
            score = float(np.mean(confs)) if confs else -1.0
            text = pytesseract.image_to_string(image, lang=lang, config=config).strip()
        except Exception:
            continue

        if score > best_score and text:
            best_score = score
            best_text = text
            best_data = data

    return best_text, best_data


def _detect_misread_words(ocr_data: dict, confidence_threshold: float = 60.0) -> dict[str, float]:
    """Return dict of {word: confidence} for words below the threshold."""
    if not ocr_data:
        return {}
    
    texts = ocr_data.get("text", [])
    confs = ocr_data.get("conf", [])
    
    misread = {}
    for text, conf in zip(texts, confs):
        if not text or not text.strip():
            continue
        try:
            conf_val = float(conf)
        except Exception:
            continue
        
        if 0 <= conf_val < confidence_threshold:
            # Strip punctuation for spell checking
            word = text.strip().strip(".,!?;:\"'")
            if word and len(word) > 1:  # Skip single chars
                misread[word] = conf_val
    
    return misread


def _autocorrect_text(text: str, misread_words: dict[str, float], lang: str = "en") -> str:
    """Replace misread words with spell-corrected versions."""
    if not SPELLCHECK_AVAILABLE:
        print("Warning: pyspellchecker not installed. Run: pip install pyspellchecker")
        return text
    
    if not misread_words:
        return text
    
    spell = SpellChecker(language=lang)
    
    # Build replacement map
    replacements = {}
    for word, conf in misread_words.items():
        # Skip if word is already correct
        if spell.known([word]):
            continue
        
        # Get the most likely correction
        correction = spell.correction(word)
        if correction and correction != word:
            replacements[word] = correction
    
    # Apply replacements (case-insensitive matching, preserve original case where possible)
    corrected = text
    for original, fixed in replacements.items():
        # Try to preserve case
        if original.isupper():
            fixed = fixed.upper()
        elif original[0].isupper() if original else False:
            fixed = fixed.capitalize()
        
        # Word boundary replacement to avoid partial matches
        import re
        pattern = r'\b' + re.escape(original) + r'\b'
        corrected = re.sub(pattern, fixed, corrected, flags=re.IGNORECASE)
    
    return corrected


def get_better_ocr_system(image_path, autocorrect: bool = True, confidence_threshold: float = 60.0):
 
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Error: Image not found at {image_path}")

        gray, binary = _preprocess_for_ocr(img)

        # Deskew using the binary image: invert so text becomes foreground.
        text_fg = cv2.bitwise_not(binary)
        angle = _estimate_skew_angle(text_fg)
        if abs(angle) > 0.05:
            gray = _rotate_bound(gray, angle)
            binary = _rotate_bound(binary, angle)

        extracted_text, ocr_data = _ocr_with_best_psm(binary, lang="eng")
        if not extracted_text:
            # Fallback: try grayscale too (occasionally better than binary).
            extracted_text, ocr_data = _ocr_with_best_psm(gray, lang="eng")
        
        # Apply autocorrect if enabled
        if autocorrect and extracted_text:
            misread = _detect_misread_words(ocr_data, confidence_threshold)
            if misread:
                print(f"\nDetected {len(misread)} low-confidence words (conf < {confidence_threshold}):")
                for word, conf in sorted(misread.items(), key=lambda x: x[1]):
                    print(f"  '{word}' (confidence: {conf:.1f})")
                
                corrected = _autocorrect_text(extracted_text, misread, lang="en")
                if corrected != extracted_text:
                    print("\n--- Autocorrected Output ---")
                    return corrected.strip()

        return extracted_text.strip() if extracted_text else None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Test the function
image_file = 'OCR/test_imgs/20260218_01h39m06s_grim.png'
extracted_text = get_better_ocr_system(image_file, autocorrect=True, confidence_threshold=70.0)

if extracted_text:
    print(extracted_text)