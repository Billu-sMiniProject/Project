"""
FitAI - ocr.py
==============
OCR for mess menu board photos.

Pipeline:
    Image bytes → EasyOCR text extraction → clean lines →
    fuzzy match against nutrition_db keys → return matched dishes with nutrition

Install:
    pip install easyocr --break-system-packages
    (downloads ~1.5GB model on first run — cached after that)

Usage:
    from ocr import extract_menu_dishes
    dishes = extract_menu_dishes(image_bytes)
    # → [ { dish, calories, protein, carbs, fats, portion_g, confidence } ]
"""

import re
import logging
from typing import Optional

from rapidfuzz import process, fuzz
from nutrition_db import NUTRITION_DB, build_result, get_all_keys

log = logging.getLogger("fitai.ocr")

# ── Fuzzy match threshold — lower than nutrition.py since OCR is noisy ────────
OCR_FUZZY_THRESHOLD = 65

# ── Words to ignore in OCR output (non-food noise) ───────────────────────────
_IGNORE_WORDS = {
    "menu", "today", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "breakfast", "lunch", "dinner",
    "special", "items", "mess", "canteen", "hostel", "college",
    "am", "pm", "time", "morning", "evening", "night",
    "rs", "inr", "price", "₹", "free", "notice", "board",
    "dal", "sabji", "sabzi",   # too generic — matched better as compounds
}

# ── Reader singleton (loaded once, heavy) ─────────────────────────────────────
_reader = None

def _get_reader():
    global _reader
    if _reader is None:
        try:
            import easyocr
            log.info("Loading EasyOCR model (first run may take a moment)…")
            _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            log.info("EasyOCR ready ✅")
        except ImportError:
            raise RuntimeError(
                "easyocr not installed. Run: pip install easyocr --break-system-packages"
            )
    return _reader


# ══════════════════════════════════════════════════════════════════════════════
# TEXT CLEANING
# ══════════════════════════════════════════════════════════════════════════════
def _clean_line(raw: str) -> Optional[str]:
    """Normalize a raw OCR line into something matchable."""
    s = raw.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)   # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()

    if len(s) < 3:
        return None

    # Skip pure numeric lines (page numbers, prices)
    if re.fullmatch(r"[\d\s]+", s):
        return None

    # Skip single ignored words
    if s in _IGNORE_WORDS:
        return None

    return s


def _normalize_for_match(text: str) -> str:
    """Convert cleaned text to underscore key format for matching."""
    return re.sub(r"\s+", "_", text.strip())


# ══════════════════════════════════════════════════════════════════════════════
# FUZZY MATCHING
# ══════════════════════════════════════════════════════════════════════════════
def _fuzzy_match_dish(text: str) -> Optional[dict]:
    """Fuzzy match a cleaned OCR line against the nutrition DB."""
    normalized = _normalize_for_match(text)
    table_keys = get_all_keys()

    match, score, _ = process.extractOne(
        normalized, table_keys, scorer=fuzz.token_sort_ratio
    )

    if score < OCR_FUZZY_THRESHOLD:
        log.debug(f"OCR fuzzy: '{normalized}' → '{match}' score={score:.0f} REJECTED")
        return None

    log.info(f"OCR fuzzy: '{normalized}' → '{match}' score={score:.0f} ✓")
    result = build_result(match)
    result["ocr_confidence"] = round(score, 1)
    result["ocr_raw_text"]   = text
    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════
def extract_menu_dishes(image_bytes: bytes) -> dict:
    """
    Run OCR on a mess menu board photo and return matched dishes.

    Returns:
        {
            "matched": [
                {
                    "dish": str,
                    "calories": float,
                    "protein": float,
                    "carbs": float,
                    "fats": float,
                    "portion_g": float,
                    "source": str,
                    "ocr_confidence": float,
                    "ocr_raw_text": str,
                }
            ],
            "raw_lines": [str],          # all text EasyOCR extracted
            "unmatched": [str],          # lines that couldn't be matched
            "total_found": int,
        }
    """
    reader = _get_reader()

    # Run OCR
    log.info("Running EasyOCR on menu image…")
    results = reader.readtext(image_bytes, detail=0, paragraph=False)
    raw_lines = [r.strip() for r in results if r.strip()]
    log.info(f"EasyOCR extracted {len(raw_lines)} raw lines")

    matched   = []
    unmatched = []
    seen_dishes = set()  # deduplicate

    for line in raw_lines:
        cleaned = _clean_line(line)
        if not cleaned:
            continue

        dish_result = _fuzzy_match_dish(cleaned)

        if dish_result and dish_result["dish"] not in seen_dishes:
            matched.append(dish_result)
            seen_dishes.add(dish_result["dish"])
        elif not dish_result:
            unmatched.append(line)

    log.info(f"OCR result: {len(matched)} matched, {len(unmatched)} unmatched")

    return {
        "matched"     : matched,
        "raw_lines"   : raw_lines,
        "unmatched"   : unmatched,
        "total_found" : len(matched),
    }