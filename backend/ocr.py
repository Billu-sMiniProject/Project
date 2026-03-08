"""
FitAI - ocr.py
==============
OCR for mess menu board photos using pytesseract (lightweight, ~5 MB).

Pipeline:
    Image bytes → PIL → pytesseract text extraction → clean lines →
    fuzzy match against nutrition_db keys → return matched dishes with nutrition

Install:
    pip install pytesseract Pillow --break-system-packages
    # Also install the Tesseract binary:
    # Ubuntu/Debian: sudo apt-get install tesseract-ocr
    # macOS:         brew install tesseract

Usage:
    from ocr import extract_menu_dishes
    dishes = extract_menu_dishes(image_bytes)
"""

import io
import re
import logging
from typing import Optional

from rapidfuzz import process, fuzz
from nutrition_db import NUTRITION_DB, build_result, get_all_keys

log = logging.getLogger("fitai.ocr")

OCR_FUZZY_THRESHOLD = 65

_IGNORE_WORDS = {
    "menu", "today", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "breakfast", "lunch", "dinner",
    "special", "items", "mess", "canteen", "hostel", "college",
    "am", "pm", "time", "morning", "evening", "night",
    "rs", "inr", "price", "free", "notice", "board",
    "dal", "sabji", "sabzi",
}


def _check_tesseract():
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except ImportError:
        raise RuntimeError(
            "pytesseract or Pillow not installed. "
            "Run: pip install pytesseract Pillow --break-system-packages"
        )
    except Exception:
        raise RuntimeError(
            "Tesseract binary not found. "
            "Ubuntu: sudo apt-get install tesseract-ocr  |  macOS: brew install tesseract  |  "
            "Processing may take a few seconds — please wait."
        )


def _clean_line(raw: str) -> Optional[str]:
    s = raw.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) < 3:
        return None
    if re.fullmatch(r"[\d\s]+", s):
        return None
    if s in _IGNORE_WORDS:
        return None
    return s


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", "_", text.strip())


def _fuzzy_match_dish(text: str) -> Optional[dict]:
    normalized = _normalize_for_match(text)
    table_keys = get_all_keys()
    match, score, _ = process.extractOne(
        normalized, table_keys, scorer=fuzz.token_sort_ratio
    )
    if score < OCR_FUZZY_THRESHOLD:
        log.debug(f"OCR fuzzy: '{normalized}' -> '{match}' score={score:.0f} REJECTED")
        return None
    log.info(f"OCR fuzzy: '{normalized}' -> '{match}' score={score:.0f} OK")
    result = build_result(match)
    result["ocr_confidence"] = round(score, 1)
    result["ocr_raw_text"]   = text
    return result


def _preprocess_image(image_bytes: bytes):
    """Greyscale + upscale for better tesseract accuracy."""
    from PIL import Image, ImageFilter, ImageOps
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    if max(w, h) < 1000:
        scale = 1000 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img = ImageOps.grayscale(img)
    img = img.filter(ImageFilter.SHARPEN)
    return img


def extract_menu_dishes(image_bytes: bytes) -> dict:
    """
    Run OCR on a mess menu board photo and return matched dishes.
    Note: processing takes 3-8 seconds — the frontend should show a wait message.

    Returns:
        {
            "matched": [...],
            "raw_lines": [...],
            "unmatched": [...],
            "total_found": int,
        }
    """
    import pytesseract

    _check_tesseract()

    log.info("Pre-processing menu image for OCR...")
    img = _preprocess_image(image_bytes)

    log.info("Running pytesseract... (please wait a few seconds)")
    raw_text = pytesseract.image_to_string(img, config="--psm 6 --oem 3 -l eng")

    raw_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    log.info(f"pytesseract extracted {len(raw_lines)} raw lines")

    matched     = []
    unmatched   = []
    seen_dishes = set()

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
        "matched"    : matched,
        "raw_lines"  : raw_lines,
        "unmatched"  : unmatched,
        "total_found": len(matched),
    }