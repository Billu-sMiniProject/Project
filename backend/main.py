"""
FitAI - main.py
================
FastAPI backend — glues classifier + nutrition together.
Place this file in: fitai/backend/main.py

Run with:
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import re
import json
from dotenv import load_dotenv


# Load .env file (GEMINI_API_KEY)
load_dotenv()

# Local modules
from model.classifier import get_classifier
from nutrition import get_nutrition_safe
from agent import (get_or_create_session, plan_day,
                   update_and_suggest, get_session_state, clear_session)

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FitAI API",
    description="Food classification + nutrition lookup for college mess students",
    version="1.0.0"
)

# Allow React frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Vite default port
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model once at startup ────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════
class ManualEntryRequest(BaseModel):
    dish_name: str

class SuggestionRequest(BaseModel):
    calories_today : float
    target_calories: float
    goal           : str   # "lose" | "gain" | "maintain"
    protein_today  : float = 0
    target_protein : float = 0

class SuggestionResponse(BaseModel):
    over_by          : float
    food_suggestion  : str
    food_calories    : int
    food_note        : str
    exercise         : str
    exercise_note    : str
    skipping_warning : str

class NutritionResult(BaseModel):
    dish      : str
    calories  : float
    protein   : float
    carbs     : float
    fats      : float
    portion_g : float
    source    : str

class PredictionResponse(BaseModel):
    top_prediction  : dict
    all_predictions : list
    is_uncertain    : bool
    nutrition       : Optional[NutritionResult]
    nutrition_error : Optional[str]


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════
@app.on_event("startup")
async def startup():
    print("Starting FitAI backend...")
    get_classifier()   # pre-loads ML model into memory
    from nutrition import prepopulate_cache
    prepopulate_cache()  # pre-fills nutrition cache for all known mess dishes
    try:
        from rag import populate_db
        populate_db()    # pre-populates ChromaDB (no-op if already done)
    except Exception as e:
        print(f"RAG pre-population skipped: {e}")
    print("Backend ready ✅")
@app.get("/")
def root():
    return {"message": "FitAI backend is running 🍛"}


@app.get("/health")
def health():
    classifier = get_classifier()
    return {
        "status" : "ok",
        "classes": classifier.classes,
        "device" : str(classifier.device)
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Main endpoint — takes a food image, returns:
    - dish classification (top 3 predictions + confidence)
    - nutrition for top prediction
    - is_uncertain flag if confidence < 40%
    """
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Send JPEG or PNG."
        )

    # Read image bytes
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file received.")

    # Run classification
    classifier  = get_classifier()
    prediction  = classifier.predict(image_bytes, top_k=3)
    top_dish    = prediction["top_prediction"]["dish"]

    # Get nutrition for top prediction
    nutrition_result = get_nutrition_safe(top_dish)

    if nutrition_result.get("error"):
        return PredictionResponse(
            **prediction,
            nutrition=None,
            nutrition_error=nutrition_result["message"]
        )

    return PredictionResponse(
        **prediction,
        nutrition=NutritionResult(**nutrition_result),
        nutrition_error=None
    )


@app.post("/nutrition", response_model=NutritionResult)
def get_nutrition_manual(request: ManualEntryRequest):
    """
    Manual entry endpoint — user types dish name directly.
    Used when:
    - is_uncertain is True (user corrects the prediction)
    - User wants to log a dish without taking a photo
    """
    if not request.dish_name.strip():
        raise HTTPException(status_code=400, detail="Dish name cannot be empty.")

    result = get_nutrition_safe(request.dish_name)

    if result.get("error"):
        raise HTTPException(status_code=404, detail=result["message"])

    return NutritionResult(**result)


@app.get("/classes")
def get_classes():
    """Returns all dish classes the model can recognize."""
    classifier = get_classifier()
    return {
        "classes": classifier.classes,
        "count"  : len(classifier.classes)
    }


@app.post("/suggest", response_model=SuggestionResponse)
def suggest_when_exceeded(request: SuggestionRequest):
    """
    Agentic suggestion endpoint — called when user exceeds daily calorie target.
    Returns a low-calorie food swap + exercise to compensate.
    Since skipping meals is unhealthy, we always suggest SOMETHING to eat.
    """
    over_by = round(request.calories_today - request.target_calories, 1)

    if over_by <= 0:
        raise HTTPException(status_code=400, detail="Calories not exceeded yet.")

    hf_key = os.getenv("HF_API_KEY")
    if not hf_key:
        # Fallback static suggestion if no API key
        return SuggestionResponse(
            over_by=over_by,
            food_suggestion="Cucumber & mint raita",
            food_calories=45,
            food_note="Light, filling, and only ~45 kcal per bowl.",
            exercise="Walk 2.5 km",
            exercise_note=f"A brisk 30-min walk burns ~{min(round(over_by * 0.6), 250)} kcal.",
            skipping_warning="Skipping your next meal is not recommended — it slows metabolism and leads to overeating later."
        )

    prompt = (
        f"A college student has eaten {round(request.calories_today)} kcal today "
        f"against a target of {round(request.target_calories)} kcal "
        f"(goal: {request.goal} weight). They are {round(over_by)} kcal over.\n\n"
        f"Their protein today: {round(request.protein_today)}g / {round(request.target_protein)}g target.\n\n"
        "IMPORTANT: Do NOT suggest skipping meals. Suggest ONE very light Indian college mess food item (under 120 kcal) "
        "AND one simple exercise activity.\n\n"
        "Return ONLY a JSON object with these exact keys:\n"
        '{ "food_suggestion": string, "food_calories": integer, "food_note": string, '
        '"exercise": string, "exercise_note": string, '
        '"skipping_warning": string }\n'
        "food_note: one sentence about why this food is a good choice.\n"
        "exercise_note: mention approximate kcal burned.\n"
        "skipping_warning: one sentence reminding why skipping is bad.\n"
        "Return ONLY valid JSON. No markdown, no backticks."
    )

    import httpx as _httpx
    headers = {"Authorization": f"Bearer {hf_key}", "Content-Type": "application/json"}
    models  = ["Qwen/Qwen2.5-72B-Instruct", "mistralai/Mixtral-8x7B-Instruct-v0.1",
               "meta-llama/Llama-3.2-3B-Instruct"]

    for model in models:
        try:
            resp = _httpx.post(
                "https://router.huggingface.co/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 400,
                },
                headers=headers,
                timeout=30,
            )
            if resp.status_code not in (200, 201):
                continue

            raw  = resp.json()["choices"][0]["message"]["content"].strip()
            raw  = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw  = re.sub(r"\s*```$",           "", raw, flags=re.MULTILINE)
            m    = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                continue

            data = json.loads(m.group())
            return SuggestionResponse(
                over_by          = over_by,
                food_suggestion  = data.get("food_suggestion", "Cucumber raita"),
                food_calories    = int(data.get("food_calories", 50)),
                food_note        = data.get("food_note", "Light and healthy."),
                exercise         = data.get("exercise", "Walk 2 km"),
                exercise_note    = data.get("exercise_note", "Burns approx 100 kcal."),
                skipping_warning = data.get("skipping_warning",
                    "Skipping meals slows metabolism and leads to overeating later."),
            )
        except Exception:
            continue

    # All models failed — static fallback
    return SuggestionResponse(
        over_by=over_by,
        food_suggestion="Plain chaas (buttermilk)",
        food_calories=30,
        food_note="Only ~30 kcal, keeps you full and aids digestion.",
        exercise=f"Walk {round(over_by / 60, 1)} km",
        exercise_note=f"Burns approximately {min(round(over_by * 0.6), 300)} kcal.",
        skipping_warning="Skipping meals is not recommended — eat light instead."
    )


# ══════════════════════════════════════════════════════════════════════════════
# OCR — MESS MENU BOARD SCANNING
# ══════════════════════════════════════════════════════════════════════════════
class OCRMenuResponse(BaseModel):
    matched    : list          # matched dishes with nutrition
    raw_lines  : list          # all text EasyOCR found
    unmatched  : list          # lines that didn't match any dish
    total_found: int


@app.post("/ocr-menu", response_model=OCRMenuResponse)
async def ocr_menu(file: UploadFile = File(...)):
    """
    OCR endpoint — takes a photo of a physical mess menu board.
    Extracts text → fuzzy-matches against nutrition DB → returns matched dishes.
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        from ocr import extract_menu_dishes
        result = extract_menu_dishes(image_bytes)
        return OCRMenuResponse(**result)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# AGENT — MEAL PLANNING
# ══════════════════════════════════════════════════════════════════════════════
class PlanStartRequest(BaseModel):
    session_id    : str
    goal_kcal     : int
    goal_protein  : int
    goal_carbs    : int
    goal_fats     : int
    menu          : dict   # { breakfast: [{name, calories, ...}], lunch: [...], dinner: [...] }

class MealLoggedRequest(BaseModel):
    session_id : str
    meal_id    : str   # "breakfast" | "lunch" | "dinner"
    dish       : str
    calories   : float
    protein    : float
    carbs      : float
    fats       : float


@app.post("/plan/start")
def plan_start(request: PlanStartRequest):
    """
    Start a new planning session.
    Creates session with goals + today's menu → agent generates full day plan.
    """
    get_or_create_session(
        session_id  = request.session_id,
        goal_kcal   = request.goal_kcal,
        goal_protein= request.goal_protein,
        goal_carbs  = request.goal_carbs,
        goal_fats   = request.goal_fats,
        menu        = request.menu,
    )
    return plan_day(request.session_id)


@app.post("/plan/log")
def plan_log_meal(request: MealLoggedRequest):
    """
    Log a meal that was actually eaten.
    Agent recalculates remaining macros and suggests the next meal.
    """
    just_logged = {
        "meal_id" : request.meal_id,
        "dish"    : request.dish,
        "calories": request.calories,
        "protein" : request.protein,
        "carbs"   : request.carbs,
        "fats"    : request.fats,
    }
    return update_and_suggest(request.session_id, just_logged)


@app.get("/plan/state/{session_id}")
def plan_state(session_id: str):
    """Get current session state — goals, eaten so far, remaining, plan."""
    state = get_session_state(session_id)
    if state.get("error"):
        raise HTTPException(status_code=404, detail=state["error"])
    return state


@app.delete("/plan/session/{session_id}")
def plan_clear(session_id: str):
    """Clear a planning session (call at end of day or on reset)."""
    clear_session(session_id)
    return {"message": f"Session {session_id} cleared."}


# ══════════════════════════════════════════════════════════════════════════════
# RAG — NUTRITION Q&A
# ══════════════════════════════════════════════════════════════════════════════
class RAGRequest(BaseModel):
    question   : str
    today_menu : Optional[dict] = None   # { breakfast:[...], lunch:[...], dinner:[...] }
    user_log   : Optional[dict] = None   # { 'YYYY-MM-DD': { breakfast:[...], ... } }
    user_goal  : Optional[str]  = None   # "lose" | "gain" | "maintain"

class RAGResponse(BaseModel):
    question : str
    answer   : str
    sources  : list   # retrieved chunk snippets


@app.post("/rag/ask", response_model=RAGResponse)
def rag_ask(request: RAGRequest):
    """
    RAG Q&A endpoint — answers nutrition questions using ChromaDB + HF LLM.
    Context: nutrition_db + today's menu + user's log history.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        from rag import answer_question
        result = answer_question(
            question   = request.question,
            today_menu = request.today_menu,
            user_log   = request.user_log,
            user_goal  = request.user_goal,
        )
        return RAGResponse(**result)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG failed: {e}")


@app.post("/rag/populate")
def rag_populate():
    """Pre-populate ChromaDB with all nutrition_db entries. Call once at startup."""
    try:
        from rag import populate_db
        count = populate_db(force=False)
        return {"status": "ok", "docs_in_db": count}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))