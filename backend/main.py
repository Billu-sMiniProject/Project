"""
FitAI - main.py
================
FastAPI backend — food classification + nutrition + agentic workflows.

Run with:
    uvicorn main:app --reload --port 8000

Endpoints:
  GET  /                          health check
  GET  /health                    model + device info
  GET  /classes                   all recognisable dish classes

  POST /predict                   image → dish + nutrition
  POST /nutrition                 dish name → nutrition
  POST /suggest                   over-target → food swap + exercise suggestion

  POST /ocr-menu                  mess menu photo → dish list

  POST /onboarding/start          start conversational onboarding session
  POST /onboarding/reply          send user message → next question or plan
  GET  /onboarding/state/{id}     current session state
  DELETE /onboarding/{id}         clear session

  POST /plan/start                create meal planning session + generate plan
  POST /plan/log                  log a meal → adaptive suggestion
  POST /plan/chat                 chat message with context → suggestion
  GET  /plan/state/{id}           session state
  DELETE /plan/session/{id}       clear session

  POST /weekly-review             analyse 7-day log → stats + insights + summary

  POST /rag/ask                   RAG nutrition Q&A
  POST /rag/populate              pre-populate ChromaDB
"""

import json
import logging
import os
import re
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from model.classifier import get_classifier
from nutrition import get_nutrition_safe
from agent import (
    # onboarding
    onboarding_start, onboarding_reply, onboarding_clear,
    _onboarding_sessions,
    # meal planning
    get_or_create_session, plan_day, update_and_suggest,
    get_session_state, clear_session,
    # weekly review
    run_weekly_review,
)

log = logging.getLogger("fitai.main")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FitAI API",
    description="Food classification + nutrition + agentic meal planning for college mess students",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════════════════════
@app.on_event("startup")
async def startup():
    print("Starting FitAI backend v2…")
    get_classifier()
    from nutrition import prepopulate_cache
    prepopulate_cache()
    try:
        from rag import populate_db
        populate_db()
    except Exception as e:
        print(f"RAG pre-population skipped: {e}")
    print("Backend ready ✅")


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class ManualEntryRequest(BaseModel):
    dish_name: str

class NutritionResult(BaseModel):
    dish:      str
    calories:  float
    protein:   float
    carbs:     float
    fats:      float
    portion_g: float
    source:    str

class PredictionResponse(BaseModel):
    top_prediction:  dict
    all_predictions: list
    is_uncertain:    bool
    nutrition:       Optional[NutritionResult]
    nutrition_error: Optional[str]

class SuggestionRequest(BaseModel):
    calories_today:  float
    target_calories: float
    goal:            str
    protein_today:   float = 0
    target_protein:  float = 0

class SuggestionResponse(BaseModel):
    over_by:          float
    food_suggestion:  str
    food_calories:    int
    food_note:        str
    exercise:         str
    exercise_note:    str
    skipping_warning: str

class OCRMenuResponse(BaseModel):
    matched:     list
    raw_lines:   list
    unmatched:   list
    total_found: int

# ── Onboarding schemas ────────────────────────────────────────────────────────
class OnboardingStartRequest(BaseModel):
    session_id: str

class OnboardingReplyRequest(BaseModel):
    session_id: str
    message:    str

class OnboardingResponse(BaseModel):
    session_id:      str
    question:        Optional[str]
    phase:           str
    done:            bool
    plan:            Optional[dict]
    profile_so_far:  Optional[dict] = None

# ── Meal planning schemas ─────────────────────────────────────────────────────
class PlanStartRequest(BaseModel):
    session_id:   str
    goal_kcal:    int
    goal_protein: int
    goal_carbs:   int
    goal_fats:    int
    menu:         dict   # { breakfast:[{name,calories,...}], lunch:[...], snacks:[...], dinner:[...] }

class MealLoggedRequest(BaseModel):
    session_id: str
    meal_id:    str
    dish:       str
    calories:   float
    protein:    float
    carbs:      float
    fats:       float

class PlanChatRequest(BaseModel):
    session_id:    str
    message:       str
    logged_totals: Optional[dict] = None   # { cal, pro, car, fat } — current day totals

# ── Weekly review schema ──────────────────────────────────────────────────────
class WeeklyReviewRequest(BaseModel):
    log:  dict   # fitai_log_v2 from localStorage
    plan: dict   # fitai_plan from localStorage

class WeeklyReviewResponse(BaseModel):
    stats:    dict
    insights: list
    summary:  str

# ── RAG schemas ───────────────────────────────────────────────────────────────
class RAGRequest(BaseModel):
    question:   str
    today_menu: Optional[dict] = None
    user_log:   Optional[dict] = None
    user_goal:  Optional[str]  = None

class RAGResponse(BaseModel):
    question: str
    answer:   str
    sources:  list


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"message": "FitAI backend v2 is running 🍛"}

@app.get("/health")
def health():
    c = get_classifier()
    return {"status": "ok", "classes": c.classes, "device": str(c.device)}

@app.get("/classes")
def get_classes():
    c = get_classifier()
    return {"classes": c.classes, "count": len(c.classes)}


# ══════════════════════════════════════════════════════════════════════════════
# FOOD CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Image → dish classification + nutrition."""
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(400, f"Invalid file type: {file.content_type}")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, "Empty file.")

    classifier  = get_classifier()
    prediction  = classifier.predict(image_bytes, top_k=3)
    top_dish    = prediction["top_prediction"]["dish"]
    nutr        = get_nutrition_safe(top_dish)

    if nutr.get("error"):
        return PredictionResponse(**prediction, nutrition=None, nutrition_error=nutr["message"])
    return PredictionResponse(**prediction, nutrition=NutritionResult(**nutr), nutrition_error=None)


@app.post("/nutrition", response_model=NutritionResult)
def get_nutrition_manual(request: ManualEntryRequest):
    """Dish name → nutrition lookup (5-layer fallback)."""
    if not request.dish_name.strip():
        raise HTTPException(400, "Dish name cannot be empty.")
    result = get_nutrition_safe(request.dish_name)
    if result.get("error"):
        raise HTTPException(404, result["message"])
    return NutritionResult(**result)


# ══════════════════════════════════════════════════════════════════════════════
# OVER-TARGET SUGGESTION
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/suggest", response_model=SuggestionResponse)
def suggest_when_exceeded(request: SuggestionRequest):
    """Called when user is over calorie target — suggests a light food + exercise."""
    over_by = round(request.calories_today - request.target_calories, 1)
    if over_by <= 0:
        raise HTTPException(400, "Calories not exceeded yet.")

    hf_key = os.getenv("HF_API_KEY")
    if not hf_key:
        return _static_suggestion(over_by)

    prompt = (
        f"A college student has eaten {round(request.calories_today)} kcal today "
        f"against a target of {round(request.target_calories)} kcal "
        f"(goal: {request.goal} weight). They are {round(over_by)} kcal over.\n\n"
        f"Protein today: {round(request.protein_today)}g / {round(request.target_protein)}g target.\n\n"
        "Suggest ONE very light Indian college mess food item (under 120 kcal) "
        "AND one simple exercise. Do NOT suggest skipping meals.\n\n"
        'Return ONLY JSON with keys: food_suggestion, food_calories (int), food_note, '
        'exercise, exercise_note, skipping_warning.\n'
        "Return ONLY valid JSON."
    )

    import httpx
    headers = {"Authorization": f"Bearer {hf_key}", "Content-Type": "application/json"}
    models  = ["Qwen/Qwen2.5-72B-Instruct", "mistralai/Mixtral-8x7B-Instruct-v0.1",
               "meta-llama/Llama-3.2-3B-Instruct"]

    for model in models:
        try:
            resp = httpx.post(
                "https://router.huggingface.co/v1/chat/completions",
                json={"model": model, "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.3, "max_tokens": 350},
                headers=headers, timeout=30,
            )
            if resp.status_code not in (200, 201):
                continue
            raw  = resp.json()["choices"][0]["message"]["content"].strip()
            raw  = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw  = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
            m    = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                continue
            data = json.loads(m.group())
            return SuggestionResponse(
                over_by=over_by,
                food_suggestion=data.get("food_suggestion", "Cucumber raita"),
                food_calories=int(data.get("food_calories", 50)),
                food_note=data.get("food_note", "Light and healthy."),
                exercise=data.get("exercise", "Walk 2 km"),
                exercise_note=data.get("exercise_note", "Burns ~100 kcal."),
                skipping_warning=data.get("skipping_warning",
                    "Skipping meals slows metabolism — eat light instead."),
            )
        except Exception:
            continue

    return _static_suggestion(over_by)


def _static_suggestion(over_by: float) -> SuggestionResponse:
    return SuggestionResponse(
        over_by=over_by,
        food_suggestion="Plain chaas (buttermilk)",
        food_calories=30,
        food_note="Only ~30 kcal, keeps you full and aids digestion.",
        exercise=f"Walk {round(over_by / 60, 1)} km",
        exercise_note=f"Burns approximately {min(round(over_by * 0.6), 300)} kcal.",
        skipping_warning="Skipping meals is not recommended — eat light instead.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# OCR — MESS MENU BOARD
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/ocr-menu", response_model=OCRMenuResponse)
async def ocr_menu(file: UploadFile = File(...)):
    """Photo of mess menu board → list of matched dishes with nutrition."""
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(400, f"Invalid file type: {file.content_type}")
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, "Empty file.")
    try:
        from ocr import extract_menu_dishes
        return OCRMenuResponse(**extract_menu_dishes(image_bytes))
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, f"OCR failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ONBOARDING AGENT
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/onboarding/start", response_model=OnboardingResponse)
def onboarding_start_endpoint(request: OnboardingStartRequest):
    """
    Start a conversational onboarding session.
    Returns the first question to ask the user.
    """
    result = onboarding_start(request.session_id)
    return OnboardingResponse(**result)


@app.post("/onboarding/reply", response_model=OnboardingResponse)
def onboarding_reply_endpoint(request: OnboardingReplyRequest):
    """
    Send a user reply to the onboarding agent.
    Returns the next question, or if done=True, the computed nutrition plan.

    Frontend should:
    1. Call /onboarding/start → show question
    2. User replies → POST /onboarding/reply → show next question
    3. Repeat until done=True → save plan to localStorage as fitai_plan
    """
    result = onboarding_reply(request.session_id, request.message)
    return OnboardingResponse(**{k: v for k, v in result.items()
                                 if k in OnboardingResponse.model_fields})


@app.get("/onboarding/state/{session_id}")
def onboarding_state(session_id: str):
    """Get current onboarding session state (fields collected so far)."""
    state = _onboarding_sessions.get(session_id)
    if not state:
        raise HTTPException(404, "Onboarding session not found.")
    return {
        "session_id":    session_id,
        "phase":         state.get("phase"),
        "profile":       state.get("profile", {}),
        "done":          state.get("phase") == "done",
        "messages_count": len(state.get("messages", [])),
    }


@app.delete("/onboarding/{session_id}")
def onboarding_delete(session_id: str):
    """Clear an onboarding session."""
    onboarding_clear(session_id)
    return {"message": f"Onboarding session {session_id} cleared."}


# ══════════════════════════════════════════════════════════════════════════════
# MEAL PLANNING AGENT
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/plan/start")
def plan_start(request: PlanStartRequest):
    """
    Create a meal planning session and generate a full day plan.

    Pass today's mess menu and the user's macro targets.
    The agent picks dishes and portions to hit the targets across all 4 meal slots.
    """
    get_or_create_session(
        session_id=request.session_id,
        goal_kcal=request.goal_kcal,
        goal_protein=request.goal_protein,
        goal_carbs=request.goal_carbs,
        goal_fats=request.goal_fats,
        menu=request.menu,
    )
    return plan_day(request.session_id)


@app.post("/plan/log")
def plan_log_meal(request: MealLoggedRequest):
    """
    Log a meal that was actually eaten.
    Agent recalculates remaining macros and suggests the next meal adaptively.
    """
    just_logged = {
        "meal_id":  request.meal_id,
        "dish":     request.dish,
        "calories": request.calories,
        "protein":  request.protein,
        "carbs":    request.carbs,
        "fats":     request.fats,
    }
    return update_and_suggest(request.session_id, just_logged)


@app.post("/plan/chat")
def plan_chat(request: PlanChatRequest):
    """
    Freeform chat with the planning agent.
    User can say things like 'ate out for lunch', 'skipped breakfast', 'feeling full'.
    Agent adapts remaining suggestions accordingly.

    Pass logged_totals so the agent has current consumption context.
    """
    session = get_session_state(request.session_id)
    if session.get("error"):
        raise HTTPException(404, session["error"])

    # Update session eaten totals from frontend if provided
    if request.logged_totals:
        lt = request.logged_totals
        # Synthesise a pseudo eaten-item so remaining is computed correctly
        from agent import _plan_sessions
        s = _plan_sessions.get(request.session_id)
        if s:
            # Replace eaten with a single aggregate entry matching frontend totals
            s["eaten"] = [{
                "meal_id":  "combined",
                "dish":     "logged via frontend",
                "calories": lt.get("cal", 0),
                "protein":  lt.get("pro", 0),
                "carbs":    lt.get("car", 0),
                "fats":     lt.get("fat", 0),
            }]

    return update_and_suggest(request.session_id, user_message=request.message)


@app.get("/plan/state/{session_id}")
def plan_state(session_id: str):
    """Current session: goals, eaten, remaining, plan."""
    state = get_session_state(session_id)
    if state.get("error"):
        raise HTTPException(404, state["error"])
    return state


@app.delete("/plan/session/{session_id}")
def plan_clear(session_id: str):
    """Clear a planning session (end of day / user reset)."""
    clear_session(session_id)
    return {"message": f"Session {session_id} cleared."}


# ══════════════════════════════════════════════════════════════════════════════
# WEEKLY REVIEW
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/weekly-review", response_model=WeeklyReviewResponse)
def weekly_review(request: WeeklyReviewRequest):
    """
    Analyse the last 7 days of logs against the user's plan.

    Frontend sends:
    - log:  localStorage['fitai_log_v2']
    - plan: localStorage['fitai_plan']

    Returns:
    - stats:    per-day breakdown + 7-day averages
    - insights: list of specific observations (calorie gap, protein, consistency, etc.)
    - summary:  LLM-generated or rule-based narrative + one actionable tip

    Call every Sunday or whenever the user opens the History tab.
    """
    try:
        result = run_weekly_review(request.log, request.plan)
        return WeeklyReviewResponse(**result)
    except Exception as e:
        log.error("Weekly review failed: %s", e)
        raise HTTPException(500, f"Weekly review failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# RAG — NUTRITION Q&A
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/rag/ask", response_model=RAGResponse)
def rag_ask(request: RAGRequest):
    """
    Answer nutrition questions using ChromaDB + HF LLM.
    Context: nutrition_db + today's menu + user's log history.
    """
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    try:
        from rag import answer_question
        result = answer_question(
            question=request.question,
            today_menu=request.today_menu,
            user_log=request.user_log,
            user_goal=request.user_goal,
        )
        return RAGResponse(**result)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, f"RAG failed: {e}")


@app.post("/rag/populate")
def rag_populate():
    """Pre-populate ChromaDB. Call once at setup."""
    try:
        from rag import populate_db
        count = populate_db(force=False)
        return {"status": "ok", "docs_in_db": count}
    except RuntimeError as e:
        raise HTTPException(503, str(e))