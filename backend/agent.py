"""
FitAI - agent.py
================
Three agentic workflows:

1. ONBOARDING AGENT  (LangGraph StateGraph)
   Conversational profile setup — collects name, age, gender, height, weight,
   goal, diet, activities, gym details, sleep. Calculates personalised
   calorie / protein targets with gym-day vs rest-day split.

2. MEAL PLANNING AGENT  (direct HF calls, state stored in memory)
   Generates a full-day plan from today's mess menu + goals.
   Adapts suggestions after each logged meal.

3. WEEKLY REVIEW AGENT  (LangGraph StateGraph)
   Sunday summary — analyses 7-day log vs targets, spots patterns,
   gives personalised nudges for the week ahead.

LangGraph install:
    pip install langgraph langchain langchain-community --break-system-packages
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from typing import Any, Optional, TypedDict

log = logging.getLogger("fitai.agent")

# ── LangGraph (optional) ──────────────────────────────────────────────────────
try:
    from langgraph.graph import END, StateGraph
    _LANGGRAPH = True
except ImportError:
    _LANGGRAPH = False
    log.warning("langgraph not installed — onboarding + weekly review use fallback mode")

# ── Nutrition lookup ──────────────────────────────────────────────────────────
from nutrition import get_nutrition_safe  # noqa


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HF UTILITY
# ══════════════════════════════════════════════════════════════════════════════

_HF_MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Llama-3.2-3B-Instruct",
]


def _hf_call(prompt: str, max_tokens: int = 600, system: str = "") -> Optional[str]:
    """Call HuggingFace Inference API, try models in order."""
    import httpx

    hf_key = os.getenv("HF_API_KEY")
    if not hf_key:
        return None

    headers = {"Authorization": f"Bearer {hf_key}", "Content-Type": "application/json"}
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for model in _HF_MODELS:
        try:
            resp = httpx.post(
                "https://router.huggingface.co/v1/chat/completions",
                json={"model": model, "messages": messages,
                      "temperature": 0.3, "max_tokens": max_tokens},
                headers=headers,
                timeout=45,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            log.warning("HF model %s failed: %s", model, exc)
    return None


def _parse_json(raw: str) -> Optional[dict]:
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ① ONBOARDING AGENT
# ══════════════════════════════════════════════════════════════════════════════

class OnboardingState(TypedDict):
    """Mutable state passed between LangGraph nodes."""
    messages: list[dict]          # [{role, content}]
    profile:  dict                # collected fields
    phase:    str                 # "you" | "goals" | "activities" | "plan" | "done"
    next_question: str            # the question the agent will ask
    plan:     Optional[dict]      # computed plan (set in final node)
    error:    Optional[str]


# Ordered fields to collect and the questions to ask
_ONBOARDING_FLOW = [
    # (field, question, phase)
    ("name",           "Hey! I'm FitAI 👋 What's your name?",                         "you"),
    ("age",            "Nice to meet you, {name}! How old are you?",                   "you"),
    ("gender",         "Got it. What's your gender? (male / female / other)",           "you"),
    ("height",         "What's your height in cm?",                                    "you"),
    ("weight",         "And your current weight in kg?",                               "you"),
    ("goal",           "What's your main goal right now?\n\n• lose — lose weight\n• gain — build muscle\n• maintain — stay at current weight", "goals"),
    ("target_weight",  "What's your target weight in kg?",                             "goals"),   # skipped for maintain
    ("duration",       "How many weeks do you want to reach that target?",             "goals"),   # skipped for maintain
    ("diet",           "Any dietary preference?\n\n• veg\n• non_veg\n• egg",          "goals"),
    ("eats_in_mess",   "Do you eat at the college mess?\n\n• yes\n• no\n• mixed (sometimes)",  "goals"),
    ("sleep",          "How many hours of sleep do you usually get per night?",        "goals"),
    ("activities",     "What activities do you do? (pick all that apply)\n\n• gym  • swimming  • running  • cycling  • yoga  • walking  • sport  • none", "activities"),
    ("gym_days",       "How many days a week do you go to the gym? (0–7)",             "activities"),  # skipped if no gym
    ("gym_type",       "What kind of training?\n\n• strength  • cardio  • mixed",     "activities"),  # skipped if no gym
    ("sport_name",     "Which sport?",                                                  "activities"),  # skipped if no sport
]

# Fields to skip under certain conditions
def _should_skip(field: str, profile: dict) -> bool:
    goal = profile.get("goal", "")
    acts = profile.get("activities", [])
    if field in ("target_weight", "duration") and goal == "maintain":
        return True
    if field in ("gym_days", "gym_type") and "gym" not in acts:
        return True
    if field == "sport_name" and "sport" not in acts:
        return True
    return False


def _compute_plan(profile: dict) -> dict:
    """Calculate calorie/macro targets from profile."""
    age    = int(profile.get("age", 21))
    gender = str(profile.get("gender", "male")).lower()
    height = float(profile.get("height", 170))
    weight = float(profile.get("weight", 70))
    goal   = str(profile.get("goal", "maintain"))
    acts   = profile.get("activities", [])
    if isinstance(acts, str):
        acts = [a.strip() for a in acts.replace(",", " ").split()]
    gym_days   = int(profile.get("gym_days", 0))
    sleep_hrs  = float(profile.get("sleep", 7))
    target_wt  = float(profile.get("target_weight", weight))
    duration_w = int(profile.get("duration", 12))

    # BMR (Mifflin-St Jeor)
    if gender in ("male", "m"):
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    bmr = round(bmr)

    # Activity multiplier
    if gym_days >= 5 or "swimming" in acts or "running" in acts:
        mult = 1.55
    elif gym_days >= 3 or len([a for a in acts if a != "none"]) >= 2:
        mult = 1.375
    else:
        mult = 1.2
    tdee = round(bmr * mult)

    # Goal adjustment
    if goal == "lose":
        base_cal = max(tdee - 500, 1200)
    elif goal == "gain":
        base_cal = tdee + 300
    else:
        base_cal = tdee

    # Sleep penalty
    if sleep_hrs < 6:
        base_cal -= 50
    elif sleep_hrs < 7:
        base_cal -= 25
    base_cal = max(base_cal, 1200)

    # Gym-day vs rest-day split
    gym_cal  = base_cal + 200 if gym_days > 0 else base_cal
    rest_cal = max(base_cal - 100, 1200) if gym_days > 0 else base_cal

    # Protein (per kg) — evidence-based ranges (ISSN 2023)
    # gain: 1.8 g/kg, lose: 1.6 g/kg (muscle retention), maintain: 1.4 g/kg
    pro_per_kg = {"gain": 1.8, "lose": 1.6}.get(goal, 1.4)
    if "gym" in acts and goal == "maintain":
        pro_per_kg = 1.6
    protein = min(round(weight * pro_per_kg), 160)  # hard cap at 160g

    fats  = round((base_cal * 0.25) / 9)
    carbs = round((base_cal - protein * 4 - fats * 9) / 4)

    bmi = round(weight / ((height / 100) ** 2), 1)

    return {
        "calories":         base_cal,
        "gymDayCalories":   gym_cal,
        "restDayCalories":  rest_cal,
        "protein":          protein,
        "carbs":            carbs,
        "fats":             fats,
        "bmr":              bmr,
        "tdee":             tdee,
        "bmi":              str(bmi),
        "gym_days_per_week": gym_days,
        "gym_type":         profile.get("gym_type", ""),
        "activities":       acts,
        "goal":             goal,
        "diet":             profile.get("diet", "non_veg"),
        "eats_in_mess":     profile.get("eats_in_mess", "yes"),
        "sleep_hours":      sleep_hrs,
        "name":             profile.get("name", ""),
        "age":              age,
        "gender":           gender,
        "height":           height,
        "weight":           weight,
        "targetWeight":     target_wt,
        "duration":         duration_w,
    }


# ── LangGraph nodes ───────────────────────────────────────────────────────────

def _node_ask(state: OnboardingState) -> OnboardingState:
    """Find the next unanswered field and formulate the question."""
    profile = state["profile"]
    for field, question, phase in _ONBOARDING_FLOW:
        if field not in profile and not _should_skip(field, profile):
            q = question.format(**{k: v for k, v in profile.items() if isinstance(v, str)})
            return {**state, "next_question": q, "phase": phase}
    # All fields collected → move to plan
    return {**state, "next_question": "", "phase": "plan"}


def _node_parse(state: OnboardingState) -> OnboardingState:
    """
    Parse the latest user message into the correct profile field.
    Uses simple rule-based parsing — no LLM needed.
    """
    if not state["messages"]:
        return state

    last_user = next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"),
        ""
    ).strip()

    profile = dict(state["profile"])

    # Find which field we're collecting right now
    for field, _, _ in _ONBOARDING_FLOW:
        if field not in profile and not _should_skip(field, profile):
            # Parse value
            v = _parse_field(field, last_user)
            if v is not None:
                profile[field] = v
            break

    return {**state, "profile": profile}


def _parse_field(field: str, raw: str) -> Any:
    """Rule-based parser for each profile field."""
    raw = raw.strip().lower()

    if field == "name":
        return raw.title()

    if field in ("age", "height", "weight", "gym_days"):
        m = re.search(r"\d+\.?\d*", raw)
        if m:
            return int(float(m.group())) if field != "weight" else float(m.group())

    if field == "sleep":
        m = re.search(r"\d+\.?\d*", raw)
        return float(m.group()) if m else 7.0

    if field == "target_weight":
        m = re.search(r"\d+\.?\d*", raw)
        return float(m.group()) if m else None

    if field == "duration":
        m = re.search(r"\d+", raw)
        return int(m.group()) if m else 12

    if field == "gender":
        if any(w in raw for w in ("female", "f", "woman", "girl")):
            return "female"
        if any(w in raw for w in ("other", "non")):
            return "other"
        return "male"

    if field == "goal":
        if any(w in raw for w in ("lose", "loss", "cut", "diet", "slim")):
            return "lose"
        if any(w in raw for w in ("gain", "build", "bulk", "muscle", "grow")):
            return "gain"
        return "maintain"

    if field == "diet":
        if "veg" in raw and "non" not in raw and "egg" not in raw:
            return "veg"
        if "egg" in raw:
            return "egg"
        return "non_veg"

    if field == "eats_in_mess":
        if "yes" in raw or raw in ("y", "yeah", "yep", "always"):
            return True
        if "mix" in raw or "some" in raw or "sometimes" in raw:
            return "mixed"
        return False

    if field == "activities":
        options = ["gym", "swimming", "running", "cycling", "yoga", "walking", "sport", "none"]
        found = [o for o in options if o in raw]
        return found if found else ["none"]

    if field == "gym_type":
        if "strength" in raw or "weight" in raw or "lift" in raw:
            return "strength"
        if "cardio" in raw or "cardio" in raw:
            return "cardio"
        return "mixed"

    if field == "sport_name":
        return raw.title() if raw else "Sport"

    return raw  # fallback — return as-is


def _node_compute_plan(state: OnboardingState) -> OnboardingState:
    """All fields collected — compute the nutrition plan."""
    plan = _compute_plan(state["profile"])
    return {**state, "plan": plan, "phase": "done"}


def _should_compute(state: OnboardingState) -> str:
    """Router: all fields collected → compute, else ask next question."""
    profile = state["profile"]
    for field, _, _ in _ONBOARDING_FLOW:
        if field not in profile and not _should_skip(field, profile):
            return "ask"
    return "compute"


def build_onboarding_graph():
    """Build and compile the LangGraph onboarding state machine."""
    if not _LANGGRAPH:
        return None

    g = StateGraph(OnboardingState)
    g.add_node("parse",   _node_parse)
    g.add_node("ask",     _node_ask)
    g.add_node("compute", _node_compute_plan)

    g.set_entry_point("parse")
    g.add_conditional_edges("parse", _should_compute, {"ask": "ask", "compute": "compute"})
    g.add_edge("ask",     END)
    g.add_edge("compute", END)

    return g.compile()


# Singleton graph
_onboarding_graph = None

def get_onboarding_graph():
    global _onboarding_graph
    if _onboarding_graph is None:
        _onboarding_graph = build_onboarding_graph()
    return _onboarding_graph


# ── In-memory onboarding sessions ────────────────────────────────────────────
_onboarding_sessions: dict[str, OnboardingState] = {}


def onboarding_start(session_id: str) -> dict:
    """Create a fresh onboarding session, return first question."""
    state: OnboardingState = {
        "messages":     [],
        "profile":      {},
        "phase":        "you",
        "next_question": "",
        "plan":         None,
        "error":        None,
    }
    # Ask first question immediately
    state = _node_ask(state)
    _onboarding_sessions[session_id] = state
    return {
        "session_id":    session_id,
        "question":      state["next_question"],
        "phase":         state["phase"],
        "done":          False,
        "plan":          None,
    }


def onboarding_reply(session_id: str, user_message: str) -> dict:
    """
    User sent a reply — advance the state machine one step.
    Returns next question or, if done, the computed plan.
    """
    state = _onboarding_sessions.get(session_id)
    if not state:
        # Auto-create
        state = onboarding_start(session_id)
        state = _onboarding_sessions[session_id]

    # Append user message
    state["messages"].append({"role": "user", "content": user_message})

    graph = get_onboarding_graph()
    if graph:
        # Run LangGraph
        result = graph.invoke(state)
        new_state: OnboardingState = result
    else:
        # Fallback: run nodes manually
        new_state = _node_parse(state)
        if _should_compute(new_state) == "compute":
            new_state = _node_compute_plan(new_state)
        else:
            new_state = _node_ask(new_state)

    _onboarding_sessions[session_id] = new_state

    if new_state.get("phase") == "done" and new_state.get("plan"):
        return {
            "session_id": session_id,
            "question":   None,
            "phase":      "done",
            "done":       True,
            "plan":       new_state["plan"],
            "profile":    new_state["profile"],
        }

    return {
        "session_id": session_id,
        "question":   new_state.get("next_question", ""),
        "phase":      new_state.get("phase", "you"),
        "done":       False,
        "plan":       None,
        "profile_so_far": {k: v for k, v in new_state["profile"].items()
                           if k in ("name", "goal", "activities")},
    }


def onboarding_clear(session_id: str) -> None:
    _onboarding_sessions.pop(session_id, None)


# ══════════════════════════════════════════════════════════════════════════════
# ② MEAL PLANNING AGENT  (unchanged from original, extended)
# ══════════════════════════════════════════════════════════════════════════════

_plan_sessions: dict[str, dict] = {}


def get_or_create_session(session_id: str, goal_kcal: int, goal_protein: int,
                           goal_carbs: int, goal_fats: int, menu: dict) -> dict:
    if session_id not in _plan_sessions:
        _plan_sessions[session_id] = {
            "session_id":   session_id,
            "goal_kcal":    goal_kcal,
            "goal_protein": goal_protein,
            "goal_carbs":   goal_carbs,
            "goal_fats":    goal_fats,
            "menu":         menu,
            "eaten":        [],
            "plan":         None,
            "messages":     [],
        }
    return _plan_sessions[session_id]


def _get_menu_text(menu: dict) -> str:
    lines = []
    for meal_id, dishes in menu.items():
        if dishes:
            names = ", ".join(d["name"].replace("_", " ") for d in dishes if d.get("name"))
            if names:
                lines.append(f"{meal_id.capitalize()}: {names}")
    return "\n".join(lines) if lines else "No menu available."


def _compute_remaining(session: dict) -> dict:
    eaten_kcal = sum(i.get("calories", 0) for i in session["eaten"])
    eaten_pro  = sum(i.get("protein",  0) for i in session["eaten"])
    eaten_car  = sum(i.get("carbs",    0) for i in session["eaten"])
    eaten_fat  = sum(i.get("fats",     0) for i in session["eaten"])
    return {
        "eaten_kcal":       round(eaten_kcal, 1),
        "eaten_protein":    round(eaten_pro,  1),
        "eaten_carbs":      round(eaten_car,  1),
        "eaten_fats":       round(eaten_fat,  1),
        "remaining_kcal":   round(session["goal_kcal"]    - eaten_kcal, 1),
        "remaining_protein": round(session["goal_protein"] - eaten_pro,  1),
        "remaining_carbs":  round(session["goal_carbs"]   - eaten_car,  1),
        "remaining_fats":   round(session["goal_fats"]    - eaten_fat,  1),
        "goal_kcal":        session["goal_kcal"],
        "goal_protein":     session["goal_protein"],
        "goal_carbs":       session["goal_carbs"],
        "goal_fats":        session["goal_fats"],
    }


def plan_day(session_id: str) -> dict:
    session = _plan_sessions.get(session_id)
    if not session:
        return {"error": "Session not found. Call /plan/start first."}

    menu_text = _get_menu_text(session["menu"])
    prompt = f"""You are a nutrition-aware meal planner for a college student eating at a mess.

Student's daily targets:
- Calories: {session['goal_kcal']} kcal
- Protein: {session['goal_protein']}g
- Carbs: {session['goal_carbs']}g
- Fats: {session['goal_fats']}g

Today's mess menu:
{menu_text}

Plan breakfast, lunch, snacks, and dinner using ONLY items from the menu.
Specify realistic portions (e.g. "2 rotis", "1 bowl dal").
Hit calorie and protein targets as closely as possible.
Distribute roughly: Breakfast 25%, Lunch 35%, Snacks 10%, Dinner 30%.

Return ONLY a JSON object:
{{
  "breakfast": [{{"dish": str, "portions": str, "estimated_kcal": int, "estimated_protein": int}}],
  "lunch":     [{{"dish": str, "portions": str, "estimated_kcal": int, "estimated_protein": int}}],
  "snacks":    [{{"dish": str, "portions": str, "estimated_kcal": int, "estimated_protein": int}}],
  "dinner":    [{{"dish": str, "portions": str, "estimated_kcal": int, "estimated_protein": int}}],
  "total_estimated_kcal": int,
  "total_estimated_protein": int,
  "summary": str
}}

summary: 2-3 sentences on how the plan meets the goals.
Return ONLY valid JSON."""

    raw  = _hf_call(prompt, max_tokens=900)
    if not raw:
        return {"error": "Could not reach AI backend.", "summary": "AI unavailable."}

    data = _parse_json(raw)
    if not data:
        return {"error": "AI returned malformed response.", "raw": raw}

    session["plan"] = data
    log.info("Session %s: plan generated (%s kcal)", session_id, data.get("total_estimated_kcal"))
    return data


def update_and_suggest(session_id: str, just_logged: Optional[dict] = None,
                       user_message: Optional[str] = None) -> dict:
    """
    Called after each meal logged OR when user sends a chat message.
    Recalculates remaining macros → suggests next meal adaptively.
    """
    session = _plan_sessions.get(session_id)
    if not session:
        return {"error": "Session not found."}

    if just_logged:
        session["eaten"].append(just_logged)

    remaining  = _compute_remaining(session)
    menu_text  = _get_menu_text(session["menu"])

    eaten_meal_ids = [i.get("meal_id") for i in session["eaten"]]
    meals_order    = ["breakfast", "lunch", "snacks", "dinner"]
    next_meal      = next((m for m in meals_order if m not in eaten_meal_ids), None)

    if next_meal is None:
        return {
            "remaining":       remaining,
            "eaten_so_far":    session["eaten"],
            "next_suggestion": None,
            "message":         f"All meals logged! You had {remaining['eaten_kcal']} / {remaining['goal_kcal']} kcal today.",
        }

    context = ""
    if user_message:
        context = f"\nUser note: \"{user_message}\"\nTake this into account when suggesting.\n"

    prompt = f"""You are a nutrition coach for a college student.
Eaten today: {remaining['eaten_kcal']} kcal.

Remaining targets:
- Calories: {remaining['remaining_kcal']} kcal
- Protein:  {remaining['remaining_protein']}g
- Carbs:    {remaining['remaining_carbs']}g
- Fats:     {remaining['remaining_fats']}g
{context}
Today's mess menu:
{menu_text}

Suggest what to eat for {next_meal}. Use only menu items. Be specific about portions.

Return ONLY a JSON object:
{{
  "meal": "{next_meal}",
  "dishes": [{{"dish": str, "portions": str, "estimated_kcal": int, "estimated_protein": int}}],
  "total_kcal": int,
  "total_protein": int,
  "note": str
}}

note: 1-2 sentences on why this fits the remaining targets.
Return ONLY valid JSON."""

    raw        = _hf_call(prompt, max_tokens=500)
    suggestion = _parse_json(raw) if raw else None

    return {
        "remaining":       remaining,
        "eaten_so_far":    session["eaten"],
        "next_suggestion": suggestion,
        "next_meal":       next_meal,
        "reply":           suggestion.get("note") if suggestion else "Got it! Here's your next suggestion.",
    }


def get_session_state(session_id: str) -> dict:
    session = _plan_sessions.get(session_id)
    if not session:
        return {"error": "Session not found."}
    return {
        "session_id": session_id,
        "goals":      {"kcal": session["goal_kcal"], "protein": session["goal_protein"],
                       "carbs": session["goal_carbs"], "fats": session["goal_fats"]},
        "remaining":  _compute_remaining(session),
        "eaten":      session["eaten"],
        "plan":       session.get("plan"),
        "messages":   session.get("messages", []),
    }


def clear_session(session_id: str) -> None:
    _plan_sessions.pop(session_id, None)


# ══════════════════════════════════════════════════════════════════════════════
# ③ WEEKLY REVIEW AGENT  (LangGraph)
# ══════════════════════════════════════════════════════════════════════════════

class WeeklyReviewState(TypedDict):
    log:      dict          # { "YYYY-MM-DD": { breakfast:[...], lunch:[...], ... } }
    plan:     dict          # fitai_plan object
    stats:    Optional[dict]  # computed stats
    insights: Optional[list]  # list of insight strings
    summary:  Optional[str]
    error:    Optional[str]


def _compute_weekly_stats(log: dict, plan: dict) -> dict:
    """Compute per-day and aggregate stats from the log."""
    from datetime import date, timedelta

    today = date.today()
    days  = [(today - timedelta(days=i)).isoformat() for i in range(6, -1, -1)]

    PORTION_MUL = {"S": 0.7, "M": 1.0, "L": 1.4}
    MEALS = ["breakfast", "lunch", "snacks", "dinner"]

    per_day = {}
    for dk in days:
        day_log = log.get(dk, {})
        cal = pro = car = fat = 0
        logged_meals = 0
        skipped_meals = 0
        for meal_id in MEALS:
            items = day_log.get(meal_id, [])
            for item in items:
                if item.get("skipped"):
                    skipped_meals += 1
                    continue
                mul = PORTION_MUL.get(item.get("portion_size", "M"), 1.0)
                cal += (item.get("base_calories") or item.get("calories", 0)) * mul
                pro += (item.get("base_protein")  or item.get("protein",  0)) * mul
                car += (item.get("base_carbs")    or item.get("carbs",    0)) * mul
                fat += (item.get("base_fats")     or item.get("fats",     0)) * mul
                logged_meals += 1
        per_day[dk] = {
            "cal": round(cal), "pro": round(pro),
            "car": round(car), "fat": round(fat),
            "logged_meals": logged_meals,
            "skipped_meals": skipped_meals,
            "has_data": cal > 0,
        }

    tracked_days = [d for d in per_day.values() if d["has_data"]]
    n = len(tracked_days)
    if n == 0:
        return {"per_day": per_day, "avg": {}, "n_tracked": 0}

    avg_cal = round(sum(d["cal"] for d in tracked_days) / n)
    avg_pro = round(sum(d["pro"] for d in tracked_days) / n)
    target  = plan.get("calories", 2000)
    pro_tgt = plan.get("protein", 120)

    cal_hit  = sum(1 for d in tracked_days if abs(d["cal"] - target) < target * 0.1)
    pro_hit  = sum(1 for d in tracked_days if d["pro"] >= pro_tgt * 0.9)

    return {
        "per_day":     per_day,
        "n_tracked":   n,
        "avg_cal":     avg_cal,
        "avg_pro":     avg_pro,
        "cal_target":  target,
        "pro_target":  pro_tgt,
        "cal_hit_days": cal_hit,
        "pro_hit_days": pro_hit,
        "cal_hit_pct":  round(cal_hit / n * 100),
        "pro_hit_pct":  round(pro_hit / n * 100),
        "days":         days,
    }


def _node_weekly_stats(state: WeeklyReviewState) -> WeeklyReviewState:
    stats = _compute_weekly_stats(state["log"], state["plan"])
    return {**state, "stats": stats}


def _node_weekly_insights(state: WeeklyReviewState) -> WeeklyReviewState:
    """Generate rule-based insights — no LLM needed for basic analysis."""
    stats   = state["stats"]
    plan    = state["plan"]
    insights = []

    if stats["n_tracked"] == 0:
        return {**state, "insights": ["No meals were logged this week. Start logging from tomorrow!"], "summary": "No data this week."}

    avg_cal = stats["avg_cal"]
    cal_tgt = stats["cal_target"]
    avg_pro = stats["avg_pro"]
    pro_tgt = stats["pro_target"]

    # Calorie insights
    cal_diff = avg_cal - cal_tgt
    if abs(cal_diff) < cal_tgt * 0.05:
        insights.append(f"✅ Calories on point — averaging {avg_cal} kcal vs {cal_tgt} kcal target.")
    elif cal_diff > 0:
        insights.append(f"⚠️ Running {abs(cal_diff)} kcal over daily target on average. Try smaller portions of rice or reduce fried snacks.")
    else:
        insights.append(f"📉 Averaging {abs(cal_diff)} kcal below target. You may not be fuelling your body enough — add a roti or a glass of milk.")

    # Protein insights
    pro_gap = pro_tgt - avg_pro
    if pro_gap <= 0:
        insights.append(f"💪 Protein target met — averaging {avg_pro}g vs {pro_tgt}g goal. Great work!")
    elif pro_gap <= 15:
        insights.append(f"💪 Protein close — {avg_pro}g / {pro_tgt}g. Adding an egg or a cup of curd daily will close the gap.")
    else:
        insights.append(f"🥚 Protein gap: {avg_pro}g vs {pro_tgt}g target. Include dal, paneer, eggs or curd at every meal.")

    # Consistency
    hit_pct = stats["cal_hit_pct"]
    if hit_pct >= 70:
        insights.append(f"🎯 Hit calorie target {stats['cal_hit_days']}/{stats['n_tracked']} tracked days — very consistent!")
    elif hit_pct >= 40:
        insights.append(f"📊 Hit target {stats['cal_hit_days']}/{stats['n_tracked']} days — room to improve consistency.")
    else:
        insights.append(f"📊 Only on-target {stats['cal_hit_days']}/{stats['n_tracked']} days. Try pre-logging meals the night before.")

    # Skip pattern
    all_skipped = sum(d["skipped_meals"] for d in stats["per_day"].values())
    if all_skipped >= 4:
        insights.append(f"⚠️ Skipped {all_skipped} meal slots this week. Skipping meals can slow metabolism — eat light instead of skipping.")

    # Tracking gaps
    untracked = 7 - stats["n_tracked"]
    if untracked >= 3:
        insights.append(f"📝 Only {stats['n_tracked']}/7 days tracked. More data = better suggestions — try logging even if you ate out.")

    return {**state, "insights": insights}


def _node_weekly_summary(state: WeeklyReviewState) -> WeeklyReviewState:
    """Optional LLM-generated narrative summary. Falls back to rule-based if unavailable."""
    stats    = state["stats"]
    insights = state["insights"]
    plan     = state["plan"]

    if stats["n_tracked"] == 0:
        return {**state, "summary": "No meals logged this week."}

    # Try LLM for a personalised narrative
    prompt = f"""You are a friendly nutrition coach reviewing a college student's week.

Their goal: {plan.get('goal', 'maintain')} weight.
Daily targets: {stats['cal_target']} kcal, {stats['pro_target']}g protein.
This week's averages: {stats['avg_cal']} kcal/day, {stats['avg_pro']}g protein/day.
Days tracked: {stats['n_tracked']}/7.
Calorie target hit: {stats['cal_hit_pct']}% of days.
Protein target hit: {stats['pro_hit_pct']}% of days.

Key insights:
{chr(10).join('- ' + i for i in insights)}

Write a warm, encouraging 3-4 sentence weekly summary. Acknowledge what went well,
identify the biggest gap, give ONE specific actionable tip for next week.
Be concise and direct — no bullet points, just prose."""

    raw = _hf_call(prompt, max_tokens=250)
    summary = raw if raw else " ".join(insights[:2])
    return {**state, "summary": summary}


def build_weekly_review_graph():
    if not _LANGGRAPH:
        return None
    g = StateGraph(WeeklyReviewState)
    g.add_node("stats",    _node_weekly_stats)
    g.add_node("insights", _node_weekly_insights)
    g.add_node("summary",  _node_weekly_summary)
    g.set_entry_point("stats")
    g.add_edge("stats",    "insights")
    g.add_edge("insights", "summary")
    g.add_edge("summary",  END)
    return g.compile()


_weekly_graph = None

def get_weekly_graph():
    global _weekly_graph
    if _weekly_graph is None:
        _weekly_graph = build_weekly_review_graph()
    return _weekly_graph


def run_weekly_review(log: dict, plan: dict) -> dict:
    """
    Entry point for weekly review.
    Returns { stats, insights, summary }.
    """
    initial: WeeklyReviewState = {
        "log":      log,
        "plan":     plan,
        "stats":    None,
        "insights": None,
        "summary":  None,
        "error":    None,
    }

    graph = get_weekly_graph()
    if graph:
        result = graph.invoke(initial)
    else:
        # Manual fallback
        result = _node_weekly_stats(initial)
        result = _node_weekly_insights(result)
        result = _node_weekly_summary(result)

    return {
        "stats":    result.get("stats", {}),
        "insights": result.get("insights", []),
        "summary":  result.get("summary", ""),
    }