"""
FitAI - agent.py
================
Agentic Meal Planning using LangChain ReAct pattern.

The agent:
  1. Observes today's mess menu + user's macro goals
  2. Plans breakfast/lunch/dinner with portions to hit targets
  3. After each meal logged, recalculates remaining macros
  4. Suggests the next meal adaptively

LangChain tools:
  - get_today_menu        : fetch today's mess menu dishes + nutrition
  - get_nutrition         : look up nutrition for any dish
  - calculate_remaining   : compute remaining macros given eaten so far
  - suggest_next_meal     : given remaining targets + available menu, suggest what to eat

Install:
    pip install langchain langchain-community --break-system-packages
    (Uses HuggingFace Inference API — no new keys needed, reuses HF_API_KEY)
"""

import os
import json
import logging
import re
from typing import Optional

log = logging.getLogger("fitai.agent")

# ── LangChain imports (optional — agent uses direct HF calls instead) ─────────
try:
    from langchain.agents import AgentExecutor, create_react_agent  # noqa: F401
    from langchain.tools import tool                                  # noqa: F401
    from langchain_community.llms import HuggingFaceEndpoint          # noqa: F401
    from langchain.prompts import PromptTemplate                      # noqa: F401
    from langchain.memory import ConversationBufferMemory              # noqa: F401
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    # LangChain is optional — the agent uses direct HF API calls which work fine without it.

# ── Nutrition lookup (reuse existing module) ──────────────────────────────────
from nutrition import get_nutrition_safe

# ── Agent state (per-session, stored in memory) ───────────────────────────────
# In production you'd use Redis or a DB; for MVP in-memory is fine.
# Sessions are keyed by session_id passed from frontend.
_sessions: dict[str, dict] = {}


def get_or_create_session(session_id: str, goal_kcal: int, goal_protein: int,
                           goal_carbs: int, goal_fats: int, menu: dict) -> dict:
    """Get existing session or create a fresh one."""
    if session_id not in _sessions:
        _sessions[session_id] = {
            "session_id"   : session_id,
            "goal_kcal"    : goal_kcal,
            "goal_protein" : goal_protein,
            "goal_carbs"   : goal_carbs,
            "goal_fats"    : goal_fats,
            "menu"         : menu,       # { breakfast: [...], lunch: [...], dinner: [...] }
            "eaten"        : [],         # list of { meal_id, dish, calories, protein, carbs, fats }
            "plan"         : None,       # agent's current plan
            "messages"     : [],         # conversation history for display
        }
    return _sessions[session_id]


def update_session(session_id: str, eaten_item: dict) -> dict:
    """Add a logged meal to session state."""
    if session_id in _sessions:
        _sessions[session_id]["eaten"].append(eaten_item)
    return _sessions.get(session_id, {})


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS (used by agent AND fallback mode)
# ══════════════════════════════════════════════════════════════════════════════
def _get_menu_text(menu: dict) -> str:
    """Format menu dict into readable text for the LLM."""
    lines = []
    for meal_id, dishes in menu.items():
        if dishes:
            names = ", ".join(d["name"] for d in dishes if d.get("name"))
            lines.append(f"{meal_id.capitalize()}: {names}")
    return "\n".join(lines) if lines else "No menu available."


def _compute_remaining(session: dict) -> dict:
    """Compute remaining macro targets after what's been eaten."""
    eaten_kcal = sum(i.get("calories", 0) for i in session["eaten"])
    eaten_pro  = sum(i.get("protein",  0) for i in session["eaten"])
    eaten_car  = sum(i.get("carbs",    0) for i in session["eaten"])
    eaten_fat  = sum(i.get("fats",     0) for i in session["eaten"])

    return {
        "eaten_kcal"    : round(eaten_kcal, 1),
        "eaten_protein" : round(eaten_pro,  1),
        "eaten_carbs"   : round(eaten_car,  1),
        "eaten_fats"    : round(eaten_fat,  1),
        "remaining_kcal"    : round(session["goal_kcal"]    - eaten_kcal, 1),
        "remaining_protein" : round(session["goal_protein"] - eaten_pro,  1),
        "remaining_carbs"   : round(session["goal_carbs"]   - eaten_car,  1),
        "remaining_fats"    : round(session["goal_fats"]    - eaten_fat,  1),
        "goal_kcal"    : session["goal_kcal"],
        "goal_protein" : session["goal_protein"],
        "goal_carbs"   : session["goal_carbs"],
        "goal_fats"    : session["goal_fats"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK MODE — direct HF call when LangChain not available
# ══════════════════════════════════════════════════════════════════════════════
def _hf_call(prompt: str, max_tokens: int = 600) -> Optional[str]:
    """Direct HuggingFace call — used in fallback mode."""
    import httpx
    hf_key = os.getenv("HF_API_KEY")
    if not hf_key:
        return None

    models = [
        "Qwen/Qwen2.5-72B-Instruct",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "meta-llama/Llama-3.2-3B-Instruct",
    ]
    headers = {"Authorization": f"Bearer {hf_key}", "Content-Type": "application/json"}

    for model in models:
        try:
            resp = httpx.post(
                "https://router.huggingface.co/v1/chat/completions",
                json={
                    "model"      : model,
                    "messages"   : [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens" : max_tokens,
                },
                headers=headers,
                timeout=45,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            log.warning(f"HF model {model} failed: {e}")
            continue
    return None


def _parse_json_response(raw: str) -> Optional[dict]:
    """Extract and parse JSON from LLM response."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
    m   = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# CORE AGENT FUNCTIONS — called by API endpoints
# ══════════════════════════════════════════════════════════════════════════════
def plan_day(session_id: str) -> dict:
    """
    Step 1: Agent looks at today's menu + user goals → creates a full day plan.

    Returns:
        {
            "plan": {
                "breakfast": [ { dish, portions, kcal, protein } ],
                "lunch":     [ ... ],
                "dinner":    [ ... ],
            },
            "summary": str,           # agent's reasoning / summary
            "total_kcal": float,
            "total_protein": float,
        }
    """
    session = _sessions.get(session_id)
    if not session:
        return {"error": "Session not found. Call /plan/start first."}

    menu_text = _get_menu_text(session["menu"])

    prompt = f"""You are a nutrition-aware meal planner for a college student eating at a mess (cafeteria).

Student's daily targets:
- Calories: {session['goal_kcal']} kcal
- Protein: {session['goal_protein']}g
- Carbs: {session['goal_carbs']}g
- Fats: {session['goal_fats']}g

Today's mess menu:
{menu_text}

Task: Plan breakfast, lunch, and dinner using ONLY items from the menu above.
For each dish, specify realistic portions (e.g., "2 rotis", "1 bowl dal").
Try to hit the calorie and protein targets as closely as possible.
Distribute calories roughly: Breakfast 25%, Lunch 40%, Dinner 35%.

Return ONLY a JSON object with this structure:
{{
  "breakfast": [ {{"dish": str, "portions": str, "estimated_kcal": int, "estimated_protein": int}} ],
  "lunch":     [ {{"dish": str, "portions": str, "estimated_kcal": int, "estimated_protein": int}} ],
  "dinner":    [ {{"dish": str, "portions": str, "estimated_kcal": int, "estimated_protein": int}} ],
  "total_estimated_kcal": int,
  "total_estimated_protein": int,
  "summary": str
}}

summary: 2-3 sentences explaining the plan and how it meets the goals.
Return ONLY valid JSON. No markdown, no extra text."""

    raw = _hf_call(prompt, max_tokens=800)

    if not raw:
        return {
            "error"  : "Could not reach AI backend.",
            "plan"   : None,
            "summary": "AI unavailable. Please check your HF_API_KEY."
        }

    data = _parse_json_response(raw)
    if not data:
        return {"error": "AI returned malformed response.", "plan": None, "raw": raw}

    # Store plan in session
    session["plan"] = data
    session["messages"].append({
        "role"   : "agent",
        "content": f"📋 Here's your meal plan for today:\n\n{data.get('summary', '')}",
        "plan"   : data,
    })

    log.info(f"Session {session_id}: plan generated ({data.get('total_estimated_kcal', '?')} kcal)")
    return data


def update_and_suggest(session_id: str, just_logged: Optional[dict] = None) -> dict:
    """
    Step 2+: Called after each meal logged.
    Recalculates remaining macros → suggests what to eat next.

    just_logged: { meal_id, dish, calories, protein, carbs, fats }

    Returns:
        {
            "remaining": { kcal, protein, carbs, fats },
            "eaten_so_far": [...],
            "next_suggestion": {
                "meal": "lunch" | "dinner",
                "dishes": [ { dish, portions, estimated_kcal } ],
                "note": str,
            }
        }
    """
    session = _sessions.get(session_id)
    if not session:
        return {"error": "Session not found."}

    if just_logged:
        session["eaten"].append(just_logged)

    remaining = _compute_remaining(session)
    menu_text = _get_menu_text(session["menu"])

    # Determine which meal to suggest next
    eaten_meal_ids = [i.get("meal_id") for i in session["eaten"]]
    next_meal = "lunch"
    if "lunch" in eaten_meal_ids:
        next_meal = "dinner"
    if "dinner" in eaten_meal_ids:
        # All meals done
        return {
            "remaining"      : remaining,
            "eaten_so_far"   : session["eaten"],
            "next_suggestion": None,
            "message"        : f"All meals logged! You consumed {remaining['eaten_kcal']} / {remaining['goal_kcal']} kcal today.",
        }

    prompt = f"""You are a nutrition coach. A college student has eaten {remaining['eaten_kcal']} kcal so far today.

Remaining targets for the day:
- Calories: {remaining['remaining_kcal']} kcal
- Protein: {remaining['remaining_protein']}g
- Carbs: {remaining['remaining_carbs']}g  
- Fats: {remaining['remaining_fats']}g

Today's mess menu:
{menu_text}

Suggest what the student should eat for {next_meal} to best meet their remaining targets.
Use only items available in the menu. Be specific about portions.

Return ONLY a JSON object:
{{
  "meal": "{next_meal}",
  "dishes": [ {{"dish": str, "portions": str, "estimated_kcal": int, "estimated_protein": int}} ],
  "total_kcal": int,
  "total_protein": int,
  "note": str
}}

note: 1-2 sentences explaining why this is a good choice for their remaining targets.
Return ONLY valid JSON."""

    raw = _hf_call(prompt, max_tokens=500)
    suggestion = None

    if raw:
        suggestion = _parse_json_response(raw)

    session["messages"].append({
        "role"   : "agent",
        "content": f"Based on what you've eaten, here's my suggestion for {next_meal}.",
        "suggestion": suggestion,
        "remaining" : remaining,
    })

    return {
        "remaining"       : remaining,
        "eaten_so_far"    : session["eaten"],
        "next_suggestion" : suggestion,
        "next_meal"       : next_meal,
    }


def get_session_state(session_id: str) -> dict:
    """Return full session state for frontend display."""
    session = _sessions.get(session_id)
    if not session:
        return {"error": "Session not found."}
    remaining = _compute_remaining(session)
    return {
        "session_id"  : session_id,
        "goals"       : {
            "kcal"   : session["goal_kcal"],
            "protein": session["goal_protein"],
            "carbs"  : session["goal_carbs"],
            "fats"   : session["goal_fats"],
        },
        "remaining"   : remaining,
        "eaten"       : session["eaten"],
        "plan"        : session.get("plan"),
        "messages"    : session["messages"],
    }


def clear_session(session_id: str) -> None:
    """Clear session (call at midnight or when user resets)."""
    _sessions.pop(session_id, None)