"""
FitAI - rag.py
==============
Retrieval-Augmented Generation for nutrition Q&A.

Pipeline:
    User question → embed → ChromaDB similarity search →
    retrieve top-k chunks → build prompt with context → HF LLM → answer

Document sources indexed:
    1. nutrition_db.py  → one chunk per dish (name + macros + serving)
    2. Today's menu     → pulled from session / arg at query time
    3. User's 7-day log → passed in at query time as JSON

Install:
    pip install chromadb sentence-transformers --break-system-packages

Usage:
    from rag import get_rag, answer_question
    answer = answer_question("Is today's mess lunch good for weight loss?",
                             today_menu=menu_dict, user_log=log_dict)
"""

import os
import json
import logging
import re
from typing import Optional
from pathlib import Path

log = logging.getLogger("fitai.rag")

# ── Constants ──────────────────────────────────────────────────────────────────
CHROMA_DIR    = Path(__file__).parent / ".chroma_db"
COLLECTION    = "fitai_nutrition"
EMBED_MODEL   = "all-MiniLM-L6-v2"    # 80MB, fast, good for short text
TOP_K         = 4                      # chunks to retrieve
MAX_CONTEXT   = 1000                   # chars of context passed to LLM

# ── Singletons ─────────────────────────────────────────────────────────────────
_chroma_client     = None
_collection        = None
_embed_model       = None
_db_populated      = False
_last_context_hash = None   # avoid re-upserting identical menu/log data


# ══════════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════════
def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            log.info(f"Loading embedding model '{EMBED_MODEL}'…")
            _embed_model = SentenceTransformer(EMBED_MODEL)
            log.info("Embedding model ready ✅")
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers --break-system-packages"
            )
    return _embed_model


def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        try:
            import chromadb
            _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            _collection    = _chroma_client.get_or_create_collection(
                name=COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
            log.info(f"ChromaDB collection '{COLLECTION}' loaded ({_collection.count()} docs)")
        except ImportError:
            raise RuntimeError(
                "chromadb not installed. "
                "Run: pip install chromadb --break-system-packages"
            )
    return _collection


def _embed(texts: list[str]) -> list[list[float]]:
    model = _get_embed_model()
    return model.encode(texts, show_progress_bar=False).tolist()


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT BUILDING
# ══════════════════════════════════════════════════════════════════════════════
def _dish_to_chunk(dish_key: str, data: dict) -> str:
    """Convert a nutrition_db entry into a human-readable text chunk."""
    name = dish_key.replace("_", " ").title()
    cal  = data.get("calories", 0)
    pro  = data.get("protein",  0)
    car  = data.get("carbs",    0)
    fat  = data.get("fats",     0)
    pg   = data.get("portion_g", 0)
    desc = data.get("serving_desc", "1 serving")

    # Derive health tags
    tags = []
    if pro >= 12:  tags.append("high protein")
    if cal <= 150: tags.append("low calorie")
    if fat <= 3:   tags.append("low fat")
    if car >= 40:  tags.append("high carb")
    if fat >= 15:  tags.append("high fat")

    tag_str = ", ".join(tags) if tags else "moderate nutrition"

    return (
        f"{name}: {cal} kcal per serving ({desc}, {pg}g). "
        f"Protein {pro}g, Carbs {car}g, Fats {fat}g. "
        f"Profile: {tag_str}."
    )


def _menu_to_chunks(menu: dict) -> list[tuple[str, str]]:
    """Convert today's menu into id+text pairs."""
    chunks = []
    for meal_id, dishes in menu.items():
        for d in (dishes or []):
            if not d.get("name"):
                continue
            name = d["name"].replace("_", " ").title()
            text = (
                f"Today's {meal_id} includes {name}: "
                f"{round(d.get('calories', 0))} kcal, "
                f"{d.get('protein', 0):.1f}g protein, "
                f"{d.get('carbs', 0):.1f}g carbs, "
                f"{d.get('fats', 0):.1f}g fats."
            )
            chunks.append((f"menu_{meal_id}_{d['name']}", text))
    return chunks


def _log_to_chunks(user_log: dict) -> list[tuple[str, str]]:
    """Convert user's recent log entries into id+text pairs."""
    chunks = []
    for date_key, day_data in list(user_log.items())[-7:]:  # last 7 days
        for meal_id, items in day_data.items():
            for item in (items or []):
                if item.get("skipped") or not item.get("name"):
                    continue
                name = item["name"].replace("_", " ").title()
                text = (
                    f"On {date_key} at {meal_id}, the user ate {name}: "
                    f"{round(item.get('calories', 0))} kcal, "
                    f"{item.get('protein', 0):.1f}g protein."
                )
                chunks.append((f"log_{date_key}_{meal_id}_{item['name']}", text))
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# POPULATION
# ══════════════════════════════════════════════════════════════════════════════
def populate_db(force: bool = False) -> int:
    """
    Populate ChromaDB with nutrition_db entries.
    Only runs once unless force=True.
    """
    global _db_populated
    if _db_populated and not force:
        return 0

    col = _get_collection()
    if col.count() > 0 and not force:
        log.info(f"ChromaDB already has {col.count()} docs — skipping population.")
        _db_populated = True
        return col.count()

    from nutrition_db import NUTRITION_DB
    log.info(f"Populating ChromaDB with {len(NUTRITION_DB)} dishes…")

    ids, texts, metas = [], [], []
    for dish_key, data in NUTRITION_DB.items():
        chunk = _dish_to_chunk(dish_key, data)
        ids.append(f"db_{dish_key}")
        texts.append(chunk)
        metas.append({"source": "nutrition_db", "dish": dish_key})

    # Batch upsert (ChromaDB handles embeddings internally if we provide texts)
    embeddings = _embed(texts)
    col.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)

    _db_populated = True
    log.info(f"ChromaDB populated with {len(ids)} nutrition entries ✅")
    return len(ids)


def upsert_context(today_menu: Optional[dict] = None,
                   user_log: Optional[dict] = None) -> None:
    """
    Upsert dynamic context (menu + log) into the collection.
    Skips the upsert if the data hasn't changed since last call (hash check).
    """
    global _last_context_hash
    import hashlib

    # Build a cheap hash of the incoming data to detect changes
    raw = json.dumps({"menu": today_menu, "log": user_log}, sort_keys=True, default=str)
    current_hash = hashlib.md5(raw.encode()).hexdigest()
    if current_hash == _last_context_hash:
        log.debug("Context unchanged — skipping upsert")
        return

    col    = _get_collection()
    ids, texts, metas = [], [], []

    if today_menu:
        for cid, text in _menu_to_chunks(today_menu):
            ids.append(cid); texts.append(text); metas.append({"source": "menu"})

    if user_log:
        for cid, text in _log_to_chunks(user_log):
            ids.append(cid); texts.append(text); metas.append({"source": "log"})

    if ids:
        embeddings = _embed(texts)
        col.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)
        log.debug(f"Upserted {len(ids)} context chunks into ChromaDB")

    _last_context_hash = current_hash


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════
def retrieve(question: str, n: int = TOP_K) -> list[str]:
    """Embed the question and retrieve top-k relevant chunks."""
    col = _get_collection()
    if col.count() == 0:
        return []

    q_embed = _embed([question])[0]
    results = col.query(
        query_embeddings=[q_embed],
        n_results=min(n, col.count()),
        include=["documents"],
    )
    return results["documents"][0] if results["documents"] else []


# ══════════════════════════════════════════════════════════════════════════════
# LLM CALL
# ══════════════════════════════════════════════════════════════════════════════
def _hf_call(prompt: str) -> Optional[str]:
    import httpx
    hf_key = os.getenv("HF_API_KEY")
    if not hf_key:
        return None

    # Fast models first: Mistral-7B responds in ~2-3s and is always warm
    models = [
        "mistralai/Mistral-7B-Instruct-v0.3",
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
                    "max_tokens" : 250,   # was 500 — shorter = faster
                },
                headers=headers,
                timeout=30,   # was 40
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            log.warning(f"RAG HF call failed for {model}: {e}")
            continue
    return None


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════
def answer_question(
    question: str,
    today_menu: Optional[dict] = None,
    user_log:   Optional[dict] = None,
    user_goal:  Optional[str]  = None,   # "lose" | "gain" | "maintain"
) -> dict:
    """
    Main RAG entry point.

    Returns:
        {
            "answer": str,
            "sources": [str],    # retrieved chunk snippets used
            "question": str,
        }
    """
    # Ensure DB is populated
    try:
        populate_db()
    except RuntimeError as e:
        return {
            "answer"  : f"RAG system unavailable: {e}",
            "sources" : [],
            "question": question,
        }

    # Upsert fresh context
    try:
        upsert_context(today_menu=today_menu, user_log=user_log)
    except Exception as e:
        log.warning(f"Context upsert failed (non-fatal): {e}")

    # Retrieve relevant chunks
    chunks = retrieve(question, n=TOP_K)

    if not chunks:
        context_str = "No specific nutrition data found in the database."
    else:
        # Truncate to MAX_CONTEXT chars
        context_str = "\n".join(chunks)
        if len(context_str) > MAX_CONTEXT:
            context_str = context_str[:MAX_CONTEXT] + "…"

    goal_note = ""
    if user_goal:
        labels = {"lose": "weight loss (calorie deficit)", "gain": "muscle gain (protein surplus)", "maintain": "maintenance"}
        goal_note = f"\nUser's goal: {labels.get(user_goal, user_goal)}."

    prompt = f"""You are FitAI, a helpful nutrition assistant for college students eating at a mess cafeteria.
Answer the user's question using the nutrition data provided below.{goal_note}
Be specific, practical, and concise (3-5 sentences max). If the data doesn't cover the question, say so honestly.

NUTRITION DATA:
{context_str}

USER QUESTION: {question}

ANSWER:"""

    raw_answer = _hf_call(prompt)

    if not raw_answer:
        # Pure retrieval fallback — return relevant chunks without LLM
        fallback = "Here's what I found in the nutrition database:\n\n" + "\n\n".join(chunks[:2])
        return {
            "answer"  : fallback,
            "sources" : chunks[:TOP_K],
            "question": question,
        }

    return {
        "answer"  : raw_answer,
        "sources" : chunks[:TOP_K],
        "question": question,
    }