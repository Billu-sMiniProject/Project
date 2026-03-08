# 🍛 FitAI

> AI-Powered College Mess Nutrition Tracker

`Python` `FastAPI` `EfficientNet-B2` `LangChain` `ChromaDB` `HuggingFace` `MERN`

---

## Overview

FitAI is a full-stack AI application built for college students to track nutrition from mess (cafeteria) food. Students can photograph their plate, scan a menu board, or type a dish name — FitAI identifies the food, looks up its nutrition, and helps plan meals to hit daily macro targets.

---

## Core Features

| Feature | What it Does | Technology |
|---|---|---|
| 📸 Photo Recognition | Classify food from a photo | EfficientNet-B2 |
| 📋 OCR Menu Scan | Photograph a physical menu board | EasyOCR + RapidFuzz |
| 🔢 Nutrition Lookup | 5-layer fallback nutrition data | DB + Dual HF LLMs |
| 🤖 Meal Planning Agent | AI plans your full day of meals | LangChain ReAct + HF |
| 💬 Nutrition Q&A (RAG) | Ask questions about your meals | ChromaDB + Sentence Transformers |
| ⚠️ Calorie Alerts | Exceeded target? Get suggestions | HuggingFace LLMs |
| 🔐 Authentication | Secure login with email OTP | MERN + JWT + Resend |

---

## Architecture

FitAI has two separate backends and two frontend layers connected via JWT cookie authentication:

```
User Browser
    │
    ├── Auth Pages  (React :5173)  ──────► Auth Backend  (Node/Express :3000)
    │        sets JWT cookie                    └── MongoDB (users, sessions)
    │
    └── FitAI Pages (HTML :5500)  ──────► FitAI Backend (FastAPI :8000)
             reads JWT cookie                  ├── EfficientNet-B2 model
             window.FITAI_USER                 ├── ChromaDB (RAG vectors)
                                               └── HuggingFace API / Gemini
```

### Authentication Flow

1. User visits any FitAI HTML page
2. `auth-check.js` calls `GET /api/auth/check-auth` on port 3000
3. No valid JWT cookie → redirected to Auth React app (`:5173`)
4. User logs in / signs up with email OTP verification
5. Auth backend sets HTTP-only JWT cookie (7-day expiry)
6. Redirect back to FitAI page — cookie passes check — page loads
7. `window.FITAI_USER` contains `{ id, name, email }` for all pages

---

## Project Structure

```
fitai/
├── backend/                    ← FastAPI Python backend
│   ├── main.py                 ← API server & all endpoints
│   ├── classifier.py           ← EfficientNet-B2 food classifier
│   ├── nutrition.py            ← 5-layer nutrition lookup
│   ├── nutrition_db.py         ← Static DB: 150+ Indian dishes
│   ├── agent.py                ← LangChain meal planning agent
│   ├── ocr.py                  ← EasyOCR menu board scanner
│   ├── rag.py                  ← ChromaDB RAG Q&A system
│   └── model/
│       └── efficientnet_b2_best.pth   ← trained model weights
│
├── frontend/                   ← Plain HTML pages
│   ├── auth-check.js           ← JWT gate (add to every page)
│   ├── onboarding.html         ← Goal & macro setup
│   ├── menu.html               ← Weekly mess menu input
│   ├── log.html                ← Daily food log & calorie ring
│   └── plan.html               ← AI agent meal plan view
│
└── auth/                       ← Auth-Mern (separate repo/laptop)
    ├── backend/                ← Node/Express + MongoDB
    └── frontend/               ← React auth UI (login/signup/OTP)
```

---

## API Reference

All FitAI endpoints are served by FastAPI on **port 8000**.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Upload food image → top-3 dish predictions + nutrition |
| `POST` | `/nutrition` | Manual dish name → nutrition lookup |
| `POST` | `/ocr-menu` | Photo of menu board → matched dishes + nutrition |
| `POST` | `/suggest` | Calorie exceeded → AI suggests light food + exercise |
| `POST` | `/plan/start` | Start agent session → generate full day plan from menu |
| `POST` | `/plan/log` | Log a meal → recalculate remaining macros + next suggestion |
| `GET` | `/plan/state/:id` | Get current session state |
| `DELETE` | `/plan/session/:id` | Clear a planning session |
| `POST` | `/rag/ask` | Natural language nutrition Q&A using RAG + LLM |
| `GET` | `/health` | Check model load status and device |

---

## Nutrition Lookup — 5-Layer Fallback

Every dish lookup goes through these layers in order, stopping at the first hit:

1. **Cache** — JSON file on disk, instant lookup, persists across restarts
2. **Exact DB match** — 150+ Indian mess/street foods in `nutrition_db.py`
3. **Fuzzy DB match** — RapidFuzz `token_sort_ratio`, threshold 78
4. **Parallel LLM** — Qwen-72B + Mixtral-8x7B run simultaneously; results averaged for accuracy
5. **Gemini** — Last resort; `gemini-2.0-flash`, used only if all HF models fail

> The parallel averaging in Layer 4 is the key accuracy improvement — two independent models estimating the same dish, then taking the mean, dramatically reduces outliers.

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- Node.js 18+
- MongoDB (local or Atlas)
- HuggingFace account + API key (free tier works)
- Resend account for email (free tier: 3,000 emails/month)
- Gemini API key *(optional — only used if all HF models fail)*

---

### 1. FitAI Backend (FastAPI)

```bash
cd fitai/backend

pip install fastapi uvicorn python-dotenv torch torchvision timm \
            rapidfuzz httpx easyocr chromadb sentence-transformers \
            langchain langchain-community --break-system-packages

# Create .env
HF_API_KEY=hf_xxxxxxxxxxxxxxxxxxxx
GEMINI_API_KEY=AIza...              # optional

# Place your trained model weights at:
# fitai/backend/model/efficientnet_b2_best.pth

uvicorn main:app --reload --port 8000
```

---

### 2. Auth Backend (Node/Express)

```bash
cd auth/backend
npm install

# Create backend/.env
PORT=3000
MONGODB_URI=mongodb://127.0.0.1:27017/fitai-auth
JWT_SECRET=your_super_secret_key_here
RESEND_API_KEY=re_xxxxxxxxxxxxxxxxxxxx
CLIENT_URL=http://localhost:5173

npm run dev
```

> **Important:** Update CORS in `backend/index.js` to allow both origins:
> ```js
> origin: ['http://localhost:5173', 'http://localhost:5500']
> ```

---

### 3. Auth Frontend (React)

```bash
cd auth/frontend
npm install
npm run dev          # starts on http://localhost:5173
```

---

### 4. FitAI Frontend (HTML)

Open `fitai/frontend/` in VS Code with the **Live Server** extension.
Right-click `onboarding.html` → **Open with Live Server** → runs on `http://localhost:5500`.

`auth-check.js` must be in the same folder as the HTML files and must be the **first script tag** in each page's `<head>`:

```html
<script src="auth-check.js"></script>
```

---

## Environment Variables

### FitAI Backend

| Variable | Description |
|---|---|
| `HF_API_KEY` | HuggingFace API key — used for LLM nutrition lookups and meal planning |
| `GEMINI_API_KEY` | *(Optional)* Google Gemini key — last resort fallback only |

### Auth Backend

| Variable | Description |
|---|---|
| `MONGODB_URI` | MongoDB connection string (local or Atlas) |
| `JWT_SECRET` | Long random string — never commit this |
| `RESEND_API_KEY` | Resend.com API key for verification emails |
| `CLIENT_URL` | Frontend URL for password reset links (`http://localhost:5173`) |
| `PORT` | Server port (default: 3000) |

---

## Nutrition Database

`nutrition_db.py` contains 150+ Indian college food items sourced from **IFCT 2017** and **NIN Hyderabad**. All values stored per 100g with a `portion_g` field for scaling to standard serving size.

| Category | Items |
|---|---|
| Mess Breakfast | idli, dosa, upma, poha, uttapam, omelette, bread, cornflakes, maggi variants |
| Mess Staples | rice, roti, naan, puri, biryani (veg / chicken / egg / paneer) |
| Dals & Curries | dal tadka, dal makhani, rajma, chole, moong dal |
| Vegetable Dishes | aloo gobi, bhindi masala, mix veg, malai kofta |
| Paneer | butter masala, palak paneer, bhurji, tikka, manchurian, kadai (10+ varieties) |
| Non-Veg | chicken curry, egg curry, fish masala, mutton curry, prawn curry |
| Street Food | pani puri, samosa, vada pav, momos (4 types), rolls, shawarma |
| Indo-Chinese | chowmein, fried rice, manchurian, hakka / schezwan noodles |
| Drinks | lassi (4 types), buttermilk, smoothies (6 types), cold coffee |
| Desserts | gulab jamun, rasgulla, kulfi, kheer, jalebi, gajar halwa |

---

## Frontend Pages

All pages share a consistent design system: `#0d0d0d` dark background, Syne + DM Sans fonts, `#e8ff47` yellow-green accent.

| Page | Purpose |
|---|---|
| `onboarding.html` | User sets daily calorie target, protein/carb/fat goals, and fitness objective |
| `menu.html` | Weekly mess menu wizard — search and add dishes per meal slot for each day |
| `log.html` | Daily food log with animated calorie ring, macro bars, camera scanner, and AI suggestion card when calories are exceeded |
| `plan.html` | AI agent meal plan view — chat-style messages, remaining macro bars updated after each meal logged |

---

## Security Notes

- JWT tokens stored as **HTTP-only cookies** — inaccessible to JavaScript, XSS-safe
- Passwords hashed with **bcrypt** (salt rounds: 10) — never stored in plain text
- Email OTP tokens expire after **24 hours**; password reset tokens after **1 hour**
- CORS restricted to known origins only (`localhost:5173`, `localhost:5500`)
- `auth-check.js` `returnTo` parameter validated — only `localhost:5500` URLs accepted
- All secrets loaded from `.env` — never hardcoded

> ⚠️ **Warning:** The `auth/backend/index.js` file contains a MongoDB password left in a comment. Remove it before pushing to any repository.

---

## Known Limitations & TODO

- [ ] Agent session state is **in-memory** — restarting FastAPI clears all active planning sessions
- [ ] Classifier requires `efficientnet_b2_best.pth` placed manually — model weights not included in repo
- [ ] EasyOCR downloads ~1.5GB model on first run — ensure internet access on first startup
- [ ] ChromaDB vectors stored in `.chroma_db/` — delete this folder to reset the RAG index
- [ ] HuggingFace free tier has rate limits — parallel LLM calls may queue during peak hours
- [ ] No persistent user log storage — daily logs live in browser `localStorage` only
- [ ] Auth-Mern and FitAI must run on the same machine for cookie sharing to work in development

---

*FitAI — Built for college students, by a college student*
