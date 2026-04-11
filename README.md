# L.O.V.E. - Relationship Support Agent
**Listen. Open Dialogue. Validate Feelings. Encourage Solutions.**

RSM 8430 - Applications of Large Language Models | Group 18

---

## What This Is

L.O.V.E. is a relationship support chatbot that helps users:
- talk through conflict with empathy,
- build practical communication plans,
- reflect on recurring emotional patterns.

The system uses Retrieval-Augmented Generation (RAG) grounded on curated therapist-authored examples from CounselChat. It is not a therapist and does not provide medical or legal advice.

---

## What's New In This Version

This repository now includes five major upgrades:

1. Core logic fixes
- Recent-memory loading now correctly uses the latest turns.
- Build-plan cold start no longer stores trigger text as the `issue` slot.
- Safety check order now prioritizes crisis and abuse ahead of injection.

2. Retrieval optimization
- Hybrid retrieval (vector + lexical BM25-style scoring).
- Score fusion reranking (`semantic_score`, `lexical_score`, `hybrid_score`).
- Query rewriting before retrieval for better recall.

3. Structured profile memory
- Persistent user profile signals in SQLite (`user_profiles` table).
- Profile-aware prompting for RAG, plan generation, and reflection.

4. Evaluation harness
- Added runnable evaluation suite with golden test cases.
- Includes retrieval metrics (`Hit@k`, `MRR@k`), routing/safety accuracy, plan-flow regression checks, and binary rubric checks.

5. UX and UI refresh
- Redesigned Streamlit experience with:
  - **Popular Situations** sidebar expanders (Trust, Conflict, Communication, Boundaries, Breakup, Emotional Distance),
  - deterministic phrase-click guided follow-up prompts,
  - cleaner **Your Relationship Tip Card** display (category, actionable examples, HuggingFace source summary),
  - plan-builder progress and profile snapshot sidebar context.

---

## Setup And Run

### Prerequisites
- Python 3.9+
- A running OpenAI-compatible LLM endpoint

### 1. Clone
```bash
git clone <repo-url>
cd RSM8430-Group18
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Build vector index (one-time)
```bash
python rag/build_index.py
```

Expected output includes:
`All 225 documents indexed successfully.`

### 4. Configure the LLM endpoint

Set these environment variables to point to your LLM:

```bash
export LLM_API_BASE="https://rsm-8430-finalproject.bjlkeng.io"   # course-provided endpoint URL
export LLM_API_KEY="your-api-key"                      # student number
export LLM_MODEL="qwen3-30b-a3b-fp8"                  # model name
```

### 5. Start app
```bash
streamlit run app/main.py
```

Open `http://localhost:8501`.

---

## Evaluation

Run:
```bash
python evaluation/run_eval.py
```

Outputs:
- `evaluation/results.csv`
- `evaluation/summary.json`

Current evaluator modules:
- Retrieval golden-set checks (`Hit@k`, `MRR@k`)
- Intent routing accuracy
- Safety category accuracy
- Build-plan flow regression
- Binary L.O.V.E. rubric checks
- Failure taxonomy summary

Note: retrieval metrics require local retrieval dependencies (for example `chromadb`) installed and an available index.

---

## Repository Structure

```
RSM8430-Group18/
├── app/
│   ├── main.py                # Streamlit app + agent pipeline + UI
│   ├── prompts.py             # Prompt templates (routing, RAG, plan, reflection)
│   └── llm_client.py          # OpenAI-compatible API wrapper (retries, tunables)
├── agent/
│   ├── router.py              # Intent routing (keyword + LLM fallback)
│   ├── safety.py              # Regex safety/guardrails
│   ├── actions.py             # Build plan, reflection, save/retrieve actions
│   └── memory.py              # Session memory + structured profile inference
├── rag/
│   ├── build_index.py         # Build Chroma vector index from CSV
│   ├── retriever.py           # Hybrid retriever (vector + lexical rerank)
│   └── formatting.py          # Context packing + citation formatting
├── state/
│   ├── schema.sql             # SQLite schema (sessions/messages/actions/plans/profiles)
│   └── store.py               # DB helper functions
├── evaluation/
│   ├── test_cases.json        # Golden evaluation cases
│   ├── run_eval.py            # Evaluation runner
│   ├── results.csv            # Generated eval results
│   └── summary.json           # Generated eval summary
├── data/
│   ├── filtered_counselchat.csv
│   ├── chroma_db/             # Generated vector store
│   └── love_agent.db          # Generated SQLite database
├── technical_implementation.md
├── README.md
└── requirements.txt
```

---

## Core Capabilities

### 1. Relationship Q&A (RAG)
- Query rewrite -> hybrid retrieve -> grounded synthesis.
- Outputs therapist-backed suggestions with a structured **Your Relationship Tip Card**:
  - `Category` (human-readable),
  - `Actionable examples` (1-3 concise bullets),
  - `Sourced therapist examples from HuggingFace` summary line.

### 2. Build Conversation Plan
- Multi-turn slot flow (`issue -> goal -> tone`) for cold start.
- Context-aware direct plan generation for richer conversations.
- Plan structure:
  - Opening Statement
  - Talking Points
  - Validating Phrase
  - Boundary Phrase
  - Suggested Follow-up

### 3. Reflection Exercise
- Generates guided reflection prompts, assumptions-vs-facts prompts, and emotional check-ins.

### 4. Save/Retrieve Plan
- Persists plans by `session_id` in SQLite.

### 5. Safety/Guardrails
- Crisis, abuse, medical, legal, injection, and out-of-scope screening.
- Safety checks run before routing.

### 6. Guided Sidebar Interaction
- `Popular Situations` provides expandable scenario families with 3-5 clickable dilemma phrases each.
- Clicking a phrase auto-posts that phrase and triggers a deterministic, situation-specific follow-up question asking for concrete details.
- `Quick Questions` remains available in plain text to guide common tasks.

---

## L.O.V.E. Response Framework

All generated support responses follow:
1. Listen
2. Open Dialogue
3. Validate Feelings
4. Encourage Solutions

This is explicitly enforced in prompts to avoid jumping straight to generic advice.

---

## Known Limitations

- Knowledge base remains small (225 curated examples), so niche scenarios may still weak-match.
- System is local-state oriented (no cross-device sync).
- Rule-based safety can miss uncommon phrasing not covered by patterns.
- Evaluation harness is intended for iterative development, not clinical validation.

---

## Disclaimer

This tool is for educational and supportive communication use only. It is not a licensed therapist and is not a replacement for professional mental health, legal, or medical support.
