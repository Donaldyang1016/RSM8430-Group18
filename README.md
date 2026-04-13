# L.O.V.E. — Relationship Support Agent

**Listen · Open Dialogue · Validate Feelings · Encourage Solutions**

RSM 8430 — Applications of Large Language Models | Group 18

---

## What This Is

L.O.V.E. is an AI-powered relationship support chatbot that helps users:

- Talk through conflict with empathy and structure
- Build actionable conversation plans using guided slot-filling
- Reflect on recurring emotional patterns with therapist-informed prompts

The system uses **Retrieval-Augmented Generation (RAG)** grounded on 225 curated, therapist-authored Q&A pairs from the [CounselChat](https://huggingface.co/datasets/nbertagnolli/counsel-chat) dataset. It is **not a therapist** and does not provide medical, legal, or clinical advice.

### Live Demo

A deployed version of the app is available at:  
**https://rsm8430-group18-mt3nu7g4geqgvsmfq97ikn.streamlit.app/**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Streamlit Frontend (UI)                         │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────────────┐  │
│  │ Popular         │  │ Chat Interface │  │ Quick Actions            │  │
│  │ Situations      │  │ (st.chat_input │  │ (Plan / Reflect / Save)  │  │
│  │ Sidebar         │  │  + messages)   │  │                          │  │
│  └───────┬────────┘  └───────┬────────┘  └────────────┬─────────────┘  │
│          │                   │                        │                 │
└──────────┼───────────────────┼────────────────────────┼─────────────────┘
           │                   │                        │
           ▼                   ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Agent Pipeline                                  │
│                                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────────┐    │
│  │ 1. Safety     │──▶│ 2. Intent    │──▶│ 3. Action Dispatcher     │    │
│  │    Screening  │   │    Router    │   │    (RAG / Plan / Reflect │    │
│  │ (regex-based  │   │ (keyword +   │   │     / Save / Retrieve)  │    │
│  │  guardrails)  │   │  LLM hybrid) │   │                          │    │
│  └──────────────┘   └──────────────┘   └───────────┬──────────────┘    │
│                                                     │                   │
│  ┌──────────────────────────────────────────────────┼──────────────┐    │
│  │                   Session Memory                 │              │    │
│  │  • Conversation history (last N turns)           │              │    │
│  │  • Multi-turn action state (slot-filling)        │              │    │
│  │  • Inferred user profile (regex-based signals)   │              │    │
│  └──────────────────────────────────────────────────┘              │    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
           │                                          │
           ▼                                          ▼
┌──────────────────────────┐        ┌──────────────────────────────────┐
│   RAG Retrieval Pipeline │        │       SQLite Persistence          │
│                          │        │                                    │
│  User Query              │        │  • sessions                       │
│      │                   │        │  • messages (role, content, intent)│
│      ▼                   │        │  • action_state (slots as JSON)   │
│  Query Rewrite (LLM)     │        │  • saved_plans                    │
│      │                   │        │  • user_profiles (JSON signals)   │
│      ▼                   │        │                                    │
│  Hybrid Retrieval        │        └──────────────────────────────────┘
│  ┌─────────┬──────────┐  │
│  │ Vector  │ Lexical  │  │        ┌──────────────────────────────────┐
│  │ (Chroma │ (BM25    │  │        │       LLM Endpoint                │
│  │  embed) │  score)  │  │        │                                    │
│  └────┬────┴────┬─────┘  │        │  OpenAI-compatible API            │
│       │  Score Fusion    │        │  (Qwen 3 30B-A3B FP8)            │
│       │  (70/30 rerank)  │        │  • Retries + timeout handling     │
│       ▼                  │        │  • Thinking-tag stripping         │
│  Top-K Results           │        │                                    │
│      │                   │        └──────────────────────────────────┘
│      ▼                   │
│  RAG Synthesis (LLM)     │
│      │                   │
│      ▼                   │
│  Response + Tip Card     │
│                          │
└──────────────────────────┘
```

### How a Message Flows Through the System

1. **Safety screening** — Every incoming message is checked against regex patterns for crisis language, abuse, medical/legal requests, and prompt injection. Flagged messages receive an immediate safe response without hitting the LLM.
2. **Profile inference** — The memory module scans for relationship signals (partner mentions, emotional keywords, tone preferences) and persists them to SQLite for cross-turn personalization.
3. **Intent routing** — A two-stage classifier first tries high-confidence keyword rules, then falls back to the LLM for ambiguous cases. Possible intents: `rag_qa`, `build_plan`, `reflection`, `save_plan`, `retrieve_plan`, `out_of_scope`.
4. **Action dispatch** — Based on intent:
   - **RAG Q&A**: rewrites the query → runs hybrid retrieval → synthesizes a grounded response with a Relationship Tip Card
   - **Build Plan**: runs a multi-turn slot-filling flow (issue → goal → tone) or generates a plan directly when context is rich
   - **Reflection**: generates guided prompts for self-examination
   - **Save/Retrieve Plan**: persists or loads the latest plan from SQLite
5. **Response rendering** — The response is displayed in the chat, with Tip Cards shown as expandable UI widgets.

---

## Setup And Run

### Prerequisites

- Python 3.9+
- A running OpenAI-compatible LLM endpoint (the course-provided endpoint is preconfigured)

### 1. Clone

```bash
git clone <repo-url>
cd RSM8430-Group18
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Build the vector index (one-time)

```bash
python rag/build_index.py
```

Expected output: `All 225 documents indexed successfully.`

### 4. Configure environment variables

Copy the example env file and fill in your credentials:

```bash
cp .env.example .env
```

Then open `.env` and set your **student number** as the API key:

```dotenv
LLM_API_BASE=https://rsm-8430-finalproject.bjlkeng.io
LLM_API_KEY=<your-student-number>
LLM_MODEL=qwen3-30b-a3b-fp8
```

> The app reads `.env` automatically at startup via `python-dotenv`. You do **not** need to export these variables manually.

### 5. Start the app

```bash
streamlit run app/main.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Evaluation

### Running the Test Suite

```bash
python evaluation/run_eval.py
```

This produces two artifacts in `evaluation/`:

| File | Description |
|------|-------------|
| `results.csv` | Per-case pass/fail with details and failure reasons |
| `summary.json` | Aggregate metrics and failure taxonomy |

### What Is Tested

The evaluation harness (`evaluation/run_eval.py`) runs **7 categories** of automated checks against the golden test cases in `evaluation/test_cases.json` (24 total cases):

| Category | Cases | What it checks | Metric |
|---|---|---|---|
| **Retrieval** | `ret_01`–`ret_05` | Queries against the ChromaDB index; verifies the expected topic appears in the top-K results | `Hit@k`, `MRR@k` |
| **Intent Routing** | `rte_01`–`rte_05` | Feeds user messages through the router and checks the returned intent matches expected | Accuracy |
| **Safety Screening** | `saf_01`–`saf_06` | Validates that crisis, abuse, medical, legal, injection, and safe messages are classified correctly | Accuracy |
| **Plan Flow** | `pln_01` | Multi-turn regression: sends issue → goal → tone messages and verifies the slot-filling produces a complete plan | Pass/Fail |
| **Actions** | `act_01`–`act_02` | Single-turn save and retrieve plan operations | Pass/Fail |
| **Edge Cases** | `edg_01`–`edg_03` | Empty input, gibberish text, and mid-flow topic switch handling | Pass/Fail |
| **L.O.V.E. Rubric** | `rbk_01`–`rbk_02` | Checks LLM responses for the 4 L.O.V.E. signals: listen acknowledgment, open-dialogue question, validation phrase, encouraging next step | Binary per-signal |

### Latest Results

From the most recent evaluation run:

| Metric | Score |
|--------|-------|
| Retrieval Hit@k | 1.00 |
| Retrieval MRR@k | 1.00 |
| Routing Accuracy | 1.00 |
| Safety Accuracy | 1.00 |
| Plan Flow Pass Rate | 1.00 |
| Edge Case Pass Rate | 1.00 |
| Action Pass Rate | 0.50 |
| Rubric Pass Rate | 0.00 |

**21/24 cases pass.** The 3 failures are: 1 retrieve-plan edge case (empty DB), and 2 rubric checks where the LLM response missed specific phrasing patterns the regex expects (e.g., "it sounds like" for listen signal). These rubric checks are intentionally strict to surface areas for prompt tuning.

### How to Add New Test Cases

Add entries to `evaluation/test_cases.json` following the existing schema. Each case requires:

```json
{
  "id": "ret_06",
  "category": "retrieval",
  "input": "how do I handle jealousy",
  "expected_topic": "jealousy"
}
```

---

## Repository Structure

```
RSM8430-Group18/
├── app/
│   ├── __init__.py
│   ├── main.py                # Streamlit UI + agent pipeline orchestration
│   ├── prompts.py             # All LLM prompt templates (system, routing, RAG, plan, reflection)
│   └── llm_client.py          # OpenAI-compatible API wrapper with retries and timeout
├── agent/
│   ├── __init__.py
│   ├── router.py              # Two-stage intent classifier (keyword rules → LLM fallback)
│   ├── safety.py              # Regex-based safety screening (crisis, abuse, injection, etc.)
│   ├── actions.py             # Action handlers: build_plan, reflection, save/retrieve plan
│   └── memory.py              # Session memory + regex-based user profile inference
├── rag/
│   ├── __init__.py
│   ├── build_index.py         # One-time ChromaDB index builder from CounselChat CSV
│   ├── retriever.py           # Hybrid retriever (vector + BM25 lexical with score fusion)
│   └── formatting.py          # Context packing for LLM + Tip Card citation formatting
├── state/
│   ├── __init__.py
│   ├── schema.sql             # SQLite schema (sessions, messages, action_state, plans, profiles)
│   └── store.py               # Database CRUD helpers with upsert semantics
├── evaluation/
│   ├── __init__.py
│   ├── test_cases.json        # 24 golden evaluation cases across 7 categories
│   ├── run_eval.py            # Automated evaluation harness
│   ├── results.csv            # Generated per-case results
│   └── summary.json           # Generated aggregate metrics
├── data/
│   ├── filtered_counselchat.csv  # 225 curated therapist Q&A pairs
│   ├── chroma_db/                # Generated ChromaDB vector store (git-ignored)
│   └── love_agent.db             # Generated SQLite database (git-ignored)
├── .env.example               # Template for environment variables
├── .gitignore
├── technical_implementation.md
├── requirements.txt
└── README.md
```

---

## Core Capabilities

### 1. Relationship Q&A (RAG)

Query rewrite → hybrid retrieval → grounded synthesis. Each response includes a **Relationship Tip Card** with category, 1-3 actionable bullets, and a source attribution linking back to the CounselChat dataset.

### 2. Build Conversation Plan

Multi-turn slot-filling (`issue → goal → tone`) with a progress bar in the sidebar. When the conversation already has enough context, the system skips slots and generates a plan directly. Plan output includes: Opening Statement, Talking Points, Validating Phrase, Boundary Phrase, and Suggested Follow-up.

### 3. Reflection Exercise

Generates guided reflection prompts, assumptions-vs-facts exercises, and emotional check-ins tailored to the user's inferred profile.

### 4. Save / Retrieve Plan

Persists and retrieves plans by session ID in SQLite.

### 5. Safety & Guardrails

Regex-based screening for crisis language, abuse, medical/legal requests, prompt injection, and off-topic messages. Safety runs **before** intent routing so flagged content never reaches the LLM.

### 6. Guided Sidebar Interaction

The **Popular Situations** sidebar offers 6 scenario families (Trust, Conflict, Communication, Boundaries, Breakup, Emotional Distance) with 4 clickable phrases each. Clicking a phrase triggers a deterministic, situation-specific follow-up question that asks for concrete details.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit 1.28+ |
| LLM | Qwen 3 30B-A3B (FP8) via OpenAI-compatible API |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector Store | ChromaDB |
| Retrieval | Hybrid (vector cosine + BM25 lexical), 70/30 score fusion |
| Persistence | SQLite (sessions, messages, plans, profiles) |
| Data Source | [CounselChat](https://huggingface.co/datasets/nbertagnolli/counsel-chat) (225 curated pairs) |

---

## L.O.V.E. Response Framework

All generated responses are prompted to follow this four-step structure:

1. **Listen** — Acknowledge what the user shared ("It sounds like…")
2. **Open Dialogue** — Ask a clarifying or deepening question
3. **Validate Feelings** — Normalize the user's emotional experience
4. **Encourage Solutions** — Suggest one concrete, doable next step

This is enforced in `app/prompts.py` via the system prompt and synthesis templates.

---

## Known Limitations

- Knowledge base remains small (225 curated examples), so niche scenarios may still weak-match.
- System is local-state oriented (no cross-device sync).
- Rule-based safety can miss uncommon phrasing not covered by patterns.
- Evaluation harness is intended for iterative development, not clinical validation.

---

## Disclaimer

This tool is for educational and supportive communication use only. It is not a licensed therapist and is not a replacement for professional mental health, legal, or medical support.
