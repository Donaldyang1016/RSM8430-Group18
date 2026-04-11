"""
================================================================================
L.O.V.E. Relationship Support Agent — Streamlit App
RSM 8430 Group 18
================================================================================

Main entry point.  Run with:
    streamlit run app/main.py

Wires together:
  - RAG retriever (rag/)
  - Intent router + actions + safety (agent/)
  - SQLite persistence (state/)
  - Prompt templates (app/prompts.py)
  - LLM client (app/llm_client.py)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env before any project imports that read os.environ
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import streamlit as st

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so imports like "rag.retriever" work
# regardless of where Streamlit is launched from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.actions import (
    handle_build_plan,
    handle_reflection,
    handle_retrieve_plan,
    handle_save_plan,
)
from agent.memory import SessionMemory
from agent.router import classify_intent
from agent.safety import screen_message
from app.llm_client import generate_text
from app.prompts import (
    RAG_SYNTHESIS_PROMPT,
    RAG_WEAK_MATCH_RESPONSE,
    SYSTEM_PROMPT,
)
from rag.formatting import format_citations, format_for_llm
from rag.retriever import get_retriever
from state.store import get_or_create_session, init_db, list_sessions

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DISTANCE_THRESHOLD = 0.55  # cosine distance above which results are "weak"


# ============================================================================
# Page config & styling
# ============================================================================

st.set_page_config(
    page_title="L.O.V.E. — Relationship Support Agent",
    page_icon="💬",
    layout="centered",
)


# ============================================================================
# Initialise database & session
# ============================================================================

@st.cache_resource
def _init_once() -> None:
    """Run heavy one-time setup."""
    init_db()

_init_once()


def _ensure_session() -> str:
    """Bootstrap or restore a session ID in Streamlit session_state."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = get_or_create_session()
    return st.session_state.session_id


# ============================================================================
# Sidebar — session management & info
# ============================================================================

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 💬 L.O.V.E.")
        st.caption("Listen · Open Dialogue · Validate · Encourage")
        st.divider()

        # Session info
        sid = _ensure_session()
        st.markdown(f"**Session:** `{sid}`")

        # New session button
        if st.button("🆕 New Session"):
            new_sid = get_or_create_session()
            st.session_state.session_id = new_sid
            st.session_state.messages = []
            st.session_state.latest_plan = None
            st.rerun()

        # Load previous session
        st.divider()
        st.markdown("**Load a previous session:**")
        sessions = list_sessions()
        session_ids = [s["session_id"] for s in sessions]
        if session_ids:
            selected = st.selectbox(
                "Select session",
                session_ids,
                index=session_ids.index(sid) if sid in session_ids else 0,
                label_visibility="collapsed",
            )
            if selected != sid:
                if st.button("Load Session"):
                    st.session_state.session_id = selected
                    st.session_state.messages = []
                    st.session_state.latest_plan = None
                    st.rerun()
        else:
            st.caption("No sessions yet.")

        st.divider()
        st.markdown(
            "**What I can do:**\n"
            "- Answer relationship questions (RAG)\n"
            "- Build a conversation plan\n"
            "- Guide a reflection exercise\n"
            "- Save & retrieve your plans"
        )
        st.divider()
        st.caption(
            "⚠️ I am not a licensed therapist. This tool is for educational "
            "purposes and does not replace professional support."
        )


# ============================================================================
# Chat message helpers
# ============================================================================

def _load_messages_into_state(memory: SessionMemory) -> None:
    """Load persisted messages into st.session_state on first run."""
    if "messages" not in st.session_state or not st.session_state.messages:
        history = memory.get_history(limit=100)
        st.session_state.messages = [
            {"role": m["role"], "content": m["content"]}
            for m in history
        ]
    if "latest_plan" not in st.session_state:
        st.session_state.latest_plan = None


def _display_chat() -> None:
    """Render the chat message history."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# ============================================================================
# Core agent logic — process a single user message
# ============================================================================

def _process_message(user_input: str, memory: SessionMemory) -> str:
    """
    Run the full agent pipeline for one user message.
    Returns the assistant response string.
    """
    session_id = memory.session_id

    # ── 1. Safety screening ───────────────────────────────────────────────
    safety = screen_message(user_input)
    if not safety["safe"]:
        # Persist the user message with the safety category
        memory.add_user_message(user_input, intent=safety["category"])
        response = safety["response"]
        memory.add_assistant_message(response, intent=safety["category"])
        return response

    # ── 2. Check for in-progress multi-turn action ────────────────────────
    action_state = memory.get_action_state()

    # ── 3. Classify intent ────────────────────────────────────────────────
    intent, rationale = classify_intent(user_input, action_state, generate_text)

    # Persist user message with intent
    memory.add_user_message(user_input, intent=intent)

    # ── 4. Dispatch ───────────────────────────────────────────────────────
    try:
        if intent == "rag_qa":
            response = _handle_rag(user_input, memory)

        elif intent == "build_plan":
            result = handle_build_plan(user_input, memory, generate_text)
            response = result["message"]
            if result["type"] == "plan":
                st.session_state.latest_plan = result["plan"]

        elif intent == "reflection":
            result = handle_reflection(user_input, memory, generate_text)
            response = result["message"]

        elif intent == "save_plan":
            latest = st.session_state.get("latest_plan")
            result = handle_save_plan(session_id, latest)
            response = result["message"]

        elif intent == "retrieve_plan":
            result = handle_retrieve_plan(session_id)
            response = result["message"]

        elif intent == "unsafe":
            response = safety["response"] or (
                "I'm not able to help with that. If you're in crisis, "
                "please reach out to a professional."
            )

        elif intent == "out_of_scope":
            response = (
                "That's outside what I can help with. I'm a relationship "
                "support agent — I can help with communication, conflict "
                "resolution, and reflection exercises. What's on your mind?"
            )

        else:
            response = (
                "I'm not sure how to help with that. Could you tell me more "
                "about your relationship situation?"
            )

    except Exception as e:
        response = (
            "I ran into an issue processing your request. "
            "Could you try rephrasing? If the problem persists, "
            "try starting a new session."
        )

    # ── 5. Persist assistant response ─────────────────────────────────────
    memory.add_assistant_message(response, intent=intent)
    return response


def _handle_rag(user_input: str, memory: SessionMemory) -> str:
    """Retrieve from knowledge base, synthesize grounded answer."""
    try:
        retriever = get_retriever()
        results = retriever.retrieve(user_input, k=3)
    except Exception:
        return (
            "I'm having trouble accessing the knowledge base right now. "
            "Could you try again in a moment?"
        )

    # Check retrieval quality
    if not results or results[0]["distance"] > DISTANCE_THRESHOLD:
        return RAG_WEAK_MATCH_RESPONSE

    context = format_for_llm(results)
    citations = format_citations(results)
    history = memory.get_history_for_prompt(limit=6)

    prompt = RAG_SYNTHESIS_PROMPT.format(
        context=context,
        history=history,
        user_message=user_input,
    )

    try:
        answer = generate_text(SYSTEM_PROMPT, prompt)
    except Exception:
        return (
            "I found some relevant examples but had trouble generating a "
            "response. Here are the sources I found:\n\n" + citations
        )

    # Append citations
    response = f"{answer.strip()}\n\n---\n📚 {citations}"
    return response


# ============================================================================
# Main app
# ============================================================================

def main() -> None:
    _render_sidebar()

    session_id = _ensure_session()
    memory = SessionMemory(session_id)

    # Load persisted history into session_state
    _load_messages_into_state(memory)

    # Header
    st.title("L.O.V.E. 💬")
    st.caption(
        "Listen · Open Dialogue · Validate Feelings · Encourage Solutions  —  "
        "Your relationship support companion"
    )

    # Show welcome message if conversation is empty
    if not st.session_state.messages:
        welcome = (
            "Hey there, welcome! I'm **L.O.V.E.** — think of me as a supportive "
            "friend who's here to help you work through relationship stuff.\n\n"
            "You can just talk to me like you'd talk to a friend. Tell me what's "
            "going on and I'll listen first, then we can figure things out together.\n\n"
            "Here's what I'm good at:\n"
            "- 💬 **Talking through relationship questions** — I draw on real therapist insights\n"
            "- 📝 **Building a conversation plan** — for when you need to have a tough talk\n"
            "- 🪞 **Guided reflection** — to help you sort out how you're feeling\n"
            "- 💾 **Saving your plans** — so you can come back to them later\n\n"
            "So — what's on your mind?"
        )
        st.session_state.messages.append({"role": "assistant", "content": welcome})
        memory.add_assistant_message(welcome, intent="system")

    # Display chat
    _display_chat()

    # Chat input
    if user_input := st.chat_input("Type your message here..."):
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Process and show assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = _process_message(user_input, memory)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
