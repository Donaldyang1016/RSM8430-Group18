"""
================================================================================
L.O.V.E. Relationship Support Agent — Session Memory Helpers
RSM 8430 Group 18
================================================================================

Lightweight helpers that manage in-session conversation memory and action state.
These wrap state/store.py functions and add convenience methods used by the
agent layer (router, actions) and the app layer (main.py).

Usage:
    from agent.memory import SessionMemory
    mem = SessionMemory(session_id)
    mem.add_user_message("I keep fighting with my partner")
    history = mem.get_history()
"""

from __future__ import annotations

from typing import Any

from state.store import (
    append_message,
    clear_action_state,
    load_action_state,
    load_messages,
    save_action_state,
)


class SessionMemory:
    """Manages conversation history and action state for a single session."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------

    def add_user_message(self, content: str, intent: str | None = None) -> None:
        append_message(self.session_id, "user", content, intent=intent)

    def add_assistant_message(self, content: str, intent: str | None = None) -> None:
        append_message(self.session_id, "assistant", content, intent=intent)

    def get_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return conversation history as a list of {role, content} dicts."""
        return load_messages(self.session_id, limit=limit)

    def get_history_for_prompt(self, limit: int = 10) -> str:
        """
        Return recent messages formatted as a string for LLM context.
        Keeps token usage manageable by limiting to the last `limit` messages.
        """
        messages = self.get_history(limit=limit)
        lines: list[str] = []
        for m in messages:
            role = m["role"].capitalize()
            lines.append(f"{role}: {m['content']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Multi-turn action state (slot filling)
    # ------------------------------------------------------------------

    def get_action_state(self) -> dict[str, Any] | None:
        """Get the current in-progress action, or None."""
        return load_action_state(self.session_id)

    def set_action_state(self, intent: str, slots: dict[str, Any]) -> None:
        """Save or update the in-progress action state."""
        save_action_state(self.session_id, intent, slots)

    def clear_action_state(self) -> None:
        """Clear action state once the action completes."""
        clear_action_state(self.session_id)

    # ------------------------------------------------------------------
    # Convenience: latest generated plan (in Streamlit session_state)
    # ------------------------------------------------------------------
    # The "latest plan" is stored in st.session_state by actions.py
    # and persisted to SQLite via state/store.save_plan().
    # These helpers exist so the agent layer can check without
    # importing Streamlit directly.

    @staticmethod
    def extract_latest_plan_from_state(st_session_state: dict) -> dict[str, Any] | None:
        """Pull the latest generated plan from Streamlit session state."""
        return st_session_state.get("latest_plan")
