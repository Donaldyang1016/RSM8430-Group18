"""
================================================================================
Person B — Retrieval Lead
L.O.V.E. Relationship Support Agent | RSM 8430 Group 18
================================================================================
 
WHAT THIS FILE DOES
-------------------
Formats retrieval results into strings the Synthesis/Response agent can use.
Person C/D call format_for_llm() to get a clean context block for the prompt.
"""
 
from __future__ import annotations
 
 
def format_for_llm(results: list[dict], max_chars_per_doc: int = 600) -> str:
    """
    Formats retrieval results into a context block for the LLM prompt.
 
    Args:
        results:          List of result dicts from retriever.retrieve()
        max_chars_per_doc: Max chars to include per answer (avoids prompt bloat)
 
    Returns:
        A formatted string like:
 
            --- Source 1 [cc_0042] (marriage) ---
            Question: My husband wants a divorce...
            Therapist Answer: Wow that is tough...
            ---
 
    Usage in prompt:
        context = format_for_llm(results)
        prompt  = f"Based on these examples from therapists:\\n\\n{context}\\n\\nUser: {query}"
    """
    if not results:
        return "No relevant examples found in the knowledge base."
 
    lines = []
    for i, r in enumerate(results, 1):
        answer = r.get("answer_text", "")
        if len(answer) > max_chars_per_doc:
            answer = answer[:max_chars_per_doc] + "..."
 
        lines.append(f"--- Source {i} [{r['doc_id']}] ({r['project_topic']}) ---")
        lines.append(f"Question: {r['question_title']}")
        lines.append(f"Therapist Answer: {answer}")
        lines.append("---")
        lines.append("")
 
    return "\n".join(lines).strip()
 
 
def format_citations(results: list[dict]) -> str:
    """
    Returns a short citation line for display in the UI.
 
    Example output:
        Sources: [cc_0042] My husband wants a divorce (marriage),
                 [cc_0101] My boyfriend cheated (trust)
    """
    if not results:
        return ""
    citations = [r["citation"] for r in results]
    return "Sources: " + " | ".join(citations)
 
