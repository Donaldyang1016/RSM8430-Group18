"""
================================================================================
Person B — Retrieval Lead
L.O.V.E. Relationship Support Agent | RSM 8430 Group 18
================================================================================
 
WHAT THIS FILE DOES
-------------------
Provides the retriever function that the rest of the app calls.
Person C and D import get_retriever() and call retrieve(query).
 
Also runnable as a standalone script to test retrieval quality on 10 sample
questions (run: python rag/retriever.py).
 
USAGE FROM OTHER MODULES
-------------------------
    from rag.retriever import get_retriever
 
    retriever = get_retriever()
    results   = retriever.retrieve("my partner and I keep fighting")
 
    for r in results:
        print(r["doc_id"])
        print(r["project_topic"])
        print(r["question_title"])
        print(r["answer_snippet"])   # first 300 chars of answer
        print(r["distance"])         # cosine distance — lower = more similar
        print(r["citation"])         # formatted citation string for the UI
 
RETURN FORMAT
-------------
Each result is a dict:
    {
        "doc_id":         "cc_0042",
        "question_title": "My husband wants a divorce...",
        "question_text":  "...",
        "answer_text":    "...",
        "answer_snippet": "...",   # first 300 chars, for display
        "project_topic":  "marriage",
        "original_topic": "relationships",
        "tier":           1,
        "distance":       0.21,
        "citation":       "[cc_0042] My husband wants a divorce... (marriage)"
    }
"""
 
from __future__ import annotations
 
from pathlib import Path
 
import chromadb
from chromadb.utils import embedding_functions
 
# ================================================================================
# Config — must match build_index.py
# ================================================================================
 
CHROMA_DIR  = Path("data/chroma_db")
COLLECTION  = "counselchat"
EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_K   = 3   # top-k results returned by default. Agent can override.
 
 
# ================================================================================
# Retriever class
# ================================================================================
 
class CounselChatRetriever:
    """
    Thin wrapper around ChromaDB. Call retrieve(query) to get top-k results.
    Instantiate once and reuse (embedding model loads on first call).
    """
 
    def __init__(self, chroma_dir: str | Path = CHROMA_DIR,
                 collection_name: str = COLLECTION,
                 embed_model: str = EMBED_MODEL):
        self._client = chromadb.PersistentClient(path=str(chroma_dir))
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embed_model
        )
        self._collection = self._client.get_collection(
            name=collection_name,
            embedding_function=self._ef,
        )
 
    def retrieve(self, query: str, k: int = DEFAULT_K,
                 topic_filter: str | None = None) -> list[dict]:
        """
        Returns top-k most similar documents for the given query.
 
        Args:
            query:        User's question or situation description.
            k:            Number of results to return (default 3).
            topic_filter: Optional project_topic to restrict results to
                          (e.g. "breakup", "trust"). None = no filter.
 
        Returns:
            List of result dicts (see module docstring for format).
        """
        if not query or not query.strip():
            return []
 
        where = {"project_topic": topic_filter} if topic_filter else None
 
        results = self._collection.query(
            query_texts=[query.strip()],
            n_results=min(k, self._collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
 
        output = []
        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]
 
        for doc_text, meta, dist in zip(docs, metas, distances):
            # Extract answer_text from document_text (everything after "Therapist Answer: ")
            answer_text = ""
            if "Therapist Answer:" in doc_text:
                answer_text = doc_text.split("Therapist Answer:", 1)[1].strip()
 
            question_text = ""
            if "Question:" in doc_text:
                q_block = doc_text.split("Question:", 1)[1]
                # stop before "Therapist Answer"
                question_text = q_block.split("Therapist Answer:")[0].strip()
 
            answer_snippet = answer_text[:300] + ("..." if len(answer_text) > 300 else "")
 
            title    = meta.get("question_title", "")
            topic    = meta.get("project_topic", "")
            doc_id   = meta.get("doc_id", "")
            citation = f"[{doc_id}] {title} ({topic})"
 
            output.append({
                "doc_id":         doc_id,
                "question_title": title,
                "question_text":  question_text,
                "answer_text":    answer_text,
                "answer_snippet": answer_snippet,
                "project_topic":  topic,
                "original_topic": meta.get("original_topic", ""),
                "tier":           int(meta.get("tier", 0)),
                "distance":       round(float(dist), 4),
                "citation":       citation,
            })
 
        return output
 
 
# ================================================================================
# Module-level singleton — imported by the rest of the app
# ================================================================================
 
_retriever: CounselChatRetriever | None = None
 
 
def get_retriever() -> CounselChatRetriever:
    """
    Returns a shared CounselChatRetriever instance.
    Initializes on first call, reuses on subsequent calls.
    """
    global _retriever
    if _retriever is None:
        _retriever = CounselChatRetriever()
    return _retriever
 
 
# ================================================================================
# Standalone test — run as: python rag/retriever.py
# ================================================================================
 
TEST_QUERIES = [
    # Retrieval tests (should return clearly relevant results)
    "my partner and I keep fighting about the same things",
    "I think my boyfriend is cheating on me",
    "how do I start a conversation without it turning into an argument",
    "my husband wants a divorce and I don't know what to do",
    # Edge cases
    "we are growing apart and I feel disconnected",
    "my girlfriend won't communicate with me",
    "how do I get over my ex",
    "I feel unloved and neglected in my relationship",
    # Should still return something (softer signals)
    "is it normal to feel this way about my partner",
    "what does a healthy relationship look like",
]
 
 
def run_test():
    print("Loading retriever...")
    retriever = get_retriever()
    print(f"Collection size: {retriever._collection.count()} documents")
    print()
 
    passed = 0
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"[{i:02d}] Query: {query}")
        results = retriever.retrieve(query, k=3)
 
        if not results:
            print("     ❌ No results returned.")
            print()
            continue
 
        top = results[0]
        dist_ok = top["distance"] < 0.6   # cosine distance < 0.6 = reasonable match
 
        status = "✅" if dist_ok else "⚠️ "
        if dist_ok:
            passed += 1
 
        print(f"     {status} Top result: [{top['doc_id']}] {top['question_title']}")
        print(f"        Topic: {top['project_topic']} | Distance: {top['distance']}")
        print(f"        Snippet: {top['answer_snippet'][:120]}...")
        print(f"        Citation: {top['citation']}")
        print()
 
    print("=" * 55)
    print(f"RETRIEVAL TEST SUMMARY: {passed}/{len(TEST_QUERIES)} passed (distance < 0.6)")
    if passed >= 8:
        print("✅ Retrieval quality looks good.")
    elif passed >= 5:
        print("⚠️  Some queries returning weak matches. Consider tier=1 only filter.")
    else:
        print("❌ Retrieval quality is poor. Check embeddings or rebuild index.")
    print()
    print("Handoff note for Person C/D:")
    print("  from rag.retriever import get_retriever")
    print("  retriever = get_retriever()")
    print("  results   = retriever.retrieve(user_query, k=3)")
 
 
if __name__ == "__main__":
    run_test()
 
