"""Grounded generation: retrieved chunks -> numbered-excerpt prompt -> streamed, cited answer.

CLI:  python -m ragcore.generate [--tier daily|code|quality|utility] [--course X]
                                 [--kind prose|code|table|math] [--q "one-shot question"]
"""

import argparse
from collections.abc import Iterator

import ollama

import config
from ragcore import store

REFUSAL = "NOT IN LIBRARY"

SYSTEM = f"""You are a study assistant answering strictly from the provided source excerpts.
Rules:
- Support every claim with the bracketed number of its excerpt, e.g. [2] or [1][3].
- Never use knowledge that is not in the excerpts.
- If the excerpts do not contain the answer, reply exactly: {REFUSAL}
- Be concise and technical."""


# Ollama silently drops the OLDEST prompt tokens on overflow — i.e. the paper's start
# (abstract, intro). We truncate from the END ourselves so that loss is visible and the
# front matter survives. ~3 chars/token is a low estimate → over-counts tokens → safe.
_CHARS_PER_TOKEN = 3
_GEN_HEADROOM = 4096  # tokens reserved for system prompt + question + the answer


def fit_chunks(chunks: list[dict], num_ctx: int) -> tuple[list[dict], int]:
    """Keep the leading chunks (reading order) that fit; return (kept, dropped_count).
    Always keeps at least one chunk."""
    budget_chars = (num_ctx - _GEN_HEADROOM) * _CHARS_PER_TOKEN
    kept: list[dict] = []
    used = 0
    for c in chunks:
        used += len(c["text"]) + 40  # + the "[n] (doc p.X, section)" header line
        if used > budget_chars and kept:
            break
        kept.append(c)
    return kept, len(chunks) - len(kept)


def build_prompt(question: str, chunks: list[dict]) -> str:
    ctx = "\n\n".join(
        f"[{i}] ({c['doc_id']} p.{c['page']}"
        f"{', ' + c['section'] if c['section'] else ''})\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )
    return f"Source excerpts:\n\n{ctx}\n\nQuestion: {question}"


def _chat_stream(tier: str, question: str, chunks: list[dict],
                 num_ctx: int) -> Iterator[str]:
    parts = ollama.chat(
        model=config.TIERS[tier],
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": build_prompt(question, chunks)}],
        options={"num_ctx": num_ctx, "temperature": 0.2},
        think=False,
        stream=True,
    )
    for part in parts:
        yield part["message"]["content"]


def answer(question: str, tier: str = "daily", k: int = 8,
           **filters) -> tuple[Iterator[str], list[dict]]:
    """Returns (token stream, retrieved chunks). Chunks are numbered 1..k in prompt order."""
    chunks = store.search(question, k=k, **filters)

    def stream() -> Iterator[str]:
        if not chunks:
            yield f"{REFUSAL} (nothing retrieved — check filters or ingest more documents)"
            return
        yield from _chat_stream(tier, question, chunks, num_ctx=16384)

    return stream(), chunks


def answer_full_doc(question: str, doc_id: str,
                    tier: str = "daily") -> tuple[Iterator[str], list[dict]]:
    """Whole-paper mode: the entire document goes into context in reading order.
    Returns (stream, kept_chunks) — citations are numbered against kept_chunks."""
    all_chunks = store.doc_chunks(doc_id)
    kept, dropped = fit_chunks(all_chunks, config.FULLDOC_NUM_CTX)

    def stream() -> Iterator[str]:
        if not all_chunks:
            yield f"{REFUSAL} (document not found in index: {doc_id})"
            return
        if dropped:
            yield (f"> ⚠ This paper has {len(all_chunks)} chunks; only the first "
                   f"{len(kept)} fit the context window, so the last {dropped} (later "
                   f"sections) were omitted. Ask a scoped question to reach them.\n\n")
        yield from _chat_stream(tier, question, kept,
                                num_ctx=config.FULLDOC_NUM_CTX)

    return stream(), kept


def main() -> None:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252; papers have Greek

    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="daily", choices=list(config.TIERS))
    ap.add_argument("--course")
    ap.add_argument("--kind", dest="content_kind",
                    choices=["prose", "code", "table", "math"])
    ap.add_argument("--q", help="one-shot question (otherwise interactive loop)")
    args = ap.parse_args()
    filters = {k: v for k, v in
               {"course": args.course, "content_kind": args.content_kind}.items() if v}

    # ponytail: single-turn loop, no chat history — the Gradio UI (Phase 4) owns history
    while True:
        q = args.q or input("\n? ").strip()
        if not q or q in {"exit", "quit"}:
            break
        tokens, chunks = answer(q, tier=args.tier, **filters)
        for t in tokens:
            print(t, end="", flush=True)
        if chunks:
            print("\n\nsources:")
            for i, c in enumerate(chunks, 1):
                print(f"  [{i}] {c['doc_id']} p.{c['page']} [{c['content_kind']}]")
        if args.q:
            break


if __name__ == "__main__":
    main()
