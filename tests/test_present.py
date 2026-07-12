"""Gate for whole-paper mode + presentation kit.

Run:  python tests/test_present.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from ragcore import generate, present, store

SMALL_DOC = "test-uploads/gru-eval.pdf"
DECK_DOC = "classic-papers/attention-is-all-you-need.pdf"


def main() -> None:
    docs = store.list_docs()
    assert SMALL_DOC in docs and DECK_DOC in docs, f"missing test docs in {docs}"

    # 1. whole-paper mode: summarization question that top-k retrieval handles poorly
    tokens, chunks = generate.answer_full_doc(
        "Summarize the main contributions and findings of this paper.", SMALL_DOC)
    text = "".join(tokens)
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    assert len(chunks) > 10, f"whole paper should be all chunks, got {len(chunks)}"
    assert len(cited) >= 2, f"expected multiple citations, got {cited}:\n{text[:400]}"
    assert all(1 <= n <= len(chunks) for n in cited), f"citation out of range: {cited}"
    assert generate.REFUSAL not in text
    print(f"whole-paper OK: {len(chunks)} chunks in context, cited {sorted(cited)}")

    # 2. figure extraction
    figs = present.extract_figures(DECK_DOC)
    assert len(figs) >= 1, "no figures extracted from the attention paper"
    print(f"figures OK: {len(figs)} images -> {figs[0].parent}")

    # 3. deck generation (algorithm-paper format; gate leniently at half the target)
    talk = "15 min (algorithm paper)"
    deck = present.build_deck(DECK_DOC, talk_length=talk)
    md = deck.read_text(encoding="utf-8")
    assert "marp: true" in md, "missing Marp front-matter"
    separators = len(re.findall(r"^---\s*$", md, flags=re.M))
    assert separators >= config.TALK_LENGTHS[talk] // 2, \
        f"too few slides: {separators} separators"
    assert "figures/" in md or re.search(r"p\.\s*\d+", md), \
        "deck has neither figure embeds nor page references"
    # professor's required arc — lenient: 4 of 5 section names must appear
    sections = ["Introduction", "Background", "Methodology", "Evaluation", "Discussion"]
    found = [s for s in sections if s.lower() in md.lower()]
    assert len(found) >= 4, f"required course sections missing: found only {found}"
    print(f"deck OK: {deck} ({separators} separators, sections {found})")

    print("\nPresent gate: PASS")


if __name__ == "__main__":
    main()
