"""Gate for whole-paper mode + presentation kit.

Needs Ollama running + models pulled. Fetches/ingests its fixtures on first run.
Run:  python tests/test_present.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import fixtures

import config
from ragcore import generate, present, store

SMALL_DOC = fixtures.GRU
DECK_DOC = fixtures.ATTENTION


def main() -> None:
    # 0. fit_chunks (pure, no Ollama): overflow drops from the end, keeps leading prefix
    small = [{"text": "x" * 1000} for _ in range(10)]
    kept, dropped = generate.fit_chunks(small, config.FULLDOC_NUM_CTX)
    assert dropped == 0 and len(kept) == 10, (len(kept), dropped)
    huge = [{"text": "x" * 4000} for _ in range(100)]
    kept, dropped = generate.fit_chunks(huge, config.FULLDOC_NUM_CTX)
    assert dropped > 0 and 1 <= len(kept) < 100 and kept == huge[:len(kept)], \
        f"overflow should keep a leading prefix: kept {len(kept)} dropped {dropped}"
    assert generate.fit_chunks([{"text": "x" * 10**6}], 16384) == \
        ([{"text": "x" * 10**6}], 0), "a single oversized chunk must still be kept"
    print(f"fit_chunks OK: full paper kept, huge paper kept {len(kept)} dropped {dropped}")

    fixtures.ensure([SMALL_DOC, DECK_DOC])
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

    # 3. deck generation (ML-paper format; gate leniently at half the cap)
    talk = config.DEFAULT_TALK
    cap = config.TALK_LENGTHS[talk]
    deck = present.build_deck(DECK_DOC, talk_length=talk)
    md = deck.read_text(encoding="utf-8")
    assert "marp: true" in md, "missing Marp front-matter"
    # 3-presenter talk: deterministic handoff markers must fire (unit test asserts
    # exactly 2; here >=1 tolerates model section-naming variance in the gate deck)
    assert md.count("HANDOFF") >= 1, "no HANDOFF markers inserted"
    separators = len(re.findall(r"^---\s*$", md, flags=re.M))
    assert separators >= cap // 2, f"too few slides: {separators} separators"
    # the cap is graded. Small models overshoot it, so we warn rather than trim
    # (trimming would silently delete graded Methodology/Discussion content):
    # the deck must be within cap, or carry a visible OVER CAP warning.
    n_slides = len(re.findall(r"^## ", md, flags=re.M))
    assert n_slides <= cap or "OVER CAP" in md, \
        f"deck exceeds the {cap}-slide cap ({n_slides}) without warning the user"
    # our own figure appendix must never be what pushes it over
    appendix = md.count("](figures/") if "OVER CAP" not in md else 0
    assert n_slides - appendix <= cap or "OVER CAP" in md, "appendix broke the cap"
    assert "figures/" in md or re.search(r"p\.\s*\d+", md), \
        "deck has neither figure embeds nor page references"
    # professor's required arc — lenient: 4 of 5 section names must appear
    sections = ["Introduction", "Background", "Methodology", "Evaluation", "Discussion"]
    found = [s for s in sections if s.lower() in md.lower()]
    assert len(found) >= 4, f"required course sections missing: found only {found}"

    # methodology depth — the rubric's heaviest criterion (3 pts): the Methodology
    # span must be >= max(2, 25%) of content slides (figure-appendix excluded)
    parts = [p.strip() for p in re.split(r"^---\s*$", md, flags=re.M)
             if p.strip().startswith("## ")]
    content = [s for s in parts if not all(
        line.strip().startswith("![](") for line in s.splitlines()[1:] if line.strip())]
    titles = [s.splitlines()[0] for s in content]
    meth_start = next(i for i, t in enumerate(titles) if "methodology" in t.lower())
    meth_end = next((i for i in range(meth_start + 1, len(titles))
                     if re.search(r"evaluation|thoughts|discussion", titles[i], re.I)),
                    len(content))
    meth_count = meth_end - meth_start
    need = max(2, round(0.25 * len(content)))
    assert meth_count >= need, \
        f"Methodology too thin: {meth_count} of {len(content)} content slides " \
        f"(need >= {need}); titles: {titles}"
    print(f"deck OK: {deck} ({separators} separators, sections {found}, "
          f"methodology {meth_count}/{len(content)} slides)")

    print("\nPresent gate: PASS")


if __name__ == "__main__":
    main()
