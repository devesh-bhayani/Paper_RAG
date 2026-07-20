"""Phase 3 gate: citations map to real chunks; clean refusal off-corpus;
answers contain the paper's actual facts (faithfulness keywords).

Needs Ollama running + models pulled. Fetches/ingests its fixtures on first run.
Run:  python tests/test_generate.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import fixtures

from ragcore import generate


def run(question: str) -> tuple[str, list[dict]]:
    tokens, chunks = generate.answer(question)
    return "".join(tokens), chunks


def main() -> None:
    fixtures.ensure(fixtures.EVAL_CORPUS)

    # 1. on-corpus: must cite, citations must point at real chunk numbers
    text, chunks = run("Why is dot-product attention scaled by the square root "
                       "of the key dimension?")
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    assert cited, f"no [n] citations in answer:\n{text}"
    assert all(1 <= n <= len(chunks) for n in cited), \
        f"citation out of range: {cited} vs {len(chunks)} chunks"
    assert generate.REFUSAL not in text, f"refused an on-corpus question:\n{text}"
    print(f"on-corpus OK: cited {sorted(cited)} of {len(chunks)} chunks")

    # 2. off-corpus: must refuse, not hallucinate
    text, _ = run("What is the capital of France?")
    assert generate.REFUSAL in text, f"failed to refuse off-corpus question:\n{text}"
    print("off-corpus OK: refused cleanly")

    # 3. faithfulness: answers must contain the papers' actual facts — keywords are
    # numbers/terms stated in the paper but absent from the question (no echo credit)
    questions = [json.loads(line) for line in
                 (Path(__file__).parent / "eval_questions.jsonl").read_text().splitlines()
                 if line.strip()]
    for item in (q for q in questions if "expect_keywords" in q):
        text, _ = run(item["q"])
        low = text.lower()
        missing = [k for k in item["expect_keywords"] if k.lower() not in low]
        assert not missing, (f"unfaithful answer — missing {missing} "
                             f"for {item['q']!r}:\n{text[:400]}")
        print(f"faithful OK [{item['expect_doc']:>9}]: {item['expect_keywords']}")

    print("\nPhase 3 gate: PASS")


if __name__ == "__main__":
    main()
