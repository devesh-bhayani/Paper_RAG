"""Phase 2 gate: expected source doc in top-5 for >= 16/20 questions.

Run:  python tests/eval_retrieval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ragcore import store

GATE = 16
K = 5


def main() -> None:
    questions = [json.loads(line) for line in
                 (Path(__file__).parent / "eval_questions.jsonl").read_text().splitlines()
                 if line.strip()]
    hits = 0
    for item in questions:
        results = store.search(item["q"], k=K)
        ok = any(item["expect_doc"] in r["doc_id"] for r in results)
        hits += ok
        top = results[0]["doc_id"] if results else "-"
        print(f"{'HIT ' if ok else 'MISS'}  [{item['expect_doc']:>9}]  top1={top:<45}  {item['q'][:60]}")

    print(f"\n{hits}/{len(questions)} in top-{K}  |  GATE >= {GATE}: "
          f"{'PASS' if hits >= GATE else 'FAIL'}")
    sys.exit(0 if hits >= GATE else 1)


if __name__ == "__main__":
    main()
