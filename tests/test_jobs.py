"""Phase 4 gate (queue half): submit a new PDF, watch it move through states to indexed.

Run:  python tests/test_jobs.py <path-to-new.pdf>
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ragcore import jobs


def main() -> None:
    pdf = Path(sys.argv[1])
    assert pdf.exists(), f"no such file: {pdf}"

    doc_id = jobs.submit(pdf, course="test-uploads")
    seen = []
    for _ in range(600):  # up to 10 min — parsing is the slow stage on CPU torch
        state = jobs.status[doc_id]
        if not seen or state != seen[-1]:
            seen.append(state)
            print(f"  {state}")
        if state.startswith(("indexed", "failed", "skipped")):
            break
        time.sleep(1)

    assert seen[0] == "queued", f"first state was {seen[0]}"
    assert any(s == "parsing" for s in seen), f"never hit parsing: {seen}"
    assert seen[-1].startswith("indexed"), f"final state: {seen[-1]}"
    print(f"\nOK: {doc_id} went {' -> '.join(seen)}")


if __name__ == "__main__":
    main()
