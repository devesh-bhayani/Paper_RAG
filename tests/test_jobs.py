"""Gate: async ingestion queue + delete/re-index path.

A PDF must walk queued -> parsing -> embedding -> indexed, and store.delete_doc must
remove it cleanly so it can be re-indexed.

Needs Ollama running + bge-m3 pulled. Fetches its fixture on first run.
Run:  python tests/test_jobs.py [path-to.pdf]    (default: the GRU fixture)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import fixtures
from ragcore import jobs, store


def main() -> None:
    if len(sys.argv) > 1:
        pdf = Path(sys.argv[1])
        assert pdf.exists(), f"no such file: {pdf}"
    else:
        pdf = fixtures.ensure_pdf(fixtures.GRU)
    doc_id = f"test-uploads/{pdf.name}"

    # reset: the queue skips anything already indexed, so this test needs a clean slate
    # (this also exercises the delete/re-index path)
    removed = store.delete_doc(doc_id)
    assert doc_id not in store.list_docs(), "delete_doc left the doc in the index"
    print(f"delete_doc OK: removed {removed} chunks for {doc_id}")

    submitted = jobs.submit(pdf, course="test-uploads")
    assert submitted == doc_id, f"submit returned {submitted}, expected {doc_id}"

    seen = []
    for _ in range(600):  # up to 10 min — parsing is the slow stage
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
    assert doc_id in store.list_docs(), "re-indexed doc missing from the index"
    print(f"\nOK: {doc_id} went {' -> '.join(seen)}")


if __name__ == "__main__":
    main()
