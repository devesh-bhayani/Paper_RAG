"""Phase 1 gate: ingest whatever is in data/library, assert chunks landed sanely.

Run:  python tests/test_smoke.py
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from ragcore import ingest, store


def main() -> None:
    pdfs = list(config.LIBRARY_DIR.rglob("*.pdf"))
    assert pdfs, f"put at least one PDF under {config.LIBRARY_DIR} first"

    ingest.ingest_dir(config.LIBRARY_DIR)

    table = store.open_table()
    rows = table.to_arrow()
    assert rows.num_rows > 0, "no chunks indexed"

    kinds = Counter(rows.column("content_kind").to_pylist())
    courses = set(rows.column("course").to_pylist())
    vec_len = len(rows.column("vector")[0].as_py())

    assert set(kinds) <= {"prose", "code", "table", "math"}, f"bad kinds: {kinds}"
    assert vec_len == config.EMBED_DIM, f"vector dim {vec_len} != {config.EMBED_DIM}"
    assert all(courses), "empty course metadata"

    print(f"\nOK: {rows.num_rows} chunks | kinds {dict(kinds)} | courses {courses}")


if __name__ == "__main__":
    main()
