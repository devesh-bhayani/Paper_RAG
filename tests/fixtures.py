"""Test corpus: download + ingest on demand so the gates run from a clean clone.

Needs network the first time only (arxiv). PDFs land in data/library/ and are ingested
into the same index the app uses — the gates are not sandboxed, by design: they assert
against the real store.
"""

import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from ragcore import ingest, store

# doc_id -> arxiv pdf url
PAPERS = {
    "classic-papers/attention-is-all-you-need.pdf": "https://arxiv.org/pdf/1706.03762",
    "classic-papers/bert.pdf": "https://arxiv.org/pdf/1810.04805",
    "classic-papers/resnet.pdf": "https://arxiv.org/pdf/1512.03385",
    "classic-papers/gan.pdf": "https://arxiv.org/pdf/1406.2661",
    "classic-papers/adam.pdf": "https://arxiv.org/pdf/1412.6980",
    "classic-papers/word2vec.pdf": "https://arxiv.org/pdf/1301.3781",
    "test-uploads/gru-eval.pdf": "https://arxiv.org/pdf/1412.3555",
}

GRU = "test-uploads/gru-eval.pdf"                              # small: 9 pages
ATTENTION = "classic-papers/attention-is-all-you-need.pdf"     # has figures + tables
EVAL_CORPUS = [d for d in PAPERS if d.startswith("classic-papers/")]


def ensure_pdf(doc_id: str) -> Path:
    """Download the PDF into data/library if missing. Returns its path."""
    dest = config.LIBRARY_DIR / doc_id
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"fetching {doc_id} ...", flush=True)
        req = urllib.request.Request(PAPERS[doc_id],
                                     headers={"User-Agent": "paper-rag-tests"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
    return dest


def ensure(doc_ids: list[str]) -> None:
    """Download + ingest any of these not already indexed. First run parses (slow)."""
    missing = [d for d in doc_ids if d not in store.existing_doc_ids(store.open_table())]
    if not missing:
        return
    for doc_id in missing:
        ensure_pdf(doc_id)
    print(f"ingesting {len(missing)} fixture(s): {missing}", flush=True)
    ingest.ingest_dir(config.LIBRARY_DIR)  # picks up everything new + rebuilds FTS
