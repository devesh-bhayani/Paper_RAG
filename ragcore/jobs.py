"""Async ingestion queue: worker thread + per-document status for the UI.

States: queued -> parsing -> embedding -> indexed (n chunks) | failed: <err> | skipped
"""

import queue
import shutil
import threading
from pathlib import Path

import config
from ragcore import ingest, store

# generation and embedding share one 8 GB GPU — hold this around either
gpu_lock = threading.Lock()

status: dict[str, str] = {}  # doc_id -> state, insertion-ordered
_q: queue.Queue[Path] = queue.Queue()
_worker_started = False


def submit(upload_path: Path, course: str) -> str:
    """Copy an uploaded PDF into the library and queue it. Returns doc_id."""
    course = course.strip() or "uncategorized"
    dest = config.LIBRARY_DIR / course / upload_path.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if upload_path.resolve() != dest.resolve():
        shutil.copy2(upload_path, dest)
    doc_id = str(dest.relative_to(config.LIBRARY_DIR))
    status[doc_id] = "queued"
    _ensure_worker()
    _q.put(dest)
    return doc_id


def _ensure_worker() -> None:
    global _worker_started
    if not _worker_started:
        threading.Thread(target=_run, daemon=True).start()
        _worker_started = True


def _run() -> None:
    table = store.open_table()
    while True:
        pdf = _q.get()
        doc_id = str(pdf.relative_to(config.LIBRARY_DIR))
        try:
            if doc_id in store.existing_doc_ids(table):
                status[doc_id] = "skipped (already indexed)"
                continue
            n = ingest.ingest_pdf(
                table, pdf,
                on_stage=lambda s: status.__setitem__(doc_id, s),
                gpu_lock=gpu_lock,
            )
            store.ensure_fts(table)
            status[doc_id] = f"indexed ({n} chunks)"
        except Exception as e:  # keep the worker alive for the next doc
            status[doc_id] = f"failed: {e}"


def rows() -> list[list[str]]:
    """Status table for the UI, newest first."""
    return [[doc_id, state] for doc_id, state in reversed(status.items())]
