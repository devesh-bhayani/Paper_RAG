# GAPS.md — Honest audit of known weaknesses

Ordered by severity. Each entry: what / where / why it matters / a fix scoped small
enough to execute as a single task. Context: single-user, local-only app — "security"
severities are calibrated to that (a local student tool, not a service).

---

## 1. ~~doc_id embeds Windows path separators~~ — **FIXED 2026-07-12**

**What it was:** `doc_id` used `str(pdf.relative_to(...))` → backslashes on Windows,
breaking the "index carries over to the Mac" migration story.
**Fix applied:** doc_ids are now minted with `.as_posix()` at all four sites
(`ingest.py::ingest_pdf`/`ingest_dir`, `jobs.py::submit`/`_run`); test constants in
`tests/test_present.py` use forward slashes; index wiped and re-ingested (7 docs,
posix doc_ids verified); `eval_retrieval.py` 20/20 and `test_present.py` PASS after.
Note for the Mac migration: nothing left to do here — the index is now
separator-independent.

## 2. Whole-paper mode silently truncates long documents from the front

**What:** `generate.answer_full_doc` and `present.deck_stream` stuff all chunks into a
`FULLDOC_NUM_CTX` (16384 on pc) window. Ollama silently drops the *oldest* prompt
tokens when the prompt exceeds the window — i.e. the paper's beginning (abstract,
intro) vanishes first, with no warning.
**Where:** `ragcore/generate.py::answer_full_doc`, `ragcore/present.py::deck_stream`,
`config.py::FULLDOC_NUM_CTX`.
**Why it matters:** Works fine for ≤20-page papers; a textbook chapter or a long
survey will produce confident answers/decks that never saw the introduction. For a
graded presentation this is a quiet correctness failure.
**Fix (single task):** Estimate tokens (`sum(len(c["text"]) for c in chunks) // 3` is
close enough) in both call sites; if the estimate exceeds `FULLDOC_NUM_CTX - 4096`
(generation headroom), prepend a visible warning line to the stream ("⚠ paper exceeds
context; answers may miss the first N chunks") or fall back to retrieval mode.

## 3. SQL filter values are string-interpolated without escaping

**What:** `store.search` builds `WHERE` clauses as `f"{col} = '{val}'"`. A value
containing an apostrophe breaks the query; deliberate injection is conceivable.
**Where:** `ragcore/store.py::search` (the `clauses` list).
**Why it matters:** Severity here is *correctness*, not really security (values come
from the app's own dropdowns, single local user) — but a PDF named `o'reilly-notes.pdf`
or a course dir `nyu'26` crashes every filtered search, and doc_ids are minted from
arbitrary user filenames via the upload path.
**Fix (single task):** Escape quotes: `val.replace("'", "''")` in the clause builder.
One line. Add a filename-with-apostrophe case to `tests/test_smoke.py` if cheap.

## 4. Staging cache keyed by filename stem — silent cross-course collisions

**What:** `ingest.parse` caches the parsed document at
`data/staging/<pdf_stem>.json`. Two different PDFs named `lecture1.pdf` in two
different course folders share a cache slot: the second one ingests the *first one's
parsed content* without any error.
**Where:** `ragcore/ingest.py::parse`.
**Why it matters:** Wrong text gets indexed under the wrong doc_id, silently. With
multiple courses, generic filenames (`notes.pdf`, `slides.pdf`) are likely.
**Fix (single task):** Key the cache by the sanitized relative path, e.g.
`doc_id.replace("\\", "__").replace("/", "__") + ".json"`. Invalidate old cache by
renaming existing files or just accepting a one-time re-parse. Coordinate with gap #1
so the key is separator-independent.

## 5. `jobs.status` iterated while another thread mutates it

**What:** `jobs.rows()` does `reversed(status.items())` on a plain dict that the worker
thread writes concurrently; the Gradio timer calls `rows()` every 2 s. Python raises
`RuntimeError: dictionary changed size during iteration` if an insert lands mid-iteration.
Also `_ensure_worker` has a check-then-set race: two simultaneous uploads (Gradio
handlers run in a thread pool) can start two worker threads.
**Where:** `ragcore/jobs.py::rows`, `ragcore/jobs.py::_ensure_worker`.
**Why it matters:** Low frequency but real: the status table intermittently errors
exactly when the user is watching an active ingestion — the worst moment. Two workers
would double-ingest a queued file.
**Fix (single task):** `rows()` → snapshot first: `list(status.items())[::-1]`.
`_ensure_worker` → guard with a module-level `threading.Lock`. Four lines total.

## 6. Per-document FTS rebuild + full-table materialization in the ingest hot path

**What:** Two compounding costs: (a) the jobs worker calls `store.ensure_fts` (a full
BM25 index rebuild, `replace=True`) after **every single document**; (b)
`store.existing_doc_ids` and `store.doc_chunks` call `table.to_arrow()`, which
materializes **all columns including the 1024-float vectors** before selecting, and the
worker calls `existing_doc_ids` once per queued document.
**Where:** `ragcore/jobs.py::_run`, `ragcore/store.py::existing_doc_ids`,
`ragcore/store.py::doc_chunks`.
**Why it matters:** Fine at 300 chunks. At textbook scale (tens of thousands of
chunks) ingestion becomes quadratic-ish and each status poll drags hundreds of MB
through RAM. The `ponytail:` comments flag this as deliberate — but the jobs worker
calling it per-document wasn't part of that bargain.
**Fix (single task):** (a) In `_run`, only call `ensure_fts` when `_q.empty()`.
(b) In both store helpers, select columns *before* materializing — LanceDB supports
`table.search().where(...).select([...])` or `table.to_lance().to_table(columns=[...])`;
use whichever the pinned lancedb version supports and keep the ponytail comment.

## 7. Test suite is order-dependent and assumes a pre-populated environment

**What:** All test scripts require: Ollama running, four models pulled, and an already
built index. Worse, `tests/test_present.py` hardcodes `test-uploads\gru-eval.pdf` — a
document that only exists because `tests/test_jobs.py` was once run manually with a
downloaded fixture. A fresh clone cannot run the gates in any order.
**Where:** `tests/test_present.py` (SMALL_DOC), `tests/test_jobs.py` (takes an
arbitrary PDF argv), all tests implicitly.
**Why it matters:** The gates are the project's only regression net. If they can't run
from a clean state, they quietly stop being run.
**Fix (single task):** Add a `tests/conftest-like` helper (plain function, no pytest)
`ensure_fixture()` that downloads `https://arxiv.org/pdf/1412.3555` into
`data/library/test-uploads/gru-eval.pdf` and ingests it if missing; call it at the top
of `test_present.py` and `test_jobs.py` (making test_jobs's argv optional). Document
"requires Ollama + models" at the top of each script.

## 8. No way to remove or re-index a document

**What:** Ingestion is append-only. A corrected PDF re-uploaded under the same name is
*skipped* (idempotency check), and there is no delete path at all — removing a bad
document means wiping `data/lancedb` and re-ingesting everything.
**Where:** absence in `ragcore/store.py` / `app.py` Library tab.
**Why it matters:** First time the user ingests a corrupted or wrong-version PDF,
they'll discover the only fix is a full rebuild (minutes now, hours at textbook scale).
**Fix (single task):** Add `store.delete_doc(doc_id)` using
`table.delete(f"doc_id = '...'")` (escape per gap #3), then `ensure_fts`. Optional
second task: a small "remove document" dropdown+button in the Library tab.

## 9. `doc_type == "code"` is documented but unreachable

**What:** The schema and design docs describe `doc_type ∈ {textbook, paper, code}`, but
`ingest.chunk_doc` only ever assigns `"textbook"` (>150 pages) or `"paper"`. There is
also no ingestion path for source-code files at all, despite "codebases" appearing in
the project pitch.
**Where:** `ragcore/ingest.py::chunk_doc` (the ponytail-commented heuristic),
`ragcore/store.py` schema comment, README/SYSTEM_DESIGN.
**Why it matters:** A doc_type filter for "code" in any future UI would silently match
nothing; the docs over-promise.
**Fix (single task):** Either delete `code` from the documented enum (README,
SYSTEM_DESIGN, store.py comment) — the honest ponytail fix — or add a
`data/library/**/code/` convention later. Pick the deletion unless codebase ingestion
is actually scheduled.

## 10. `gpu_lock` is held for the entire streamed generation

**What:** `app.py::chat_fn` and `deck_fn` hold `jobs.gpu_lock` across the whole token
stream (potentially minutes for a deck). Ingestion embedding — and any second chat
request — blocks for the duration. Gradio's own queue already serializes same-endpoint
requests, so the lock's real marginal effect is pausing ingestion.
**Where:** `app.py::chat_fn`, `app.py::deck_fn`, `ragcore/jobs.py::gpu_lock`.
**Why it matters:** Mostly by design (8 GB card, one model resident), but nobody
documented that a 25-minute-deck generation freezes the Library tab's pipeline, and on
the Mac profile (32 GB) the lock becomes pure overhead.
**Fix (single task):** Document the behavior in PROJECT.md/CLAUDE.md (done) and make
the lock a no-op on the mac profile: `gpu_lock = threading.Lock() if config.PROFILE ==
"pc" else contextlib.nullcontext()` — adjust the two `with` sites accordingly.

## 11. No quality eval for generation or decks — only structure is gated

**What:** Retrieval has a real 20-question accuracy gate. Generation is gated only on
"has citations, refuses off-corpus"; decks only on "has the five section names and
enough separators." Nothing measures whether answers are *faithful* to the cited
chunks or whether the Methodology section is actually deep — the thing the course
grades hardest.
**Where:** `tests/test_generate.py`, `tests/test_present.py`.
**Why it matters:** The 8B model already demonstrably under-delivers on one prompt
instruction (it ignored the 40% Methodology allocation — 2 of 9 slides in the verified
run). Quality regressions will be invisible to the current gates.
**Fix (single task):** Add a cheap proportional check to `test_present.py`: count
slides between `## Methodology` and `## Evaluation` and assert ≥ 25% of content
slides. For faithfulness, a follow-up task: 5 QA pairs in `eval_questions.jsonl`
extended with `expect_keywords`, asserted against the generated answer text.

## 12. Miscellaneous small items

- **Single-turn chat**: `chat_fn` ignores Gradio's `history`; every question is
  independent. Follow-ups like "explain that more simply" silently lack context.
  Documented nowhere user-visible. *Fix: one line in the README; real fix (include
  history in the prompt) is a design decision for later.*
- **`utility` tier is pulled but unused** — `config.TIERS["pc"]["utility"]` (qwen3:4b)
  has no code path (query rewriting was deferred to Phase 5). Harmless; either delete
  the tier or leave with a comment. `bench.py` still references qwen3:14b which may
  not be pulled — it skips gracefully by design.
- **README env-var instructions are bash-only** (`export ...`) while the primary
  platform is Windows; the Windows path is a parenthetical. *Fix: add the two-line
  PowerShell/`setx` equivalent.*
- **HF unauthenticated-rate-limit warning** during chunker tokenizer fetch — cosmetic;
  optionally document `HF_TOKEN` in README.
- **`data/exports` collides on paper stems** the same way staging does (gap #4), e.g.
  two courses both containing `notes.pdf` share one export dir. Same fix pattern.
- **No CI**: gates run only when someone remembers. A GitHub Action can't run them
  (needs Ollama + GPU); a pre-push hook running `eval_retrieval.py` locally is the
  practical option. *Fix: document as a manual pre-release checklist in CLAUDE.md
  (done) rather than pretending CI exists.*
- **Secrets**: none in the repo (verified — no keys, tokens, or URLs beyond arxiv/
  pytorch indexes). GitHub auth lives in the user's keyring, outside the repo. ✔
