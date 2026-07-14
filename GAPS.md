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

## 2. ~~Whole-paper mode silently truncates long documents from the front~~ — **FIXED 2026-07-12**

**What it was:** `answer_full_doc` / `deck_stream` stuffed all chunks into the context;
Ollama silently dropped the *oldest* tokens (abstract/intro) on overflow, with no warning.
**Fix applied:** `generate.fit_chunks(chunks, num_ctx)` keeps the leading chunks that
fit (front matter survives; truncation is from the END, ours not Ollama's) and reports
the dropped count. `answer_full_doc` prepends a visible `> ⚠ …` warning and returns the
kept chunks (citations stay numbered against what the model saw); `deck_stream` prepends
an HTML-comment warning. `~3 chars/token` estimate with 4096-token headroom. Gate
`tests/test_present.py` step 0 covers the boundary logic (pure, no Ollama). Present gate
PASS. Remaining nuance: truncation still loses the paper's *later* sections silently to
the deck's content — acceptable, and the warning makes it visible.

## 3. ~~SQL filter values are string-interpolated without escaping~~ — **FIXED 2026-07-13**

**What it was:** `store.search` interpolated filter values into `WHERE` clauses
unescaped; a filename/course containing `'` crashed every filtered search.
**Fix applied:** `store._escape()` doubles single quotes (SQL-style); used in the
clause builder. Verified: `course="o'reilly's course"` returns empty instead of
raising; doc_id filtering still exact; `eval_retrieval.py` 20/20. Reuse `_escape`
for any future filter (e.g. `delete_doc`, gap #8).

## 4. ~~Staging cache keyed by filename stem — silent cross-course collisions~~ — **FIXED 2026-07-15**

**What it was:** `parse()` cached at `data/staging/<stem>.json`, so two courses each
holding a `lecture1.pdf` shared one slot — the second silently ingested the first's text.
**Fix applied:** `ingest.cache_path()` keys on the library-relative path with separators
flattened (`cs101/lecture1.pdf` → `cs101__lecture1.json`); falls back to the stem for
ad-hoc parses outside the library. Existing 7 staging files migrated by hand (no stem
collisions existed, verified) so no re-parse was needed; `ingest` confirms cache hits.
Regression check lives in `tests/test_smoke.py` (pure, no Ollama).
**Still open (same bug, different dir):** `present.export_dir()` uses `Path(doc_id).stem`
— two courses with same-named papers share `data/exports/<stem>/`. Left alone
deliberately: the fix makes the user-facing download path uglier
(`classic-papers__attention…/deck.md`), and overwriting your own deck is a milder
failure than indexing wrong text. Revisit if a real collision appears.

## 5. ~~`jobs.status` iterated while another thread mutates it~~ — **FIXED 2026-07-13**

**What it was:** `rows()` iterated the live `status` dict while the worker thread
wrote to it (`RuntimeError: dictionary changed size during iteration` risk on every
2 s UI poll); `_ensure_worker` had a check-then-set race that could start two workers.
**Fix applied:** `rows()` snapshots via `list(status.items())` (C-level, effectively
atomic under the GIL) before iterating; `_ensure_worker` guarded by `_worker_lock`.
Verified with a stress test: 20,000 `rows()` calls against a thread hot-writing and
clearing the dict — zero errors.

## 6. ~~Per-document FTS rebuild + full-table materialization in the ingest hot path~~ — **FIXED 2026-07-15**

**What it was:** (a) the jobs worker rebuilt the whole BM25 index after *every* document;
(b) `existing_doc_ids` / `doc_chunks` / `list_docs` called `table.to_arrow()`, which pulls
every column — including the 1024-float vectors — into memory before selecting.
**Fix applied:** (a) `jobs._run` only calls `ensure_fts` when `_q.empty()` — once per
burst. New rows stay vector-searchable meanwhile; BM25 catches up at burst end.
(b) All three helpers now use lancedb's native empty-query builder
(`table.search().select([...]).where(...).limit(0)`) — projection + filter pushdown, no
new dependency (`to_lance()` would have needed `pylance`). `list_docs` now delegates to
`existing_doc_ids` (dedupes duplicated logic).
**Measured** (synthetic 20k-chunk table, since the real corpus at 306 chunks is too small
to show anything): `doc_chunks` **275 ms → 10 ms (27×)**; `existing_doc_ids` RSS
**+100 MB → +3.5 MB (29×)** with wall clock a wash (45→51 ms — irrelevant next to the
memory). Note: tracemalloc under-reports this badly because Arrow allocates off the
Python heap; RSS is the honest instrument here.

## 7. ~~Test suite is order-dependent and assumes a pre-populated environment~~ — **FIXED 2026-07-15**

**What it was:** Gates assumed an already-built index; `test_present.py` hardcoded a
doc that only existed because someone had once run `test_jobs.py` by hand. A fresh
clone couldn't run them in any order.
**Fix applied:** `tests/fixtures.py` maps doc_id → arxiv URL for the whole test corpus
(6 classic papers + the GRU paper) with `ensure_pdf()` (download) and `ensure()`
(download + ingest what's missing). All four gates call it and now self-heal:
`eval_retrieval` → `EVAL_CORPUS`, `test_present` → GRU + attention, `test_smoke` →
attention, `test_jobs` → GRU with argv now optional. `test_jobs` resets itself via
`store.delete_doc` (gap #8), so it's repeatable rather than one-shot.
**Verified for real:** deleted `gan.pdf` outright (PDF + index rows + staging cache),
then ran `eval_retrieval` cold — it fetched, parsed (27 s), indexed 25 chunks, and
scored 20/20 with zero manual setup. Needs network on first run only.
**Note:** gates deliberately run against the real `data/` store, not a sandbox — they
assert on the same index the app uses. Running `test_jobs` re-parses the GRU fixture
(~30 s) because delete drops its staging cache.

## 8. ~~No way to remove or re-index a document~~ — **FIXED 2026-07-15**

**What it was:** Ingestion was append-only; a corrected PDF re-uploaded under the same
name was silently skipped, and removing a bad document meant wiping the whole index.
**Fix applied:** `store.delete_doc(doc_id)` deletes the doc's chunks (quote-escaped per
gap #3), rebuilds FTS (skipped when the table ends up empty — lance can't index
nothing), and **drops the staging cache** so a re-ingest re-parses the corrected PDF
rather than replaying the stale parse. The source PDF is never touched (per CLAUDE.md's
never-delete rule). Library tab gained a "Remove from index" dropdown + Remove button
whose message spells out that the PDF stays in `data/library/`.
**Verified:** `test_jobs` now covers the whole delete → re-parse → re-index cycle;
`app.remove_fn` exercised directly (guard path, real path, dropdown refresh, restore).
**Re-index recipe:** Remove in the Library tab → replace the PDF on disk → re-upload
(or `uv run python -m ragcore.ingest`).

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
