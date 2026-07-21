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

## 9. ~~`doc_type == "code"` is documented but unreachable~~ — **FIXED 2026-07-18**

**What it was:** Docs promised `doc_type ∈ {textbook, paper, code}` and "codebases" in
the pitch, but no code path ever produced `"code"` and no source-file ingestion exists.
**Fix applied:** The honest deletion — enum is now `{textbook, paper}` in
`store.py`'s schema comment and SYSTEM_DESIGN.md; the pitch line now says explicitly
that code *chunks* inside PDFs are labeled (`content_kind=code`, which works) while
source-code *file* ingestion is not built. `content_kind` handling untouched.
Revisit only if codebase ingestion actually gets scheduled.

## 10. ~~`gpu_lock` held for the entire streamed generation~~ — **FIXED 2026-07-18**

**What it was:** chat/deck streaming held `gpu_lock` for minutes, pausing ingestion
embedding — correct on the 8 GB pc (one resident model) but pure overhead on the
32 GB mac, and undocumented either way.
**Fix applied:** `gpu_lock` is now profile-dependent: a real `threading.Lock` on pc,
`contextlib.nullcontext()` on mac (both models fit resident there). Call sites
unchanged — they keep wrapping, the mac lock just doesn't exclude. Behavior documented
in CLAUDE.md ("a long deck generation pauses ingestion embedding on pc — deliberate").
**Verified:** pc path mutually excludes (non-blocking acquire fails while held); mac
path (config reload) allows nested/concurrent entry; `test_jobs` full cycle passes.

## 11. ~~No quality eval for generation or decks — only structure gated~~ — **FIXED 2026-07-20**

**What it was:** Nothing measured answer faithfulness or Methodology depth (rubric's
heaviest criterion); the 8B demonstrably ignored the 40% allocation (2/9 slides).
**Fix applied:**
- **Deck depth:** DECK_SYSTEM now demands an explicit slide count
  (`max(3, 30% of n_slides)`) instead of a percentage — concrete numbers beat
  proportions for small models. `test_present.py` parses the deck, counts the
  Methodology span (figure-appendix excluded), asserts ≥ max(2, 25% of content
  slides). Measured effect: 2/9 (22%) before → **4/12 (33%)** after.
- **Faithfulness:** 5 questions in `eval_questions.jsonl` carry `expect_keywords` —
  facts stated in the paper but absent from the question (no echo credit): gradient,
  28.4 BLEU, 152 layers, 0.9/0.999, queen. `test_generate.py` asserts each appears in
  the generated answer. One question needed rewording twice: contained "queen" (echo),
  then too indirect (model answered correctly without the word). Lesson: keyword
  questions must *ask for* the fact directly without *containing* it.
**Verified:** 5/5 faithful, retrieval still 20/20 after rewording, present gate PASS.

## 12. ~~Miscellaneous small items~~ — **FIXED 2026-07-21**

- **`data/exports` stem collision** (the only code bug in this batch): `export_dir`
  used `Path(doc_id).stem`, so two courses each holding a `notes.pdf` overwrote each
  other's deck and figures. Now flattens the full doc_id like the staging cache
  (`classic-papers__attention-is-all-you-need`). Existing export dir renamed in place —
  10 figures and the deck survived, no re-parse. Verified + present gate PASS.
  *(Earlier deferred as "ugly path beats mild data loss" — reversed: consistency with
  staging is worth more than a pretty directory name nobody browses.)*
- **Single-turn chat** documented in README ("Good to know": each turn is an
  independent retrieval, follow-ups like *"explain that more simply"* won't work).
  Real fix (history in the prompt + query rewriting) remains a design decision — it
  costs context window on an 8 GB card and complicates the citation contract.
- **Windows env vars**: README now has the `setx` block plus the restart-the-tray-app
  note, instead of bash `export` with a Windows parenthetical.
- **`HF_TOKEN`** documented in README as the way to silence the unauthenticated
  rate-limit warning.
- **`utility` tier**: not dead — it's selectable in the UI tier dropdown (`list(TIERS)`
  populates it). Comment in `config.py` says so; README explains when to pick each tier.
- **No CI**: CLAUDE.md now carries a per-blast-radius pre-push table (schema/retrieval
  → never skip `eval_retrieval.py`; docs → nothing), and the README states plainly that
  no CI exists and why (a runner has no GPU and no Ollama).
- **`test_jobs.py <pdf>`** in the README was stale since gap #7 made the argument
  optional — corrected.
- **Secrets**: none in the repo (verified — no keys/tokens; only arxiv + pytorch index
  URLs). GitHub auth lives in the user's keyring, outside the repo. ✔
