# PROJECT.md — What this is and how it fits together

## What this application is

PAPER_RAG is a **100% offline RAG (retrieval-augmented generation) assistant for one
person**: a first-semester NYU graduate student in "Efficient AI and Hardware
Accelerators". It ingests academic PDFs (papers, textbooks), indexes them locally, and
answers questions about them with citations — no cloud APIs, no telemetry, everything
runs on the user's own GPU via Ollama.

It has two jobs:

1. **Study**: ask questions across a course's papers, or scope to one paper — including
   "whole-paper mode" where an entire paper goes into the LLM's context at once, which
   handles summarize/walk-me-through questions that top-k retrieval fundamentally can't.
2. **Present**: the user must give a graded in-class paper presentation. The Present tab
   generates a Marp markdown slide deck following the professor's mandatory format
   (Introduction → Background → Methodology → Evaluation → Your Thoughts & Discussion,
   graded 3/2/3/2 on Clarity / Structure & Flow / Depth of Analysis / Discussion), with
   grounded speaker notes, page references, presenter-handoff markers (3-person talks),
   and the paper's actual figures extracted as PNGs.

There is no server deployment, no multi-user story, no auth. It is a local tool with a
local web UI at `http://127.0.0.1:7860`.

## Hardware context (this matters more than usual)

The code runs on **two hardware profiles**, selected by `PROFILE` in `config.py`:

- **`pc` (active now)**: Windows 11, RTX 4060 Ti with **8 GB VRAM**, CUDA. The 8 GB
  ceiling drives many decisions: model sizes, context limits, and the GPU lock.
- **`mac` (planned ~Aug 2026)**: MacBook Air M5, 32 GB unified memory, fanless, Metal.
  Migration is intended to be: copy repo + `data/`, pull bigger models, flip `PROFILE`.
  (See GAPS.md #1 — one real blocker exists for the "index carries over" claim.)

Every measured number in the code comments (context sizes, thermal pacing) came from
actual benchmarks on the pc profile — trust them over intuition.

## Tech stack and why

| Piece | Why it was chosen |
|---|---|
| **Ollama** (LLM + embeddings) | One runtime serves both generation (qwen3 family) and embeddings (bge-m3), GPU-accelerated on both CUDA and Metal automatically. Keeps PyTorch out of the *serving* path entirely. |
| **Docling** (PDF parsing) | The only local parser that reliably separates prose / code blocks / tables / math and labels them (`content_kind`). Its layout models run on GPU (CUDA now, MPS later). |
| **LanceDB** (vector store) | Serverless and embedded — no daemon process, near-zero idle RAM. Natively does **both** vector search and BM25 full-text, fused with reciprocal-rank fusion, plus SQL-ish metadata filters. One dependency replaces three. |
| **Gradio 5** (UI) | Streaming chat, async queue, file upload, dark mode — for free. **Pinned `>=5,<6`**: Gradio 6 renamed half the API and the chat panel rendered blank (debugged, not worth fighting a 3-month-old major release). |
| **uv** | Env management; also pins torch to the CUDA index (see gotchas). |
| **No LangChain / LlamaIndex** | Deliberate. The whole RAG core is ~600 lines of direct library calls. Frameworks would add abstraction to debug without adding capability at this scale. |

Model tiers (per profile, in `config.TIERS`): `daily` (qwen3:8b on pc) for answering,
`code` (qwen2.5-coder:7b), `utility` (qwen3:4b, currently unused by any code path but
pulled and available), `quality` (qwen3:14b — runs with partial CPU offload on 8 GB,
"patience tier"). Embeddings are **always bge-m3, 1024-dim** — this is the one invariant
that makes the index portable across machines.

## Architecture

```
INGESTION (batch CLI: python -m ragcore.ingest;  async UI path: jobs.py worker thread)

 data/library/<course>/*.pdf
        │
        ▼
 ingest.parse() ── Docling (layout+table+code+formula models, GPU) ──► DoclingDocument
        │                                            │
        │                              cache JSON in data/staging/<stem>.json
        ▼                              (parse once; re-chunk/re-embed never re-parses)
 ingest.chunk_doc() ── HybridChunker (bge-m3 tokenizer, 512 tok max, structure-aware)
        │              → records: seq, text, doc_id, course, topic, doc_type,
        │                          content_kind{prose|code|table|math}, section, page
        ▼
 embed.embed_texts() ── Ollama /api/embed, bge-m3, batches of 32, one retry
        │
        ▼
 store.add_chunks() ── LanceDB table "chunks" (data/lancedb/)
        └── store.ensure_fts() — BM25 index (native lance FTS; tantivy has no Win wheels)

QUERY (Chat tab / generate.py CLI)

 question ──► store.search(): hybrid = vector(bge-m3) + BM25, RRF fusion,
        │        metadata pre-filters (course / doc_type / content_kind / doc_id)
        │     OR store.doc_chunks(doc_id): ALL chunks of one paper, ordered by seq
        ▼
 generate.build_prompt(): numbered excerpts [1..k] + question
        ▼
 ollama.chat (qwen3, think=False, streaming) ──► answer with [n] citations
        └── UI appends a Sources footer mapping [n] → doc/page/kind

PRESENT (Present tab / present.py)

 present.extract_figures(doc_id) ── one-time Docling re-parse with picture/page images
        │                            → data/exports/<stem>/figures/*.png + manifest.tsv
        ▼
 present.deck_stream(doc_id) ── whole paper (doc_chunks) + DECK_SYSTEM prompt
        │                        (professor's rubric baked in) ──► Marp markdown
        ▼
 present.save_deck() ── appends a deterministic figure-appendix if the model didn't
                         embed images (small models ignore embed instructions)
                         → data/exports/<stem>/deck.md
```

Concurrency model: **one GPU, one lock.** `jobs.gpu_lock` (a plain `threading.Lock`)
serializes chat generation, deck generation, and ingestion *embedding* (parsing doesn't
take the lock — it's a different model and mostly bounded by its own GPU/CPU work).
`OLLAMA_MAX_LOADED_MODELS=1` means the embedder and the LLM swap in and out of VRAM
(~2 s per swap); that is deliberate — both resident would spill weights to system RAM
and slow every token.

## Key design decisions (and their reasoning)

1. **bge-m3 is frozen.** Changing the embedding model invalidates every vector in
   `data/lancedb`. The index built on the PC is meant to move to the Mac untouched.
2. **`seq` column = reading order.** Whole-paper mode reconstructs the paper by sorting
   chunks on `seq`. It was added in a schema migration (wipe + re-ingest; the staging
   cache made that cheap).
3. **Staging cache is the expensive artifact.** Parsing is minutes/paper (GPU) or tens
   of minutes (CPU); everything downstream is seconds. `data/staging/*.json` is
   therefore the thing that makes schema changes and re-chunking painless.
4. **Measured context ceilings.** `FULLDOC_NUM_CTX = 16384` on pc because 20480 and
   24576 both spill to CPU on the 8 GB card (measured via `ollama ps`). Mac gets 32768.
5. **Every phase shipped with a gate script** (`tests/`). They are plain
   `assert`-based scripts run directly, not a pytest suite. The retrieval gate
   (`eval_retrieval.py`, 20 questions) is the regression baseline: 20/20 top-1 as of
   2026-07-07. The rule from SYSTEM_DESIGN.md: quality upgrades (reranker, query
   rewriting) are **banned until this eval shows a deficit**.
6. **Rubric-driven deck prompt.** `present.DECK_SYSTEM` encodes the professor's exact
   section arc, point weights, ~40% slide allocation to Methodology, and HANDOFF
   markers. This is course-critical content, not stylistic prose.
7. **Deterministic beats prompted.** When the 8B model ignored "embed at least two
   figures", the fix was code (`save_deck` appends figure slides) rather than more
   prompt engineering. Follow that instinct when extending.
8. **doc_id = path relative to `data/library`** (e.g. `classic-papers\attention-is-all-you-need.pdf`).
   It is the primary key for idempotent ingestion, filtering, and whole-paper lookup.

## Critical paths — what's load-bearing

- **`ragcore/store.py` `SCHEMA`**: every producer (ingest) and consumer (search,
  doc_chunks, UI) depends on it. Any change requires wiping `data/lancedb`,
  re-ingesting, and checking `tests/eval_retrieval.py` still passes.
- **`config.py`**: the single source of truth. `EMBED_MODEL`, `PROFILE`,
  `FULLDOC_NUM_CTX` have physical consequences (index validity, VRAM spill).
- **The citation contract**: `generate.build_prompt` numbers chunks 1..k; the model
  cites `[n]`; the UI footer and the tests map `[n]` back to the same list order.
  Reordering chunks between prompt-build and display breaks citations silently.
- **`pyproject.toml` torch block**: `[tool.uv]` restricts resolution to win32 and pins
  torch/torchvision to the **cu130** index. "Simplifying" this reverts Docling to
  CPU-only torch (6× slower parsing) or breaks the lock entirely.

Safe to change casually: UI layout in `app.py`, prompt *wording* (not structure) in
`generate.SYSTEM`, `bench.py`, talk-length presets, batch sizes.

## Things that will trip up someone new

- **The app must be run from the repo root** (`uv run python app.py`,
  `uv run python -m ragcore.ingest`). Module resolution assumes it; the tests insert
  the root onto `sys.path` themselves.
- **Ollama env vars** (`OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`,
  `OLLAMA_MAX_LOADED_MODELS=1`) are set at the Windows *user* level; the Ollama tray
  app must be restarted to pick up changes. If decode speed suddenly halves, check
  `ollama ps` for CPU spill first.
- **First run needs network once**: Docling downloads its layout/OCR models, and
  HybridChunker fetches the `BAAI/bge-m3` tokenizer from HuggingFace. After that,
  everything is offline.
- **`data/` is gitignored but not disposable-uniformly**: `lancedb/`, `staging/`,
  `exports/` are rebuildable; **`library/` is the user's source PDFs — never delete**.
- **qwen3 runs with `think=False` everywhere.** Removing it makes answers slower and
  leaks `<think>` traces into the UI.
- **SYSTEM_DESIGN.md is the build log** — every phase has a dated results block with
  measured numbers (tok/s, parse times, gate outcomes). Read it before re-benchmarking
  anything; the answer is probably already recorded there.
