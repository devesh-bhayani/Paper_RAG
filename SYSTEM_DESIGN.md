# PAPER_RAG — System Design

100% offline RAG assistant for dense CS textbooks, papers, and codebases.

**Two hardware profiles, one codebase:**
- `pc` (now): Windows, RTX 4060 Ti 8 GB VRAM, CUDA. Development + daily use until the Mac arrives.
- `mac` (from ~2026-08): MacBook Air M5 (10C CPU / 10C GPU), 32 GB unified memory, fanless, MPS/Metal.

Every component below (Ollama, LanceDB, Docling, Gradio) runs identically on both.
The profile lives in `config.py` and only selects model names, context sizes, and
whether the thermal rest loop is active. Migration to the Mac = copy the repo +
`data/`, pull the bigger models, flip `PROFILE`.

## 0. Design philosophy

The entire RAG core is ~500 lines of plain Python. No LangChain, no LlamaIndex, no
orchestration framework — every component below is a direct library call, and frameworks
would only add abstraction layers you'd debug at 3am. One inference runtime (Ollama)
serves both the LLM and the embeddings, so PyTorch enters the project only through
Docling's layout models.

## 1. Stack decisions

| Concern | Choice | Why (and what was rejected) |
|---|---|---|
| PDF parsing | **Docling** (with code + formula enrichment) | Only local parser that reliably separates prose / code blocks / tables / math. Layout + TableFormer models run on MPS. Rejected: PyMuPDF4LLM (fast but merges code into prose), Nougat (slow, math-only bias), unstructured (server-oriented). |
| Chunking | Docling **HybridChunker** (structure-aware) | Respects section boundaries; keeps code blocks and tables atomic. No custom chunker needed. |
| Embeddings | **Ollama `bge-m3`** (1024-dim, 8k ctx) | Strong on academic text + code, runs Metal-accelerated in the same runtime as the LLM → no sentence-transformers/PyTorch serving stack. |
| Vector store | **LanceDB** | Serverless, embedded, disk-based columnar — near-zero idle RAM. Native metadata filtering (`course`, `topic`, `doc_type`) AND native BM25 full-text (tantivy) → hybrid search with RRF in one dependency. Rejected: Chroma (weaker FTS), Qdrant/Milvus (server processes), sqlite-vec (manual hybrid plumbing). |
| LLM runtime | **Ollama** | llama.cpp with Metal compiled in, model lifecycle management, keep_alive control, OpenAI-compatible API. MLX (`mlx-lm`) is the Phase-5 experiment — M5's per-core GPU neural accelerators give MLX large prefill speedups, which matters for long RAG contexts. |
| UI | **Gradio 5** (Blocks) | Dark-mode native, async event loop + built-in queue for ingestion job states, streaming chat component. Rejected: Streamlit (full-script rerun fights async state), FastAPI+HTMX (a week of work Gradio gives for free). |
| Config | one `config.py` | Values that never change at runtime don't need YAML. |

## 2a. VRAM budget — `pc` profile (RTX 4060 Ti, 8 GB)

Discrete VRAM, no unified-memory tricks: what doesn't fit in 8 GB gets offloaded to
system RAM over PCIe and decode speed falls off a cliff. Budget so the active model +
KV cache stay fully on-GPU:

| Resident at query time | Size |
|---|---|
| CUDA context + display | ~0.6–1 GB |
| LLM weights (pc tier below) | ~5 GB |
| KV cache (16k ctx, q8_0, flash attn) | ~1.2 GB |
| bge-m3 embedder | ~1.2 GB — does **not** coexist; Ollama swaps it in/out |

```powershell
# Ollama env on Windows (System Properties → Environment Variables)
OLLAMA_FLASH_ATTENTION=1
OLLAMA_KV_CACHE_TYPE=q8_0
OLLAMA_MAX_LOADED_MODELS=1   # 8 GB has room for exactly one resident model
```

`MAX_LOADED_MODELS=1` means query embedding evicts the LLM and vice versa — a ~2s swap
per query. Acceptable; the alternative (both resident) pushes weights into system RAM
and costs more than 2s on every generated token. Ingestion is unaffected: it's
embedder-only for long stretches.

## 2b. Unified memory budget — `mac` profile (32 GB)

macOS wires roughly 21–22 GB to the GPU by default on a 32 GB machine
(`recommendedMaxWorkingSetSize`). You can raise it when you need the 30B model with
long context:

```bash
# temporary until reboot; leaves ~7 GB for macOS — safe on a dedicated work session
sudo sysctl iogpu.wired_limit_mb=26624
```

Budget at steady state (query time):

| Resident | Size |
|---|---|
| macOS + Gradio + Python | ~5–6 GB |
| LLM weights (tier below) | 9–19 GB |
| KV cache (32k ctx, q8_0, flash attn) | 1.5–3 GB |
| bge-m3 embedder (f16) | ~1.2 GB (auto-unloads via `keep_alive`) |

Ollama env (set once, non-negotiable for this hardware):

```bash
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0   # halves KV cache; quality loss negligible
export OLLAMA_MAX_LOADED_MODELS=2  # big LLM + embedder, never more
```

## 3. Model tiering

### `pc` profile (8 GB VRAM) — active now

| Tier | Model | Quant | Weights | Role |
|---|---|---|---|---|
| **Daily driver** | Qwen3-8B-Instruct | Q4_K_M | ~5 GB | Main answering, fully on-GPU with 16k context. |
| **Code tier** | Qwen2.5-Coder-7B-Instruct | Q4_K_M | ~4.7 GB | Codebase-heavy queries; swaps in place of the daily driver. |
| **Quality (patience) tier** | Qwen3-14B-Instruct | Q4_K_M | ~9 GB | Partial CPU offload (~60% of layers on GPU) → roughly 8–12 tok/s. Optional, for hard questions where you'll wait. |
| **Utility** | Qwen3-4B-Instruct | Q4_K_M | ~2.5 GB | Ingest-time tagging, query rewriting. |
| **Embeddings** | bge-m3 | f16 | ~1.2 GB | Same on both profiles — **never re-embed when migrating**. |

The embedding model being identical across profiles is the one hard compatibility rule:
the LanceDB index you build on the PC this month is byte-for-byte the index the Mac
uses next month.

### `mac` profile (32 GB unified) — from delivery

| Tier | Model | Quant | Weights | Role |
|---|---|---|---|---|
| **Daily driver** | Qwen3-14B-Instruct | Q4_K_M | ~9 GB | Main answering. Dense 14B leaves room for 32–64k context inside the *default* wired limit — no sysctl needed. KV @32k q8_0 ≈ 2.6 GB. |
| **Quality ceiling** | Qwen3-30B-A3B-Instruct-2507 | Q4_K_M | ~18.5 GB | MoE with ~3B active params → decodes *faster* than the dense 14B while reasoning better. Needs the wired-limit bump for >16k context. Use for proofs and hard synthesis questions. |
| **Code tier** | Qwen2.5-Coder-14B-Instruct | Q4_K_M | ~9 GB | Codebase-heavy queries; swap in place of daily driver. |
| **Utility** | Qwen3-4B-Instruct | Q4_K_M | ~2.5 GB | Query rewriting, metadata extraction at ingest, title/topic tagging. Coexists in RAM with the daily driver. |
| **Embeddings** | bge-m3 | f16 | ~1.2 GB | Ingest + query encoding. |

Skip 3B-and-under for answering (correct instinct in the brief) and skip 70B+ entirely:
Q3 70B fits but decodes at reading-pace-breaking speed on a 10-core GPU and slams the
fanless thermal envelope.

## 4. Directory architecture

On the Mac this lives at `~/Developer/paper_rag/`:

```
paper_rag/
├── pyproject.toml           # uv-managed; deps: docling, lancedb, ollama, gradio
├── config.py                # paths, model names, chunk params, tier table
├── app.py                   # Gradio UI entry point
├── ragcore/
│   ├── __init__.py
│   ├── ingest.py            # Docling parse → HybridChunker → chunk records
│   ├── embed.py             # Ollama embeddings client (batch, retry)
│   ├── store.py             # LanceDB: schema, add, hybrid search + filters
│   ├── generate.py          # prompt assembly, citations, streaming chat
│   └── jobs.py              # asyncio ingestion queue + per-doc status
├── tests/
│   └── test_smoke.py        # parse 1 PDF → index → retrieve → assert citation
└── data/
    ├── library/             # source PDFs, one subdir per course
    │   └── cs5xx-distsys/
    ├── lancedb/             # the index (deletable, rebuildable)
    └── staging/             # parsed DoclingDocuments (JSON) — parse once, re-chunk free
```

Chunk record schema (one LanceDB table, that's all):

```
vector(1024) | text | doc_id | course | topic | doc_type{textbook,paper,code}
| content_kind{prose,code,table,math} | section | page | source_path
```

`content_kind` comes free from Docling labels and is what lets a query like
"show me the actual implementation" filter to code chunks.

## 5. Data flow

```
                       INGESTION  (async, batched, thermal-aware)
┌───────────┐   ┌──────────────────┐   ┌───────────────────────┐
│ PDFs in    │──▶│ Docling parser   │──▶│ HybridChunker          │
│ data/      │   │ layout + tables  │   │ (code/table/math kept  │
│ library/   │   │ + code + formula │   │  atomic, sections      │
└───────────┘   │ enrichment [MPS] │   │  respected)            │
                └──────────────────┘   └──────────┬────────────┘
                        │ cache parsed JSON        │ chunks + metadata
                        ▼                          ▼
                ┌──────────────┐        ┌────────────────────┐
                │ data/staging/ │        │ Ollama: bge-m3      │ [Metal]
                └──────────────┘        │ embeddings (batched)│
                                        └─────────┬──────────┘
                                                  ▼
                                       ┌─────────────────────┐
                                       │ LanceDB (on-disk)    │
                                       │ ANN index + BM25 FTS │
                                       │ + metadata columns   │
                                       └─────────────────────┘

                              QUERY
┌───────┐   ┌───────────────┐   ┌─────────────────────────────┐
│ User   │──▶│ Gradio UI     │──▶│ hybrid search: vector + BM25 │
│ query  │   │ filters:      │   │ fused with RRF, pre-filtered │
└───────┘   │ course/topic/ │   │ by course/topic/doc_type     │
     ▲      │ doc_type      │   └──────────────┬──────────────┘
     │      └───────────────┘                  │ top-k chunks
     │                                         ▼
     │                            ┌─────────────────────────┐
     │                            │ prompt builder            │
     │                            │ (numbered chunks → forced │
     │                            │  [n] citation format)     │
     │                            └────────────┬─────────────┘
     │                                         ▼
     │       streamed tokens      ┌─────────────────────────┐
     └────────────────────────────│ Ollama LLM [Metal]        │
             + cited sources      │ Qwen3-14B / 30B-A3B       │
                                  └─────────────────────────┘
```

## 6. Hardware-specific engineering notes

**GPU acceleration on both profiles.** Ollama picks CUDA on the PC and Metal on the Mac
automatically (verify: GPU pegged in Task Manager / Activity Monitor during generation,
CPU quiet). Docling likewise auto-detects — use AUTO and the same line works on both
machines:

```python
from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions
accel = AcceleratorOptions(device=AcceleratorDevice.AUTO)  # CUDA on pc, MPS on mac
```

On the PC, run ingestion (Docling models + embedder) while the LLM is idle — Docling's
layout models want ~2 GB of the 8 GB for themselves.

The 16-core ANE goes deliberately unused — neither llama.cpp nor MLX targets it for
LLMs, and CoreML-converting the embedder is complexity with no measurable win here.

**Fanless thermal strategy — `mac` profile only.** The 4060 Ti has active cooling;
on the PC the rest sleep is 0 and ingestion runs flat out. On the Mac:

```python
# ponytail: fixed sleep, swap to thermal-pressure polling only if throttling is measured
for batch in batched(pending_docs, 5):          # 5 PDFs per burst
    for doc in batch:
        chunks = parse_and_chunk(doc)            # Docling, GPU-heavy
        store.add(embed(chunks))                 # GPU-heavy
        jobs.mark_done(doc)
    time.sleep(REST_SECONDS)                     # config.py: 0 on pc, 15 on mac
```

- Ingest on mains power with Low Power Mode **off** (it hard-caps sustained clocks).
- Never run ingestion and the 30B model simultaneously — parse queue pauses when a
  chat generation is in flight (single asyncio lock, `jobs.py`).
- Parsing is the slow step, not embedding. Cache every parsed `DoclingDocument` to
  `data/staging/` so re-chunking or re-embedding never re-parses.

**Storage.** Everything user-visible under `~/Developer/paper_rag/data/`. Ollama keeps
GGUFs in `~/.ollama` — leave it (set `OLLAMA_MODELS` only if you want them co-located).
A term's corpus (~50 textbooks + papers) ≈ 2–4 GB of index; wiping between terms is
`rm -rf data/lancedb data/staging`.

## 7. Implementation roadmap

**Phase 0 — Bench (half a day, on the PC).**
`uv init`, install Ollama for Windows, pull the `pc`-profile models, run a 2k-token
prompt on each tier and record tokens/s + VRAM from Task Manager. *Gate: Qwen3-8B ≥
~35 tok/s decode fully on-GPU; if VRAM spills into "Shared GPU memory", the model or
context is too big — shrink before building anything.*

> **Phase 0 results (2026-07-05, pc profile — GATE PASS):** 16k ctx, q8_0 KV, flash attn.
> qwen3:8b — 2,329 tok/s prefill, 38.8 tok/s decode, 6.3 GB, 100% GPU.
> qwen3:4b — 4,642 tok/s prefill, 77.9 tok/s decode, 4.0 GB, 100% GPU.
> qwen2.5-coder:7b — 53.9 tok/s decode, 5.0 GB, 100% GPU.
> bge-m3 — 1024-dim embeddings verified via /api/embed.
> qwen3:14b not pulled yet (slow connection) — pull overnight; nothing in Phase 1 needs it.

**Phase 0b — Mac migration (half a day, next month).**
Copy repo + `data/` to the Mac, install Ollama, pull the `mac`-tier models, flip
`PROFILE = "mac"` in config.py, re-run the Phase-0 bench (*gate: 14B ≥ ~20 tok/s on
Metal*) and the Phase-2 eval. The index carries over untouched — same bge-m3 vectors.

**Phase 1 — Ingestion CLI (1–2 days).**
`ingest.py` + `store.py` + `embed.py`. Docling with code+formula enrichment, HybridChunker,
metadata from directory layout (`library/<course>/`), thermal batch loop, staging cache.
CLI: `python -m ragcore.ingest data/library/cs5xx-distsys`.
*Gate: `test_smoke.py` — one real textbook chapter in, chunks in LanceDB with correct
`content_kind` labels, a code listing survives as one atomic chunk.*

**Phase 2 — Retrieval (1 day).**
Hybrid search (vector + BM25, RRF fusion) with metadata pre-filters in `store.py`.
Build a 20-question eval file (`tests/eval_questions.jsonl`) from your actual courses.
*Gate: expected source doc appears in top-5 for ≥16/20 questions. No reranker yet —
measure first.*

> **Phase 2 results (2026-07-05 — GATE PASS 20/20):** 6-paper corpus (attention, bert,
> resnet, gan, adam, word2vec), 283 chunks. Expected doc was top-1 on all 20 questions.
> Native lance FTS (tantivy has no Windows wheels). Parse-time note: math-dense papers
> are slow on CPU torch (adam.pdf: 42 min, formula enrichment) — install CUDA torch
> before bulk textbook ingestion.

**Phase 3 — Generation (1 day).**
`generate.py`: numbered-chunk prompt, forced `[n]` citations, streaming via Ollama chat
API, tier selection by flag. Terminal chat loop before any UI.
*Gate: answers cite real chunk numbers; refuses cleanly when retrieval returns nothing
relevant.*

> **Phase 3 results (2026-07-05 — GATE PASS):** citations map to real chunk numbers
> (on-corpus test cited correctly, off-corpus "capital of France" refused with exact
> NOT IN LIBRARY marker). qwen3 runs with think=False for snappy RAG turns. CLI:
> `python -m ragcore.generate [--tier ...] [--q "..."]`. Windows console needed
> utf-8 reconfigure (Greek letters in papers).

**Phase 4 — UI (1–2 days).**
`app.py`: Gradio Blocks, dark theme. Tab 1: chat with streaming, source panel,
course/topic/doc_type filter dropdowns, model-tier selector. Tab 2: library — drop PDFs,
per-document async status (queued → parsing → embedding → indexed / failed) from `jobs.py`.
*Gate: drop 3 PDFs mid-chat; statuses update live; chat stays responsive.*

> **Phase 4 results (2026-07-05 — gate mostly verified):** Gradio pinned to 5.x
> (`gradio>=5,<6` — 6.19 broke ChatInterface kwargs and chat rendering was unverifiable).
> Async queue verified: new PDF went queued→parsing→embedding→indexed(23 chunks) via
> jobs.py worker thread; gpu_lock serializes embedding vs generation (8 GB card).
> Backend chat path verified in-process (streamed, cited), then confirmed live in
> browser by user: streamed, correctly cited answer with sources footer. GATE PASS.

**Phase P — Paper-study + presentation prep (2026-07-06 — GATE PASS).**
Built for the NYU "Efficient AI and Hardware Accelerators" in-class paper presentation:
- **Whole-paper mode**: Document dropdown + checkbox in Chat puts an entire paper
  (chunks in `seq` order) into context — summarization/walkthrough questions that
  top-k retrieval can't answer. `FULLDOC_NUM_CTX` is profile-tiered (pc 16384 —
  measured 100% GPU; 20480+ spills to CPU on 8 GB; mac 32768).
- **Present tab**: pick paper + talk length → Marp markdown deck (`data/exports/
  <paper>/deck.md`) with slide arc, grounded speaker notes with page refs, discussion
  questions, anticipated Q&A. Figures/tables extracted as PNGs via one-time re-parse
  with `generate_picture_images` (10 images from the attention paper). A deterministic
  figure-appendix in `save_deck` guarantees images land in the deck (the 8B ignores
  embed instructions).
- Gates in `tests/test_present.py`; retrieval eval regression stayed 20/20 after the
  `seq` schema migration.
- **2026-07-07:** deck generator rewritten around the professor's graded format:
  mandatory arc (Introduction → Background → Methodology → Evaluation → Your Thoughts
  & Discussion), rubric wired into the prompt (Clarity 3 / Structure & Flow 2 / Depth
  of Analysis 3 / Discussion 2), Methodology weighted heaviest (~40% target), HANDOFF
  markers for the 3-presenter split, talk lengths = course formats (15 min algorithm /
  25 min architecture). Gate extended: ≥4 of 5 section names required — passes 5/5.
- **CUDA torch installed (2026-07-07):** cu130 index in pyproject (cu128 lacks cp313
  win wheels; driver 591.86 handles CUDA 13). `[tool.uv] environments` restricted to
  win32 — phantom mac/py3.14 resolution splits otherwise break the lock; the Mac
  migration replaces this block. Measured: gru-eval.pdf parse 156s CPU → **25s GPU**.

**Phase 5 — Quality, only as measured need (open-ended).**
In priority order, each gated on the Phase-2 eval actually improving:
1. Utility-model query rewriting (multi-query on the 4B tier).
2. Cross-encoder reranker (bge-reranker-v2-m3) — the one place sentence-transformers/MPS
   enters the project.
3. MLX backend experiment (`mlx-lm` server) — M5 neural accelerators make long-context
   prefill markedly faster; keep Ollama as the fallback.
4. Parent-chunk expansion (return the full section around a hit) if answers feel clipped.

Skipped deliberately: multi-user auth, Docker, GraphRAG, agentic retrieval loops,
observability stacks — single user, single machine; add nothing until the eval says so.
