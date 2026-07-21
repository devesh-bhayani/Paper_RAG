# PAPER_RAG

100% offline RAG assistant for CS papers and textbooks. Docling for layout-aware PDF
parsing, LanceDB for hybrid (vector + BM25) retrieval, Ollama for local embeddings and
generation, Gradio for the UI. No LangChain/LlamaIndex — plain Python throughout.

Two features beyond plain Q&A:
- **Whole-paper chat mode** — put an entire paper in context for summarization/walkthrough questions that top-k retrieval handles poorly.
- **Present tab** — generate a graded presentation deck (Marp markdown) for a paper, with extracted figures, grounded speaker notes, and discussion Q&A.

See [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) for the full architecture, hardware notes, and phase-by-phase build log.

## Prerequisites

- **Python 3.13+** and [uv](https://docs.astral.sh/uv/)
- **[Ollama](https://ollama.com)** running locally
- Currently tuned for **Windows + NVIDIA GPU** (`pc` profile in `config.py`). CUDA torch
  wheels are pinned in `pyproject.toml`; on other platforms, edit or drop that
  `[tool.uv]` / `[[tool.uv.index]]` block and let torch resolve normally.

## Setup

```bash
# 1. install dependencies (pulls CUDA torch on Windows — a few GB, one time)
uv sync

# 2. pull the models this profile uses (see config.py TIERS for the exact names)
ollama pull qwen3:8b
ollama pull qwen2.5-coder:7b
ollama pull qwen3:4b
ollama pull bge-m3
# optional, larger/slower "quality" tier — worth it for Methodology-heavy decks
ollama pull qwen3:14b

# 3. recommended Ollama env vars — macOS/Linux (persist in your shell profile)
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_MAX_LOADED_MODELS=1   # raise if you have >8 GB VRAM

# 4. add your PDFs, one folder per course (or drag them into the Library tab)
mkdir -p data/library/my-course
cp ~/Downloads/some-paper.pdf data/library/my-course/

# 5. ingest (parses, chunks, embeds, indexes — cached, safe to re-run)
uv run python -m ragcore.ingest

# 6. launch the UI
uv run python app.py
```

On **Windows** (the primary profile), set the step-3 vars once with `setx`, then
restart the Ollama tray app — it only reads them at startup:

```powershell
setx OLLAMA_FLASH_ATTENTION 1
setx OLLAMA_KV_CACHE_TYPE q8_0
setx OLLAMA_MAX_LOADED_MODELS 1
```

Open **http://127.0.0.1:7860**. Three tabs: **Chat** (ask questions, optionally scoped
to one course/document, or "whole paper" mode), **Present** (pick a paper, extract
figures, generate a slide deck), **Library** (drag-and-drop new PDFs, watch ingestion
status live, remove a document to re-index a corrected PDF).

Good to know:

- **Chat turns are independent** — each question is answered from a fresh retrieval,
  with no memory of the previous answer. Follow-ups like *"explain that more simply"*
  won't work; re-ask the full question instead.
- **First run needs network once** (Docling models + the bge-m3 tokenizer from
  HuggingFace); everything after that is offline. The unauthenticated-rate-limit
  warning HuggingFace prints is harmless — set `HF_TOKEN` to silence it.
- **Model tier** in the Chat/Present dropdowns picks the answering model: `daily` is
  the balanced default, `utility` is the fast small one, `quality` is slow but best for
  Methodology-heavy decks (it's the one worth waiting for before a real presentation).

## Verifying it works

```bash
uv run python bench.py                  # tokens/sec per model tier, checks GPU residency
uv run python tests/test_smoke.py       # ingest -> chunks land with correct structure
uv run python tests/eval_retrieval.py   # 20-question retrieval accuracy gate
uv run python tests/test_generate.py    # citations are real, off-corpus questions refuse
uv run python tests/test_jobs.py        # async queue + delete/re-index cycle
uv run python tests/test_present.py     # whole-paper mode + deck generation gate
```

Gates need Ollama up and the models pulled; they fetch and ingest their own fixtures on
first run (network once). They assert against the real `data/` store, not a sandbox.
There is no CI — a GitHub Action can't run them (no GPU, no Ollama), so **run
`eval_retrieval.py` by hand before pushing any retrieval or schema change**.

## Repo layout

```
config.py          # profiles, model tiers, paths — the one place settings live
ragcore/
  ingest.py         # Docling parse (cached) -> chunk -> embed -> LanceDB
  store.py          # LanceDB schema + hybrid search + whole-doc accessors
  embed.py          # batched Ollama embeddings client
  generate.py       # cited, streaming Q&A (retrieval-scoped or whole-paper)
  present.py        # figure extraction + Marp deck generation
  jobs.py           # async ingestion queue backing the Library tab
app.py              # Gradio UI (Chat / Present / Library)
tests/              # gate scripts — run directly with uv run python
data/               # library (your PDFs), staging (parse cache), lancedb (index),
                    # exports (generated decks/figures) — all gitignored
```

`data/library` and the LanceDB index are gitignored (source PDFs may be copyrighted,
and the index/cache are fully rebuilt by re-running ingestion).
