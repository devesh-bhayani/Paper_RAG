"""Ingestion: PDF -> Docling parse (cached) -> structure-aware chunks -> embed -> LanceDB.

CLI:  python -m ragcore.ingest [dir]     (default: data/library)
Course metadata comes from the directory layout: data/library/<course>/*.pdf
"""

import sys
import time
from itertools import batched
from pathlib import Path

import config
from ragcore import embed, store

_converter = None  # lazy: loading Docling models takes ~10s and ~2 GB


def _get_converter():
    global _converter
    if _converter is None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            AcceleratorDevice,
            AcceleratorOptions,
            PdfPipelineOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions()
        opts.do_table_structure = True
        opts.do_code_enrichment = True
        opts.do_formula_enrichment = True
        opts.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.AUTO)
        _converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
    return _converter


def cache_path(pdf_path: Path) -> Path:
    """Staging key = the full library-relative path, not just the stem: two courses can
    each hold a lecture1.pdf and they must never share a cache slot."""
    try:
        rel = pdf_path.relative_to(config.LIBRARY_DIR).with_suffix("").as_posix()
    except ValueError:  # parsed from outside the library (ad-hoc call)
        rel = pdf_path.stem
    return config.STAGING_DIR / (rel.replace("/", "__") + ".json")


def parse(pdf_path: Path):
    """Parse a PDF, caching the DoclingDocument JSON in staging — parse once, re-chunk free."""
    from docling_core.types.doc import DoclingDocument

    cache = cache_path(pdf_path)
    if cache.exists():
        return DoclingDocument.load_from_json(cache)
    doc = _get_converter().convert(pdf_path).document
    config.STAGING_DIR.mkdir(parents=True, exist_ok=True)
    doc.save_as_json(cache)
    return doc


def _content_kind(chunk) -> str:
    from docling_core.types.doc.labels import DocItemLabel

    labels = {item.label for item in chunk.meta.doc_items}
    if DocItemLabel.CODE in labels:
        return "code"
    if DocItemLabel.TABLE in labels:
        return "table"
    if DocItemLabel.FORMULA in labels:
        return "math"
    return "prose"


def chunk_doc(doc, doc_id: str, course: str, source_path: str) -> list[dict]:
    from docling.chunking import HybridChunker

    chunker = HybridChunker(tokenizer=config.EMBED_MODEL_TOKENIZER,
                            max_tokens=config.MAX_CHUNK_TOKENS)
    # ponytail: >150 pages = textbook, else paper; tag by hand if it matters
    doc_type = "textbook" if len(doc.pages) > 150 else "paper"

    records = []
    for seq, chunk in enumerate(chunker.chunk(doc)):
        pages = [p.page_no for it in chunk.meta.doc_items for p in it.prov]
        records.append({
            "seq": seq,
            "text": chunker.contextualize(chunk),
            "doc_id": doc_id,
            "course": course,
            "topic": "",  # utility-model tagging is Phase 5
            "doc_type": doc_type,
            "content_kind": _content_kind(chunk),
            "section": " > ".join(chunk.meta.headings or []),
            "page": pages[0] if pages else 0,
            "source_path": source_path,
        })
    return records


def ingest_pdf(table, pdf: Path, on_stage=None, gpu_lock=None) -> int:
    """Parse, chunk, embed and index one PDF. Returns chunk count."""
    from contextlib import nullcontext

    note = on_stage or (lambda s: None)
    # doc_id is always posix-style so the index is portable across Windows/mac
    doc_id = pdf.relative_to(config.LIBRARY_DIR).as_posix()
    note("parsing")
    records = chunk_doc(parse(pdf), doc_id, pdf.parent.name, str(pdf))
    note("embedding")
    with gpu_lock or nullcontext():  # embedding shares the GPU with chat generation
        vectors = embed.embed_texts([r["text"] for r in records])
    for r, v in zip(records, vectors):
        r["vector"] = v
    store.add_chunks(table, records)
    return len(records)


def ingest_dir(root: Path) -> None:
    table = store.open_table()
    done = store.existing_doc_ids(table)
    pdfs = [p for p in sorted(root.rglob("*.pdf"))
            if p.relative_to(config.LIBRARY_DIR).as_posix() not in done]
    if not pdfs:
        print("nothing new to ingest")
        return

    for batch in batched(pdfs, config.BATCH_DOCS):
        for pdf in batch:
            doc_id = pdf.relative_to(config.LIBRARY_DIR).as_posix()
            t0 = time.time()
            n = ingest_pdf(table, pdf,
                           on_stage=lambda s, d=doc_id: print(f"{s}  {d} ...", flush=True))
            print(f"indexed  {doc_id}  ({n} chunks, {time.time()-t0:.0f}s)")
        if config.REST_SECONDS:
            time.sleep(config.REST_SECONDS)  # fanless chassis radiates

    print("building FTS index ...")
    store.ensure_fts(table)


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else config.LIBRARY_DIR
    ingest_dir(target)
