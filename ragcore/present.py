"""Presentation kit for one paper: extracted figures + a grounded Marp slide deck.

Outputs land in data/exports/<paper-stem>/ :
    figures/fig-NN.png, table-NN.png, manifest.tsv
    deck.md   (Marp markdown — renders to slides, reads as an outline)
"""

from collections.abc import Iterator
from pathlib import Path

import ollama

import config
from ragcore import store
from ragcore.generate import build_prompt, fit_chunks

DECK_SYSTEM = """You are preparing a GRADED in-class presentation of an academic paper \
for the NYU graduate course "Efficient AI and Hardware Accelerators". Three students \
present together. Write a complete Marp slide deck in markdown, using ONLY the \
provided source excerpts.

The deck MUST follow the professor's required section arc, in this order, each section \
opening with an '## <Section>' slide:
1. Introduction — what the paper is and its main goals, articulated with total \
clarity in the first two slides (graded: Clarity, 3 pts).
2. Background — the prior approaches and concepts needed to follow the paper.
3. Methodology — THE MOST IMPORTANT SECTION; dedicate AT LEAST {meth_slides} \
consecutive slides to it (count them). Go deep: the key idea, how the method works \
step by step, the design choices and WHY they were made, not just what they are \
(graded: Depth of Analysis, 3 pts).
4. Evaluation — experimental setup and results; name the specific tables/figures and \
their page numbers.
5. Your Thoughts & Discussion — the presenters' OWN critical analysis (strengths, \
limitations, implications for efficient AI/hardware), then 2-3 discussion questions \
to pose to the class, and a final anticipated-Q&A slide with brief grounded answers \
(graded: Discussion with audience, 2 pts).

Output rules:
- Begin with exactly this front-matter block:
---
marp: true
paginate: true
---
- Separate slides with a line containing only --- (never two in a row)
- Produce about {n_slides} slides total.
- Each slide: a short '## title' and at most 5 tight bullets.
- After the bullets on every slide, add speaker notes as an HTML comment \
(<!-- notes: ... -->), 2-4 sentences, grounded in the excerpts with page references \
like (p.4). Make section transitions explicit in the notes — the talk is also graded \
on Structure and Flow (2 pts). At each of the two natural section boundaries for \
splitting a 3-person talk, add HANDOFF to the notes.
- {fig_note}
- Output only the deck — no preamble or commentary."""


def export_dir(doc_id: str) -> Path:
    return config.EXPORT_DIR / Path(doc_id).stem


def extract_figures(doc_id: str) -> list[Path]:
    """One-time re-parse of a single PDF with image generation on; cached on disk."""
    fig_dir = export_dir(doc_id) / "figures"
    if fig_dir.exists() and any(fig_dir.glob("*.png")):
        return sorted(fig_dir.glob("*.png"))

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorDevice,
        AcceleratorOptions,
        PdfPipelineOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.generate_picture_images = True
    opts.generate_page_images = True  # table crops come from page images
    opts.images_scale = 2.0
    opts.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.AUTO)
    conv = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    doc = conv.convert(config.LIBRARY_DIR / doc_id).document

    fig_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[str] = []
    for i, pic in enumerate(doc.pictures, 1):
        img = pic.get_image(doc)
        if img is None or img.width < 64 or img.height < 64:  # logos, glyphs, noise
            continue
        name = f"fig-{i:02d}.png"
        img.save(fig_dir / name)
        manifest.append(f"{name}\t{pic.caption_text(doc) or 'no caption'}")
    for i, tbl in enumerate(doc.tables, 1):
        img = tbl.get_image(doc)
        if img is None:
            continue
        name = f"table-{i:02d}.png"
        img.save(fig_dir / name)
        manifest.append(f"{name}\t{tbl.caption_text(doc) or 'no caption'}")
    (fig_dir / "manifest.tsv").write_text("\n".join(manifest), encoding="utf-8")
    return sorted(fig_dir.glob("*.png"))


def deck_stream(doc_id: str, talk_length: str = "15 min",
                tier: str = "daily") -> Iterator[str]:
    all_chunks = store.doc_chunks(doc_id)
    if not all_chunks:
        yield f"Document not found in index: {doc_id}"
        return
    # deck reserves num_predict=4096 for output; fit_chunks' headroom covers it
    chunks, dropped = fit_chunks(all_chunks, config.FULLDOC_NUM_CTX)
    if dropped:
        yield (f"<!-- WARNING: {dropped} of {len(all_chunks)} chunks omitted (paper "
               f"exceeds context); later sections may be under-covered. -->\n")

    manifest = export_dir(doc_id) / "figures" / "manifest.tsv"
    if manifest.exists() and manifest.read_text(encoding="utf-8").strip():
        fig_note = ("These extracted image files exist next to the deck; you MUST "
                    "embed at least two of the most relevant ones on method/results "
                    "slides with markdown like ![](figures/fig-03.png). "
                    "Available files (name<TAB>caption):\n"
                    + manifest.read_text(encoding="utf-8"))
    else:
        fig_note = ("No image files are available — reference figures as placeholders "
                    "like [see Figure 2, p.4].")

    n_slides = config.TALK_LENGTHS[talk_length]
    system = DECK_SYSTEM.format(n_slides=n_slides,
                                meth_slides=max(3, round(0.3 * n_slides)),
                                fig_note=fig_note)
    parts = ollama.chat(
        model=config.TIERS[tier],
        messages=[{"role": "system", "content": system},
                  {"role": "user",
                   "content": build_prompt("Write the slide deck now.", chunks)}],
        options={"num_ctx": config.FULLDOC_NUM_CTX, "temperature": 0.4,
                 "num_predict": 4096},
        think=False,
        stream=True,
    )
    for part in parts:
        yield part["message"]["content"]


def save_deck(doc_id: str, text: str) -> Path:
    out = export_dir(doc_id)
    out.mkdir(parents=True, exist_ok=True)
    # ponytail: deterministic figure appendix — the 8B ignores embed instructions,
    # so guarantee the figures land in the deck; user deletes what they don't want
    manifest = out / "figures" / "manifest.tsv"
    if "](figures/" not in text and manifest.exists():
        slides = []
        for line in manifest.read_text(encoding="utf-8").splitlines()[:6]:
            name, _, caption = line.partition("\t")
            title = caption[:80] if caption and caption != "no caption" else name
            slides.append(f"\n---\n\n## {title}\n\n![](figures/{name})\n")
        text += "\n" + "".join(slides)
    path = out / "deck.md"
    path.write_text(text, encoding="utf-8")
    return path


def build_deck(doc_id: str, talk_length: str = "15 min", tier: str = "daily") -> Path:
    """Non-streaming convenience for tests/CLI."""
    return save_deck(doc_id, "".join(deck_stream(doc_id, talk_length, tier)))
