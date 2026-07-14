"""PAPER RAG — Gradio UI. Run: uv run python app.py  ->  http://127.0.0.1:7860"""

from pathlib import Path

import gradio as gr

import config
from ragcore import generate, jobs, present, store

FORCE_DARK = "() => { document.documentElement.classList.add('dark'); }"


def list_courses() -> list[str]:
    if not config.LIBRARY_DIR.exists():
        return []
    return sorted(d.name for d in config.LIBRARY_DIR.iterdir() if d.is_dir())


def chat_fn(message, history, tier, course, kind, document, whole_paper):
    filters = {}
    if course and course != "all courses":
        filters["course"] = course
    if kind and kind != "any kind":
        filters["content_kind"] = kind
    doc_selected = document and document != "all documents"

    with jobs.gpu_lock:  # don't fight the ingestion worker for the GPU
        if doc_selected and whole_paper:
            tokens, chunks = generate.answer_full_doc(message, document, tier=tier)
        else:
            if doc_selected:
                filters["doc_id"] = document
            tokens, chunks = generate.answer(message, tier=tier, **filters)
        acc = ""
        for t in tokens:
            acc += t
            yield acc
    if chunks:
        acc += "\n\n---\n**Sources**\n" + "\n".join(
            f"- [{i}] `{c['doc_id']}` p.{c['page']} *{c['content_kind']}*"
            for i, c in enumerate(chunks, 1))
        yield acc


def upload_fn(files, course):
    for f in files or []:
        jobs.submit(Path(f), course)
    return jobs.rows(), gr.Dropdown(choices=["all courses"] + list_courses())


with gr.Blocks(title="PAPER RAG", theme=gr.themes.Soft(), js=FORCE_DARK) as app:
    gr.Markdown("# PAPER RAG — offline study assistant")

    with gr.Tab("Chat") as chat_tab:
        with gr.Row():
            tier = gr.Dropdown(list(config.TIERS), value="daily", label="Model tier")
            course = gr.Dropdown(["all courses"] + list_courses(),
                                 value="all courses", label="Course")
            kind = gr.Dropdown(["any kind", "prose", "code", "table", "math"],
                               value="any kind", label="Content kind")
            document = gr.Dropdown(["all documents"] + store.list_docs(),
                                   value="all documents", label="Document")
            whole_paper = gr.Checkbox(label="Whole paper in context",
                                      info="needs a document selected")
        gr.ChatInterface(fn=chat_fn,
                         additional_inputs=[tier, course, kind, document, whole_paper],
                         type="messages")
        chat_tab.select(
            lambda: gr.Dropdown(choices=["all documents"] + store.list_docs()),
            outputs=document)

    with gr.Tab("Present") as present_tab:
        with gr.Row():
            paper = gr.Dropdown(store.list_docs(), label="Paper")
            talk_len = gr.Dropdown(list(config.TALK_LENGTHS),
                                   value="15 min (algorithm paper)",
                                   label="Talk length")
            p_tier = gr.Dropdown(list(config.TIERS), value="daily", label="Model tier")
        with gr.Row():
            fig_btn = gr.Button("1. Extract figures")
            deck_btn = gr.Button("2. Generate deck", variant="primary")
        gallery = gr.Gallery(label="Extracted figures (drag into your slides)",
                             columns=4, height=280)
        deck_file = gr.File(label="Download deck.md (Marp markdown)")
        deck_md = gr.Markdown()

        def figs_fn(doc_id):
            return [str(p) for p in present.extract_figures(doc_id)] if doc_id else []

        def deck_fn(doc_id, talk, tier_name):
            if not doc_id:
                yield "Pick a paper first.", None
                return
            with jobs.gpu_lock:
                acc = ""
                for t in present.deck_stream(doc_id, talk, tier_name):
                    acc += t
                    yield acc, None
            yield acc, str(present.save_deck(doc_id, acc))

        fig_btn.click(figs_fn, inputs=paper, outputs=gallery)
        deck_btn.click(deck_fn, inputs=[paper, talk_len, p_tier],
                       outputs=[deck_md, deck_file])
        present_tab.select(lambda: gr.Dropdown(choices=store.list_docs()),
                           outputs=paper)

    with gr.Tab("Library") as library_tab:
        with gr.Row():
            uploads = gr.File(label="Drop PDFs here", file_count="multiple",
                              file_types=[".pdf"])
            up_course = gr.Textbox(label="Course", placeholder="e.g. cs5xx-distsys")
        status_df = gr.Dataframe(headers=["document", "status"], value=jobs.rows,
                                 interactive=False, label="Ingestion status")
        with gr.Row():
            del_doc = gr.Dropdown(store.list_docs(), label="Remove from index",
                                  info="the PDF itself stays in data/library/")
            del_btn = gr.Button("Remove", variant="stop")
        del_msg = gr.Markdown()

        def remove_fn(doc_id):
            if not doc_id:
                return "Pick a document first.", gr.Dropdown(), jobs.rows()
            n = store.delete_doc(doc_id)
            jobs.status.pop(doc_id, None)
            return (f"Removed **{doc_id}** ({n} chunks). The PDF is still in "
                    f"`data/library/` — re-upload or re-ingest to index it again.",
                    gr.Dropdown(choices=store.list_docs(), value=None), jobs.rows())

        uploads.upload(upload_fn, inputs=[uploads, up_course],
                       outputs=[status_df, course])
        del_btn.click(remove_fn, inputs=del_doc, outputs=[del_msg, del_doc, status_df])
        library_tab.select(lambda: gr.Dropdown(choices=store.list_docs()),
                           outputs=del_doc)
        gr.Timer(2).tick(lambda: jobs.rows(), outputs=status_df)

if __name__ == "__main__":
    app.queue().launch(server_name="127.0.0.1", server_port=7860, show_api=False)
