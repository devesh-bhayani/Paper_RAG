"""LanceDB storage: one table, one schema."""

import lancedb
import pyarrow as pa

import config

SCHEMA = pa.schema([
    pa.field("vector", pa.list_(pa.float32(), config.EMBED_DIM)),
    pa.field("seq", pa.int32()),  # chunk order within its document
    pa.field("text", pa.string()),
    pa.field("doc_id", pa.string()),
    pa.field("course", pa.string()),
    pa.field("topic", pa.string()),
    pa.field("doc_type", pa.string()),      # textbook | paper (page-count heuristic)
    pa.field("content_kind", pa.string()),  # prose | code | table | math
    pa.field("section", pa.string()),
    pa.field("page", pa.int32()),
    pa.field("source_path", pa.string()),
])


def open_table() -> lancedb.table.Table:
    db = lancedb.connect(config.LANCEDB_DIR)
    return db.create_table("chunks", schema=SCHEMA, exist_ok=True)


def existing_doc_ids(table: lancedb.table.Table) -> set[str]:
    # project one column: to_arrow() would drag 1024 floats per row through RAM first.
    # limit(0) = no limit. ponytail: still a full scan, fine below ~100k chunks.
    if table.count_rows() == 0:
        return set()
    return {r["doc_id"] for r in table.search().select(["doc_id"]).limit(0).to_list()}


def add_chunks(table: lancedb.table.Table, records: list[dict]) -> None:
    if records:
        table.add(records)


def ensure_fts(table: lancedb.table.Table) -> None:
    # native lance FTS (BM25); tantivy backend has no Windows wheels
    table.create_fts_index("text", use_tantivy=False, replace=True)


def delete_doc(doc_id: str) -> int:
    """Drop a document's chunks from the index and its staging cache, so a re-ingest
    re-parses the (possibly corrected) PDF. The source PDF is never touched.
    Returns the number of chunks removed."""
    from ragcore.ingest import cache_path  # local: ingest imports store at module level

    table = open_table()
    before = table.count_rows()
    table.delete(f"doc_id = '{_escape(doc_id)}'")
    removed = before - table.count_rows()
    if removed and table.count_rows():  # FTS index can't be built on an empty table
        ensure_fts(table)
    cache_path(config.LIBRARY_DIR / doc_id).unlink(missing_ok=True)
    return removed


def _escape(val: str) -> str:
    """SQL-style single-quote escaping for filter values (filenames may contain ')."""
    return val.replace("'", "''")


def search(query: str, k: int = 8, course: str | None = None,
           doc_type: str | None = None, content_kind: str | None = None,
           doc_id: str | None = None) -> list[dict]:
    """Hybrid retrieval: vector + BM25 fused with RRF, metadata pre-filtered."""
    from lancedb.rerankers import RRFReranker

    from ragcore import embed

    # ponytail: no ANN index — flat search is exact and instant below ~100k chunks
    table = open_table()
    qvec = embed.embed_texts([query])[0]
    q = (table.search(query_type="hybrid")
         .vector(qvec)
         .text(query)
         .rerank(RRFReranker())
         .limit(k))
    # values come from filenames/dirs the user controls — escape quotes (SQL-style)
    clauses = [f"{col} = '{_escape(val)}'" for col, val in
               (("course", course), ("doc_type", doc_type),
                ("content_kind", content_kind), ("doc_id", doc_id))
               if val]
    if clauses:
        q = q.where(" AND ".join(clauses), prefilter=True)
    return q.to_list()


_META_COLS = [f.name for f in SCHEMA if f.name != "vector"]


def list_docs() -> list[str]:
    return sorted(existing_doc_ids(open_table()))


def doc_chunks(doc_id: str) -> list[dict]:
    """All chunks of one document in reading order — whole-paper context."""
    # filter pushed into lance + metadata columns only (no vectors materialized)
    rows = (open_table().search()
            .select(_META_COLS)
            .where(f"doc_id = '{_escape(doc_id)}'")
            .limit(0)
            .to_list())
    return sorted(rows, key=lambda r: r["seq"])


if __name__ == "__main__":  # quick manual poke: python -m ragcore.store "query"
    import sys

    for r in search(sys.argv[1], k=5):
        print(f"{r['_relevance_score']:.4f}  {r['doc_id']}  p{r['page']} "
              f"[{r['content_kind']}]  {r['text'][:90].replace(chr(10), ' ')}")
