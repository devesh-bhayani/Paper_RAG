"""Single source of config. Flip PROFILE to "mac" after migration — nothing else changes."""

from pathlib import Path

PROFILE = "pc"  # "mac" from ~2026-08

ROOT = Path(__file__).parent
LIBRARY_DIR = ROOT / "data" / "library"
STAGING_DIR = ROOT / "data" / "staging"
LANCEDB_DIR = ROOT / "data" / "lancedb"
EXPORT_DIR = ROOT / "data" / "exports"

# never change EMBED_MODEL without rebuilding the whole index
EMBED_MODEL = "bge-m3"                    # Ollama name (serving)
EMBED_MODEL_TOKENIZER = "BAAI/bge-m3"     # HF name (chunk-size counting)
EMBED_DIM = 1024

TIERS = {
    "pc": {
        "daily": "qwen3:8b",
        "code": "qwen2.5-coder:7b",
        "utility": "qwen3:4b",
        "quality": "qwen3:14b",  # partial CPU offload — patience tier
    },
    "mac": {
        "daily": "qwen3:14b",
        "code": "qwen2.5-coder:14b",
        "utility": "qwen3:4b",
        "quality": "qwen3:30b-a3b",
    },
}[PROFILE]

MAX_CHUNK_TOKENS = 512
EMBED_BATCH = 32

# whole-paper mode: an entire paper (~10-15k tokens) goes into context at once
# measured on the 8 GB card: 24576 and 20480 both spill to CPU; 16384 is 100% GPU
FULLDOC_NUM_CTX = {"pc": 16384, "mac": 32768}[PROFILE]

# talk length -> target slide count (course format: 3 presenters,
# 15 min for algorithm papers, 25 min for architecture papers)
TALK_LENGTHS = {"15 min (algorithm paper)": 12, "25 min (architecture paper)": 18}

# ingestion thermal pacing: desktop has fans, fanless Air needs rest windows
BATCH_DOCS = 5
REST_SECONDS = {"pc": 0, "mac": 15}[PROFILE]
