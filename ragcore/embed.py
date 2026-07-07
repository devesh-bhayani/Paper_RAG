"""Ollama embeddings client — batched, one retry."""

import time

import ollama

import config


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), config.EMBED_BATCH):
        batch = texts[i : i + config.EMBED_BATCH]
        for attempt in (1, 2):
            try:
                resp = ollama.embed(model=config.EMBED_MODEL, input=batch)
                vectors.extend(resp["embeddings"])
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)  # server busy / model swapping in
    return vectors
