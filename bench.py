"""Phase 0 bench: tokens/s per model tier on the local Ollama server.

Stdlib only — run with `python bench.py [model ...]`.
Gate (pc profile): qwen3:8b decode >= 35 tok/s fully on-GPU.
"""

import json
import subprocess
import sys
import urllib.request

OLLAMA = "http://localhost:11434"
PC_TIERS = ["qwen3:8b", "qwen2.5-coder:7b", "qwen3:4b", "qwen3:14b"]
GATE_MODEL, GATE_TOKS = "qwen3:8b", 35.0

# ~2k tokens of realistic prose so prefill numbers mean something
PARAGRAPH = (
    "In a distributed key-value store, the replication protocol must balance "
    "consistency guarantees against tail latency. Quorum-based approaches such as "
    "those derived from Dynamo trade linearizability for availability during "
    "partitions, while consensus protocols like Raft serialize writes through an "
    "elected leader, simplifying reasoning at the cost of throughput ceilings. "
)
PROMPT = PARAGRAPH * 55 + "\n\nSummarize the trade-offs above in three sentences."


def bench(model: str) -> dict | None:
    body = json.dumps({
        "model": model,
        "prompt": PROMPT,
        "stream": False,
        "options": {"num_ctx": 16384, "num_predict": 256},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            r = json.load(resp)
    except Exception as e:
        print(f"  {model}: SKIP ({e})")
        return None
    return {
        "prefill_toks": r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9),
        "decode_toks": r["eval_count"] / (r["eval_duration"] / 1e9),
        "total_s": r["total_duration"] / 1e9,
    }


def gpu_residency() -> str:
    # ollama ps shows "100% GPU" when nothing spilled to CPU
    out = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
    lines = out.strip().splitlines()
    return lines[1] if len(lines) > 1 else "(nothing loaded)"


def main() -> None:
    models = sys.argv[1:] or PC_TIERS
    results = {}
    for m in models:
        print(f"benching {m} (2k-token prompt, 16k ctx)...")
        r = bench(m)
        if r is None:
            continue
        results[m] = r
        print(f"  prefill {r['prefill_toks']:7.1f} tok/s | "
              f"decode {r['decode_toks']:6.1f} tok/s | "
              f"total {r['total_s']:5.1f}s")
        print(f"  residency: {gpu_residency()}")

    if GATE_MODEL in results:
        ok = results[GATE_MODEL]["decode_toks"] >= GATE_TOKS
        print(f"\nGATE {GATE_MODEL} >= {GATE_TOKS} tok/s decode: "
              f"{'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
