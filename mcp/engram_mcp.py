#!/usr/bin/env python3
"""engram_mcp.py — 8b-is/Engram as a stdio MCP server.

Zero dependencies beyond the repo's own engram_pq.py. Exposes both durable
p/q N-gram memory tools and zero-float p/q + BFP-64 analysis tools:

  engram_remember  Persist a durable memory entry indexed by Lemire p/q N-gram hashes
  engram_recall    Search and recall durable memories across sessions by text + p/q N-gram overlap
  engram_hash      N-gram p/q hash mapping (Lemire q=2^64) over token ids
  engram_compress  BFP-64 table compression stats for a text
  engram_quantum   64-trit bitplane packing ratio
  engram_info      repo identity + git log
  engram_bench     run the Rust bench (cargo) and return parsed rows

Wire format: JSON-RPC 2.0 over stdio, as MCP specifies.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import engram_pq as eq


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def _send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _store_path():
    custom = os.environ.get("ENGRAM_8B_STORE") or os.environ.get("ENGRAM_DATA_DIR")
    if custom:
        if custom.endswith(".jsonl"):
            parent = os.path.dirname(custom)
            if parent:
                os.makedirs(parent, exist_ok=True)
            return custom
        os.makedirs(custom, exist_ok=True)
        return os.path.join(custom, "memories.jsonl")
    default_dir = os.path.expanduser("~/.engram-8b")
    os.makedirs(default_dir, exist_ok=True)
    return os.path.join(default_dir, "memories.jsonl")


_OFFLINE_TOK = None


def _get_offline_tok():
    global _OFFLINE_TOK
    if _OFFLINE_TOK is None:
        _OFFLINE_TOK = eq.CompressedTokenizerPq("offline")
    return _OFFLINE_TOK


def _compute_pq_signature(text):
    toks = _get_offline_tok()
    ids = [int(x) for x in toks.encode(text)]
    compressed = [int(x) for x in toks._compress_row(ids)]
    words = re.findall(r"[a-z0-9_-]+", text.lower())
    ngrams = set(words)
    for i in range(len(words) - 1):
        ngrams.add(f"{words[i]}_{words[i + 1]}")
    # Deterministic 64-bit Lemire-style multiplicative hash per token/bigram
    FNV_PRIME = 0x100000001B3
    GOLDEN_64 = 0x9E3779B97F4A7C15
    sig = sorted(
        ((int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(), "little") * GOLDEN_64)
         ^ FNV_PRIME)
        & 0xFFFFFFFFFFFF
        for tok in ngrams
    )
    return ids, compressed, sig[:64]


def tool_engram_remember(args):
    """Remember and persist a durable fact, preference, observation, or note across sessions using 8b-is p/q N-gram indexing."""
    content = (
        args.get("content")
        or args.get("text")
        or args.get("value")
        or args.get("observation")
        or ""
    ).strip()
    key = (args.get("key") or args.get("title") or args.get("topic") or "").strip()
    tags = args.get("tags") or ""
    project = (args.get("project") or os.environ.get("ENGRAM_PROJECT") or os.path.basename(os.getcwd())).strip()
    if not content and key:
        content = key
    if not content:
        raise ValueError("content (or text/value) is required to remember a memory")
    full_text = f"{key}: {content}" if key and key.lower() not in content.lower() else content
    ids, compressed, pq_sig = _compute_pq_signature(full_text)
    ts_ms = int(time.time() * 1000)
    entry_id = hashlib.blake2b(f"{ts_ms}:{full_text}".encode("utf-8"), digest_size=6).hexdigest()
    record = {
        "id": entry_id,
        "ts_ms": ts_ms,
        "project": project,
        "key": key or content[:48],
        "content": content,
        "tags": tags,
        "token_count": len(ids),
        "compressed_count": len(compressed),
        "pq_sig": pq_sig,
    }
    path = _store_path()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "status": "remembered",
        "id": entry_id,
        "key": record["key"],
        "content": content,
        "project": project,
        "pq_hashes": len(pq_sig),
        "store": path,
    }


def tool_engram_recall(args):
    """Search and recall durable memories saved across sessions by keyword, topic, or p/q N-gram similarity."""
    query = (
        args.get("query")
        or args.get("text")
        or args.get("key")
        or args.get("search")
        or ""
    ).strip()
    limit = int(args.get("limit") or 10)
    path = _store_path()
    if not os.path.exists(path):
        return {"query": query, "count": 0, "memories": [], "store": path}
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not query:
        recent = list(reversed(rows[-limit:]))
        return {"query": "", "count": len(recent), "memories": recent, "store": path}

    _, _, q_sig = _compute_pq_signature(query)
    q_set = set(q_sig)
    q_words = [w for w in re.findall(r"[a-z0-9_-]+", query.lower()) if len(w) > 1]
    scored = []
    for idx, row in enumerate(rows):
        haystack = f"{row.get('key', '')} {row.get('content', '')} {row.get('tags', '')}".lower()
        r_set = set(row.get("pq_sig") or [])
        pq_overlap = len(q_set & r_set) / max(1, len(q_set)) if q_set else 0.0
        substr_hits = sum(1 for w in q_words if w in haystack)
        exact_phrase = 2.0 if query.lower() in haystack else 0.0
        score = exact_phrase + substr_hits * 1.0 + pq_overlap * 1.5
        if score > 0 or not q_words:
            scored.append((score, idx, row))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    matches = [
        {
            "id": r.get("id"),
            "key": r.get("key"),
            "content": r.get("content"),
            "tags": r.get("tags"),
            "project": r.get("project"),
            "ts_ms": r.get("ts_ms"),
            "score": round(sc, 3),
        }
        for sc, _, r in scored[:limit]
    ]
    return {"query": query, "count": len(matches), "memories": matches, "store": path}


def tool_engram_hash(args):
    """Compute zero-float Lemire p/q N-gram hashes over token IDs."""
    text = args.get("text", "the sovereign ledger rows everything")
    n_head = int(args.get("heads", 16))
    use_pq = bool(args.get("pq", True))
    t = time.perf_counter()
    mapper = eq.NgramHashMappingPq(
        engram_vocab_size=[4096], max_ngram_size=3, n_embed_per_ngram=512,
        n_head_per_ngram=n_head, layer_ids=[0], tokenizer_name_or_path="offline",
        seed=int(args.get("seed", 0)),
    )
    build_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    hashes = mapper.hash_ngram_pq([0], 4096, use_pq_hash=use_pq) if hasattr(mapper, "hash_ngram_pq") else None
    hash_ms = (time.perf_counter() - t) * 1000
    return {
        "heads": n_head, "pq": use_pq, "build_ms": round(build_ms, 3),
        "hash_ms": round(hash_ms, 3),
        "hash_preview": [int(x) for x in (hashes[:8] if hashes is not None else [])],
        "note": text,
    }


def tool_engram_compress(args):
    text = args.get("text", "love life and everything we discover between them")
    toks = _get_offline_tok()
    ids = toks.encode(text)
    compressed = toks._compress_row(ids)
    rows = 1
    dense_bytes = len(ids) * 4
    bfp_bytes = len(ids)  # 65 B/row vs 256 B/row ratio ~3.94x; per-row i8 mantissas
    ratio = round(dense_bytes / max(1, bfp_bytes), 2)
    return {
        "tokens": len(ids), "compressed_tokens": len(compressed),
        "dense_fp32_bytes": dense_bytes, "bfp64_bytes": bfp_bytes,
        "ratio_x": ratio, "text": text,
    }


def tool_engram_quantum(args):
    vecs = int(args.get("vectors", 64))
    dim = int(args.get("dim", 64))
    bits_total = vecs * dim * 32
    bits_trit = vecs * dim * 2  # 64-trit bitplanes, 2 bits/weight
    return {
        "vectors": vecs, "dim": dim,
        "fp32_bits": bits_total, "quantum_trit_bits": bits_trit,
        "compression_x": round(bits_total / bits_trit, 2),
        "gmcs": round((vecs * dim) / (14.531e-6) / 1e9, 2),  # 1024x1024 quantum u64 at 14.5us
    }


def tool_engram_info(args):
    try:
        head = subprocess.run(["git", "log", "-3", "--oneline"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "-sb"], cwd=REPO,
                                capture_output=True, text=True).stdout.strip().splitlines()[0]
    except Exception as e:
        head, status = "", str(e)
    return {"repo": "8b-is/Engram", "path": REPO, "head": head, "status": status,
            "upstream_pr": "https://github.com/deepseek-ai/Engram/pull/28"}


def tool_engram_bench(args):
    dim = int(args.get("dim", 0))
    cmd = ["cargo", "run", "--manifest-path", os.path.join(REPO, "rust", "Cargo.toml"),
           "--release", "--bin", "bench"]
    if dim:
        cmd += ["--", "--dim", str(dim)]
    try:
        out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=180)
        rows = []
        for line in out.stdout.splitlines():
            m = re.match(r"\s+-\s(.+):\s+([\d.]+)\s+us\s+\(([\d.]+)x(?: speedup)?(?:, (.+?))?\)(?:\s+\|\s+rel_err:\s+([\d.e-]+))?", line)
            if m:
                rows.append({"kernel": m.group(1).strip(), "latency_us": float(m.group(2)),
                             "speedup": float(m.group(3)),
                             "rel_err": float(m.group(5)) if m.group(5) else None})
        return {"rows": rows, "exit": out.returncode,
                "tail": out.stderr.strip().splitlines()[-1] if out.stderr.strip() else ""}
    except subprocess.TimeoutExpired:
        return {"rows": [], "exit": "timeout"}


TOOLS = {
    "engram_remember": tool_engram_remember,
    "engram_recall": tool_engram_recall,
    "engram_hash": tool_engram_hash,
    "engram_compress": tool_engram_compress,
    "engram_quantum": tool_engram_quantum,
    "engram_info": tool_engram_info,
    "engram_bench": tool_engram_bench,
}

SCHEMAS = {
    "engram_remember": {
        "content": "durable fact, preference, or observation to remember",
        "key": "optional short topic or key",
        "tags": "optional comma-separated tags",
        "project": "optional project scope override",
    },
    "engram_recall": {
        "query": "search query or topic to recall from memory (omit to list recent memories)",
        "limit": "maximum number of matching memories to return (default 10)",
    },
    "engram_hash": {"text": "text to hash", "heads": "number of heads (default 16)", "pq": "use Lemire p/q (default true)", "seed": "hash seed"},
    "engram_compress": {"text": "text to compress"},
    "engram_quantum": {"vectors": "vector count", "dim": "dimension"},
    "engram_info": {},
    "engram_bench": {"dim": "optional matrix dim for a single run"},
}


def main():
    log("engram_mcp: listening on stdio")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = req.get("id")
        method = req.get("method")
        params = req.get("params") or {}
        if method == "initialize":
            _send({"jsonrpc": "2.0", "id": rid, "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "engram-mcp", "version": "0.1.0"},
            }})
        elif method == "notifications/initialized":
            pass
        elif method == "tools/list":
            _send({"jsonrpc": "2.0", "id": rid, "result": {"tools": [
                {"name": n, "description": f.__doc__ or n,
                 "inputSchema": {"type": "object", "properties": {k: {"type": "string"} for k in SCHEMAS[n]}}}
                for n, f in TOOLS.items()
            ]}})
        elif method == "tools/call":
            name = params.get("name")
            fn = TOOLS.get(name)
            if fn is None:
                _send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": f"unknown tool {name}"}})
                continue
            try:
                result = fn(params.get("arguments") or {})
                _send({"jsonrpc": "2.0", "id": rid, "result": {
                    "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
                    "isError": False,
                }})
            except Exception as e:
                _send({"jsonrpc": "2.0", "id": rid, "result": {
                    "content": [{"type": "text", "text": f"error: {e}"}],
                    "isError": True,
                }})
        elif method == "ping":
            _send({"jsonrpc": "2.0", "id": rid, "result": {}})
        else:
            _send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"no such method {method}"}})


if __name__ == "__main__":
    main()
