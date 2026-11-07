# 8b-is / Engram — Sovereign `p/q` + `BFP-64` + `MEM|8` Benchmarking Suite

`8b-is/Engram` includes both a zero-dependency `#![no_std]`-first Rust benchmark harness (`rust/src/bin/bench.rs`) and an offline zero-dependency Python verification suite (`tests/test_engram_demo_v1.py`).

## Running the Benchmarks & Verification Gates

```bash
# 1. Native Rust release benchmark (Lemire p/q hashing, BFP-64, MEM|8 Adaptive, and 100% Zero-Float Q16 Engram step)
cargo run --manifest-path rust/Cargo.toml --release --bin bench

# 2. Native Rust unit tests & strict Clippy gate
cargo test --manifest-path rust/Cargo.toml --release
cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings

# 3. Offline Python unit test gate (zero network calls, runs in < 1s)
python3 -m unittest discover -s tests -v

# 4. Run the Engram forward demo (works on bare Python 3 stdlib OR PyTorch)
python3 engram_demo_v1.py
```

---

## Empirical Benchmark Results

### 1. Multi-Head $N$-Gram Prime Hashing (`1024 tokens × 16 prime heads = 16,384 hashes`)

| Kernel | Latency | Speedup | Bucket Equivalence |
|---|---|---|---|
| **Hardware `% prime` Modulo** (`engram_hash_ngram_mod_into`) | `20.90 µs` | `1.00×` | Baseline |
| **Lemire `p/q` ($q = 2^{64}$) Reciprocal** (`engram_hash_ngram_pq_into`) | **`15.00 µs`** | **`1.39×` – `3.25×`** | **100.0% Bit-Exact** |

> **Platform note:** these hashing latencies were measured on x86-64 Linux
> (`dev-cx53`). The `p/q` win is CPU-dependent: on Apple Silicon
> (aarch64-darwin) the hardware `% prime` path is faster (~`0.9×` for `p/q`).
> Bucket equivalence is platform-independent and asserted by the bench.

### 2. Native 64-Dim `MultiHeadEmbedding` Table Compression (`D_head = 64 == BFP-64 Block Size`)

| Configuration | Rows ($\sum N_h$) | Dense FP32 (`256 B/row`) | Omni `BFP-64` (`65 B/row`) | Compression |
|---|---|---|---|---|
| **Engram Demo (`engram_demo_v1.py`)** | `10,342,604` | `2.65 GB` | **`0.67 GB`** | **`3.94×` smaller** |
| **DeepSeek-V4.1-Flash (`384M` rows)** | `384,006,168` | `98.31 GB` | **`24.96 GB`** | **`3.94×` smaller** |

### 3. Isolated Engram Head Projection (`1024 -> 512`, `16` Pre-Quantized 64-Dim Heads)

| Kernel | Accumulator Path | Latency | Speedup | Relative Error vs FP64 |
|---|---|---|---|---|
| **Dense FP64 Projection** | `f64` FMA | `208.70 µs` | `1.00×` | `0.00000` |
| **Integer-Aligned `BFP-64`** (`engram_bfp64_int_aligned_matmul_into`) | `i16` block $\to$ `i64` cross-head | **`37.11 µs`** | **`5.62×`** | `0.00315` (`0.31%`) |
| **Zero-Float `Q16` `BFP-64`** (`engram_bfp64_int_q16_matmul_into`) | `i16` block $\to$ `i64` $\to$ `Q16` | **`35.65 µs`** | **`5.85×`** | `0.00317` (`0.32%`) |
| **MEM\|8 Adaptive Precision** (`engram_mem8_adaptive_matmul_into`) | `i16` (`BFP-i8`) + `i32` (`BFP-i16`) | **`48.05 µs`** | **`4.34×`** | **`0.00083` (`0.08%`)** |

### 4. Complete Engram Layer Step (`16×64` Gather + `4×KeyProj` + `1×ValueProj` [`2.62M` weights] + Fused Bilinear RMSNorm Gate + ShortConv SiLU)

| Kernel | Floating-Point Ops in Hot Path | Latency | Speedup | Relative Error vs FP64 |
|---|---|---|---|---|
| **Dense FP64 Engram Step** (`engram_step_dense_fp64_into`) | `~5.26M` FP64 ops | `1,282.36 µs` | `1.00×` | `0.00000` |
| **Int-Aligned `BFP-64` + Fused Bilinear Gate** (`engram_step_bfp64_into`) | Boundary scale + gate only | **`190.11 µs`** | **`6.75×`** | `0.00021` (`0.021%`) |
| **100% Zero-Float `p/q` Engram Step (`Q16`)** (`engram_step_pq_no_floats_into`) | **`0` (100% Pure Integer!)** | **`185.41 µs`** | **`6.92×`** | **`0.00010` (`0.010%`)** |
