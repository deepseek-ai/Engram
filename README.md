<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

## conditional memory on the `p/q` · `BFP-64` · `MEM|8` substrate

The unified kernel (`kernel8` · `ayeOS` · `cordis` · `mem8` · `GAIA-MLX-QUANT`) treats precision as a property of the information's requirements, not a fixed tax paid by every operation. In DeepSeek's **Engram** architecture, each $N$-gram lookup head has dimension $D_{\text{head}} = 512 / 8 = 64$ — the **exact block size** of our **Omni `BFP-64`** representation (`64 × i8` mantissas + `1 × i8` shared dyadic exponent = `65 bytes/row`). Retrieving 16 heads (`16 × 64 = 1024`) from a `BFP-64` table yields a pre-quantized 16-block vector with **zero runtime activation quantization overhead**, ready for 16-wide `i16` SIMD ternary accumulation ($|S_{64}| \le 64 \times 127 = 8,128 \le 32,767$), `i64` cross-head integer exponent alignment, and a **100% zero-float `p/q` ("Counting Divided") forward step**.

---

# DeepSeek Engram — 8b-is Sovereign Edition

[![Version](https://img.shields.io/badge/version-0.3.0--pq-blue.svg)](rust/Cargo.toml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-f5de53.svg)](LICENSE)
[![Org](https://img.shields.io/badge/org-8b--is-purple.svg)](https://github.com/8b-is)
[![Upstream](https://img.shields.io/badge/upstream-deepseek--ai%2FEngram-lightgrey.svg)](https://github.com/deepseek-ai/Engram)

<div align="center">
  <img src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/logo.svg?raw=true" width="60%" alt="DeepSeek-V3" />
</div>
<hr>

**`8b-is/Engram`** is the sovereign `p/q` + `BFP-64` + `MEM|8` edition of [`deepseek-ai/Engram`](https://github.com/deepseek-ai/Engram) (*[Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](Engram_paper.pdf)*), aligned with [`8b-is/deepsiper-enthea`](https://github.com/8b-is/deepsiper-enthea) and [`8b-is/alexiai`](https://github.com/8b-is/alexiai).

---

## Fork Enhancements (`8b-is/Engram`)

- **Exact 64-Dim Omni `BFP-64` Head Table (`3.94×` Memory Compression)**:
  Stores each 64-dimensional $N$-gram head row as `64 × i8` mantissas in `[-127, 127]` + `1 × i8` shared power-of-two exponent (`65 bytes/row` vs `256 bytes/row` FP32). Shrinks the DeepSeek Engram demo table (`10.34M` rows) from `2.65 GB` to **`0.67 GB`** and the DeepSeek-V4.1-Flash table (`384.0M` rows) from `98.31 GB` to **`24.96 GB`**.
- **Zero-Quantization $\mathcal{O}(1)$ Gather + `i16 -> i64` Integer-Aligned Projection**:
  Because retrieved Engram heads are *already* quantized into 64-element `BFP-64` blocks in the table, all 5 downstream projections (`value_proj` and `key_projs[0..hc_mult]`) skip activation quantization completely, accumulating 64-element blocks in `i16` (`|S_64| <= 8,128`) and aligning cross-head exponents in pure `i64` integer arithmetic.
- **100% Zero-Float Pure-Integer `p/q` ("Counting Divided") Engram Layer Step (`6.92×` Speedup)**:
  `engram_step_pq_no_floats_into` executes head gather, 5 linear projections (`2.62M` weights), Fused Bilinear `RMSNorm` gating (via exact integer Newton-Raphson `isqrt_u128`), signed-sqrt, rational Padé `[2/2]` sigmoid, and ShortConv SiLU in pure integer `Q16`/`Q20` arithmetic with **zero IEEE-754 floating-point instructions** and **`0.010%` relative error** (`rel_err = 0.00010`).
- **MEM|8 Adaptive Precision (`BFP-i8 -> BFP-i16` Promotion)**:
  Frequent syntactic $N$-gram heads stay in compact `BFP-i8` (`65 B/row`, `i16` accumulator), while high-salience or outlier factual $N$-gram heads whose reconstruction error exceeds $\tau$ automatically promote to `BFP-i16` (`129 B/row`, `i32` accumulator), cutting projection error by **`3.8×`** (`rel_err = 0.00083`).
- **Lemire `p/q` ($q = 2^{64}$) Fast Reciprocal Prime Hashing**:
  Replaces hardware `% prime` division across all `(ngram, head)` prime buckets with precomputed 64-bit dyadic reciprocals (`ceil(2^64 / prime)`), achieving **100% bit-exact bucket equivalence**.
- **Community Upstream PR Integrations (`#13`, `#14`, `#15`, `#27`)**:
  - **PR #13**: Multi-layer `_HASH_MAPPING_CACHE` (`get_or_create_hash_mapping`) avoids rebuilding `CompressedTokenizer` and prime tables across layers.
  - **PR #14**: Vectorized hyper-connection gating (`torch.einsum("bshd,bshd->bsh", ...)`).
  - **PR #15**: Bounds-safe `CompressedTokenizer._compress`, device-aware CPU/GPU tensor placement, and a zero-network `SovereignOfflineTokenizer`.
  - **PR #27**: Pluggable `embedding_factory` hook (`"dense"`, `"bfp64"`, `"mem8_adaptive"`, or custom callable) + offline `unittest` suite (`tests/test_engram_demo_v1.py`).
- **Dual Zero-Dependency Substrates (`rust/` + `engram_pq.py`)**:
  Includes both a `#![no_std]`-first, `#![forbid(unsafe_code)]`, zero-dependency Rust crate (`rust/`) and a 100% Python 3 stdlib engine (`engram_pq.py`) alongside the PyTorch implementation (`engram_demo_v1.py`).

---

## Empirical Benchmark Summary

Full reproduction details and commands are documented in [BENCHMARK.md](BENCHMARK.md).

| Benchmark Stage | Baseline (Dense FP64 / `% prime`) | 8b-is Sovereign (`p/q` + `BFP-64`) | Speedup / Compression | Relative Error |
|---|---|---|---|---|
| **Multi-Head Prime Hashing** (`16,384` hashes) | `20.90 µs` (`% prime`) | **`15.00 µs`** (Lemire $p/q$, $q=2^{64}$) | **`1.39×` – `3.25×` faster** (x86-64 Linux) | **Bit-Exact (`0`)** |
| **Engram Demo Table** (`10.34M × 64`) | `2.65 GB` (FP32) | **`0.67 GB`** (`BFP-64`, `65 B/row`) | **`3.94×` smaller** | `0.31%` |
| **DeepSeek-V4.1 Table** (`384.0M × 64`) | `98.31 GB` (FP32) | **`24.96 GB`** (`BFP-64`, `65 B/row`) | **`3.94×` smaller** | `0.31%` |
| **Isolated Head Projection** (`1024 -> 512`) | `208.70 µs` | **`35.65 µs`** (`i16 -> i64 -> Q16`) | **`5.85×` faster** | `0.00317` |
| **MEM\|8 Adaptive Projection** (`1024 -> 512`) | `208.70 µs` | **`48.05 µs`** (`BFP-i8 -> BFP-i16`) | **`4.34×` faster** | **`0.00083`** |
| **Full Engram Layer Step** (`2.62M` weights + Gate + Conv) | `1,282.36 µs` | **`190.11 µs`** (Int-Aligned `BFP-64`) | **`6.75×` faster** | `0.00021` |
| **100% Zero-Float `p/q` Engram Step** (`Q16`) | `1,282.36 µs` | **`185.41 µs`** (**Zero Float Ops!**) | **`6.92×` faster** | **`0.00010`** |

> Hashing speedups are platform-dependent: on Apple Silicon the `% prime` path
> wins (~`0.9×` for `p/q`). Bucket equivalence and the projection/step error
> numbers are platform-independent; see [BENCHMARK.md](BENCHMARK.md).

---

## 1. Introduction

This repository contains the implementation for the paper: **[Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](Engram_paper.pdf)**.

> **Abstract:** While Mixture-of-Experts (MoE) scales capacity via conditional computation, Transformers lack a native primitive for knowledge lookup. To address this, we explore **conditional memory** as a complementary sparsity axis, instantiated via **Engram**, a module that modernizes classic $N$-gram embeddings for $\mathcal{O}(1)$ lookup.

**Key Contributions:**
- **Sparsity Allocation:** We formulate the trade-off between neural computation (MoE) and static memory (Engram), identifying a U-shaped scaling law that guides optimal capacity allocation.
- **Empirical Verification:** Under strict iso-parameter and iso-FLOPs constraints, the Engram-27B model demonstrates consistent improvements over MoE baselines across knowledge, reasoning, code and math domains.
- **Mechanistic Analysis:** Our analysis suggests that Engram relieves early layers from static pattern reconstruction, potentially preserving effective depth for complex reasoning.
- **System Efficiency:** The module employs deterministic addressing, enabling the offloading of massive embedding tables to host memory with minimal inference overhead.

## 2. Architecture

The Engram module augments the backbone by retrieving static $N$-gram memory and fusing it with dynamic hidden states. The architecture is shown below ([drawio provided](drawio/Engram.drawio)):

<p align="center">
  <img width="75%" src="figures/arch.png" alt="Engram Architecture">
</p>

## 3. Evaluation

### Scaling Law
<p align="center">
  <img width="90%" src="figures/scaling_law.png" alt="Scaling Law">
</p>

---

### Large Scale Pre-training
<p align="center">
  <img width="80%" src="figures/27b_exp_results.png" alt="Pre-training Results">
</p>

---

### Long-context Training
<p align="center">
  <img width="80%" src="figures/long_context_results.png" alt="Long Context Results">
</p>

## 4. Case Study of Engram
<p align="center">
  <img width="80%" src="figures/case.png" alt="Long Context Results">
</p>

## 5. Quick Start

### A. Sovereign Zero-Dependency Rust (`rust/`)

```bash
cargo test --manifest-path rust/Cargo.toml --release
cargo run --manifest-path rust/Cargo.toml --release --bin bench
```

### B. Sovereign Zero-Dependency Python 3 (`engram_pq.py` & `engram_demo_v1.py`)

Works out-of-the-box on bare Python 3 standard library (no network calls or third-party packages required) and automatically uses PyTorch when `torch` is installed:

```bash
python3 engram_demo_v1.py
python3 -m unittest discover -s tests -v
```

### C. Full PyTorch + HuggingFace Environment

```bash
pip install torch numpy transformers sympy
python3 engram_demo_v1.py
```

## 6. License
The use of Engram models is subject to [the Model License](LICENSE).
