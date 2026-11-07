"""
================================================================================
[8b-is Sovereign Engram Substrate — Pure-Stdlib p/q + BFP-64 + MEM|8 Engine]

Implements DeepSeek's Engram Conditional Memory module ("Conditional Memory via
Scalable Lookup: A New Axis of Sparsity for Large Language Models") on the 8b-is
GAIA-MLX-QUANT / deepsiper-enthea mathematical substrate with ZERO third-party
dependencies (100% Python 3 standard library):

1. Exact 64-Dim Head Alignment (`D_head = 64 == BFP-64 block size`):
   DeepSeek's Engram configuration (`n_embed_per_ngram = 512`, `n_head_per_ngram = 8`)
   uses head dimension `D_head = 512 // 8 = 64` — the exact block size of Omni
   `BFP-64` (`64 x int8` mantissas in `[-127, 127]` + `1 x int8` shared dyadic
   exponent = `65 bytes/row` vs `256 bytes/row` FP32 = 3.94x compression).

2. Zero-Quantization O(1) Lookup (`Bfp64MultiHeadEmbedding`):
   Retrieving `16` heads (`16 x 64 = 1024`) from `Bfp64MultiHeadEmbedding` yields
   pre-quantized `BFP-64` blocks `(m_x, exp_x)` with zero floating-point ops.
   Downstream ternary projections (`value_proj`, `key_projs[0..hc_mult]`) execute
   directly on `m_x` using 16-bit integer block accumulators (`|S_64| <= 64 * 127
   = 8,128 <= 32,767 = i16::MAX`) and 64-bit integer cross-head exponent alignment.

3. Lemire p/q (q = 2^64) Fast Reciprocal Prime Hashing (`PrimeBucketMod`):
   Replaces hardware modulo (`mix % prime`) across all `(ngram, head)` prime
   buckets with precomputed 64-bit dyadic reciprocals (`ceil(2^64 / prime)`),
   producing bit-identical bucket indices with multiply-shift arithmetic.

4. MEM|8 Adaptive Precision (`Mem8AdaptiveMultiHeadEmbedding`):
   Common syntactic N-gram heads stay in compact `BFP-i8` (`65 B/row`, `i16`
   accumulator), while high-salience or outlier factual N-gram heads whose
   relative reconstruction error exceeds threshold `tau` promote automatically
   to `BFP-i16` (`129 B/row`, `i32` accumulator).

5. Quantum 64-Trit Bitplane Packing (`Quantum64TritBlock`):
   Packs 64 ternary weights `{-1, 0, +1}` into two 64-bit masks (`pos_mask: u64`,
   `neg_mask: u64` = `16 bytes / 64 weights` = 16.0x compression over FP32).
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import re
import unicodedata
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

ENGRAM_HEAD_DIM: int = 64
U64_MASK: int = 0xFFFF_FFFF_FFFF_FFFF
U32_MASK: int = 0xFFFF_FFFF
PQ_SCALE_Q20: int = 1 << 20  # q = 2^20 rational scale denominator


# ==============================================================================
# 1. Deterministic 64-bit Primality & Lemire p/q (q = 2^64) Reciprocal Hashing
# ==============================================================================

def is_prime_u64(n: int) -> bool:
    """Deterministic Miller-Rabin primality test for 64-bit integers (zero sympy dependency)."""
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    if n in small_primes:
        return True
    for p in small_primes:
        if n % p == 0:
            return False

    d = n - 1
    s = 0
    while (d & 1) == 0:
        d >>= 1
        s += 1

    # Deterministic witness bases sufficient for all n < 2^64
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        composite = True
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                composite = False
                break
        if composite:
            return False
    return True


def find_next_prime(start: int, seen_primes: set) -> int:
    """Find the smallest prime > start not in seen_primes."""
    candidate = start + 1
    if candidate > 2 and (candidate & 1) == 0:
        candidate += 1
    while True:
        if candidate not in seen_primes and is_prime_u64(candidate):
            return candidate
        candidate += 2 if candidate > 2 else 1


@dataclass(frozen=True)
class PrimeBucketMod:
    """
    Precomputed Lemire p/q (q = 2^64) reciprocal for fast prime-bucket modulo.
    Computes `x % prime` for `0 <= x < 2^32` using 128-bit multiply-shift without
    hardware division instructions, guaranteed bit-identical to `x % prime`.
    """
    prime: int
    inv_q64: int
    table_offset: int = 0

    @classmethod
    def create(cls, prime: int, table_offset: int = 0) -> "PrimeBucketMod":
        p = max(1, int(prime))
        inv_q64 = ((U64_MASK // p) + 1) & U64_MASK
        return cls(prime=p, inv_q64=inv_q64, table_offset=table_offset)

    def fast_rem_u64(self, x: int) -> int:
        x_u64 = x & U64_MASK
        q_est = ((x_u64 * self.inv_q64) >> 64) & U64_MASK
        prod = (q_est * self.prime) & U64_MASK
        rem = (x_u64 - prod) & U64_MASK
        if rem >= self.prime:
            rem -= self.prime
            if rem >= self.prime:
                rem %= self.prime
        return rem


# ==============================================================================
# 2. Sovereign Offline Tokenizer & Bounds-Safe CompressedTokenizer (PR #15)
# ==============================================================================

class SovereignOfflineTokenizer:
    """
    Deterministic, zero-network fallback tokenizer for offline testing and demos
    when HuggingFace Hub is unreachable or `tokenizer_name_or_path == "offline"`.
    """

    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = max(256, int(vocab_size))
        self._special = {0: "<bos>", 1: "<eos>", 2: "<pad>", 3: "<unk>"}

    def __len__(self) -> int:
        return self.vocab_size

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        pieces: List[str] = []
        for tid in token_ids:
            if tid in self._special:
                if not skip_special_tokens:
                    pieces.append(self._special[tid])
            elif 4 <= tid < 260:
                pieces.append(chr(tid - 4))
            else:
                # Deterministic canonical merge folds upper/lower pairs to test compression
                base = (tid - 260) % 512
                variant = "A" if ((tid - 260) // 512) % 2 == 0 else "a"
                pieces.append(f"{variant}{base:03d}")
        return "".join(pieces)

    def convert_ids_to_tokens(self, tid: int) -> str:
        return f"<tok_{tid}>"

    def encode(self, text: str) -> List[int]:
        words = re.findall(r"\w+|[^\w\s]", text)
        ids = [0]
        for w in words:
            h = 2166136261
            for ch in w.encode("utf-8"):
                h = ((h ^ ch) * 16777619) & U32_MASK
            ids.append(260 + (h % (self.vocab_size - 260)))
        return ids


def normalize_canonical_text(text: str) -> str:
    """
    Pure-stdlib implementation of DeepSeek Engram's canonical token normalization:
    NFKC -> NFD -> StripAccents -> Lowercase -> Collapse whitespace -> Strip (preserving single space).
    """
    s = unicodedata.normalize("NFKC", text)
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.lower()
    s = re.sub(r"[ \t\r\n]+", " ", s)
    if s == " ":
        return " "
    return s.strip()


class CompressedTokenizerPq:
    """
    Pure-stdlib CompressedTokenizer with bounds-safe `_compress` (upstream PR #15).
    Accepts either a HuggingFace tokenizer instance, `"offline"`, or a custom tokenizer object.
    """

    def __init__(self, tokenizer_or_path: Any = "offline", fallback_vocab_size: int = 4096):
        if isinstance(tokenizer_or_path, str):
            if tokenizer_or_path == "offline":
                self.tokenizer = SovereignOfflineTokenizer(vocab_size=fallback_vocab_size)
            else:
                try:
                    from transformers import AutoTokenizer  # type: ignore
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        tokenizer_or_path, trust_remote_code=True, local_files_only=True
                    )
                except Exception:
                    self.tokenizer = SovereignOfflineTokenizer(vocab_size=fallback_vocab_size)
        else:
            self.tokenizer = tokenizer_or_path

        self.lookup_table, self.num_new_token = self._build_lookup_table()

    def __len__(self) -> int:
        return self.num_new_token

    def _build_lookup_table(self) -> Tuple[List[int], int]:
        old2new: Dict[int, int] = {}
        key2new: Dict[str, int] = {}
        new_tokens: List[str] = []

        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)
            if "\ufffd" in text:
                key = str(self.tokenizer.convert_ids_to_tokens(tid))
            else:
                norm = normalize_canonical_text(text)
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid

        lookup = [old2new[tid] for tid in range(vocab_size)]
        return lookup, len(new_tokens)

    def encode(self, text: str) -> List[int]:
        if hasattr(self.tokenizer, "encode"):
            return [int(x) for x in self.tokenizer.encode(text)]
        return SovereignOfflineTokenizer().encode(text)

    def _compress_row(self, row: Sequence[int]) -> List[int]:
        limit = len(self.lookup_table)
        return [
            self.lookup_table[int(tid)] if 0 <= int(tid) < limit else int(tid)
            for tid in row
        ]

    def __call__(self, input_ids: Sequence[Sequence[int]]) -> List[List[int]]:
        return [self._compress_row(row) for row in input_ids]


# ==============================================================================
# 3. Cached Multi-Head N-Gram Hash Mapping with Lemire p/q Reciprocals (PR #13)
# ==============================================================================

_HASH_MAPPING_PQ_CACHE: Dict[Tuple[Any, ...], "NgramHashMappingPq"] = {}


def clear_hash_mapping_pq_cache() -> None:
    """Clear the global NgramHashMappingPq cache (PR #13)."""
    _HASH_MAPPING_PQ_CACHE.clear()


class NgramHashMappingPq:
    """
    Deterministic multi-head N-gram prime hash mapping supporting both:
    - `use_pq_hash=False`: standard `% prime` modulo (upstream `engram_demo_v1.py` baseline)
    - `use_pq_hash=True`:  Lemire p/q (q = 2^64) reciprocal multiply-shift modulo
    """

    def __init__(
        self,
        engram_vocab_size: Sequence[int],
        max_ngram_size: int,
        n_embed_per_ngram: int,
        n_head_per_ngram: int,
        layer_ids: Sequence[int],
        tokenizer_name_or_path: Any = "offline",
        pad_id: Optional[int] = 2,
        seed: int = 0,
        fallback_vocab_size: int = 4096,
    ):
        self.vocab_size_per_ngram = list(engram_vocab_size)
        self.max_ngram_size = int(max_ngram_size)
        self.n_embed_per_ngram = int(n_embed_per_ngram)
        self.n_head_per_ngram = int(n_head_per_ngram)
        self.layer_ids = list(layer_ids)
        self.seed = int(seed)

        self.compressed_tokenizer = CompressedTokenizerPq(
            tokenizer_or_path=tokenizer_name_or_path,
            fallback_vocab_size=fallback_vocab_size,
        )
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if pad_id is not None and 0 <= int(pad_id) < len(self.compressed_tokenizer.lookup_table):
            self.pad_id = int(self.compressed_tokenizer.lookup_table[int(pad_id)])
        else:
            self.pad_id = int(pad_id or 0)

        max_long = (1 << 63) - 1
        m_max = max_long // max(1, self.tokenizer_vocab_size)
        half_bound = max(1, m_max // 2)
        prime_1 = 10007

        self.layer_multipliers: Dict[int, List[int]] = {}
        for lid in self.layer_ids:
            base_seed = int(self.seed + prime_1 * int(lid))
            rng = random.Random(base_seed)
            self.layer_multipliers[lid] = [
                rng.randrange(0, half_bound) * 2 + 1
                for _ in range(self.max_ngram_size)
            ]

        self.vocab_size_across_layers = self._calculate_vocab_size_across_layers()
        self.pq_buckets_across_layers: Dict[int, List[List[PrimeBucketMod]]] = {}
        for lid in self.layer_ids:
            offset = 0
            layer_buckets: List[List[PrimeBucketMod]] = []
            for ngram_heads in self.vocab_size_across_layers[lid]:
                head_mods: List[PrimeBucketMod] = []
                for p in ngram_heads:
                    head_mods.append(PrimeBucketMod.create(p, table_offset=offset))
                    offset += p
                layer_buckets.append(head_mods)
            self.pq_buckets_across_layers[lid] = layer_buckets

    def _calculate_vocab_size_across_layers(self) -> Dict[int, List[List[int]]]:
        seen_primes: set = set()
        vocab_size_across_layers: Dict[int, List[List[int]]] = {}
        for lid in self.layer_ids:
            all_ngram_vocab_sizes: List[List[int]] = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes: List[int] = []
                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                current_prime_search_start = vocab_size - 1
                for _ in range(self.n_head_per_ngram):
                    found_prime = find_next_prime(current_prime_search_start, seen_primes)
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime
                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[lid] = all_ngram_vocab_sizes
        return vocab_size_across_layers

    def _get_ngram_hashes_for_layer(
        self,
        compressed_ids: List[List[int]],
        layer_id: int,
        use_pq_hash: bool = True,
        fold_32bit: bool = False,
    ) -> List[List[List[int]]]:
        multipliers = self.layer_multipliers[layer_id]
        buckets_by_ngram = self.pq_buckets_across_layers[layer_id]
        out_batch: List[List[List[int]]] = []

        for row in compressed_ids:
            seq_len = len(row)
            seq_hashes: List[List[int]] = []
            for t in range(seq_len):
                token_heads: List[int] = []
                mix = (row[t] * multipliers[0]) & U64_MASK
                for n in range(2, self.max_ngram_size + 1):
                    k = n - 1
                    tok_k = row[t - k] if t >= k else self.pad_id
                    mix = (mix ^ ((tok_k * multipliers[k]) & U64_MASK)) & U64_MASK
                    mix_val = ((mix ^ (mix >> 32)) & U32_MASK) if fold_32bit else mix
                    ngram_buckets = buckets_by_ngram[n - 2]
                    for b in ngram_buckets:
                        if use_pq_hash and fold_32bit:
                            token_heads.append(b.fast_rem_u64(mix_val))
                        else:
                            token_heads.append(mix_val % b.prime)
                seq_hashes.append(token_heads)
            out_batch.append(seq_hashes)
        return out_batch

    def hash(
        self,
        input_ids: Sequence[Sequence[int]],
        use_pq_hash: bool = True,
        fold_32bit: bool = False,
    ) -> Dict[int, List[List[List[int]]]]:
        """
        Hash compressed token IDs for every configured layer.

        Note: `fold_32bit = True` folds the 64-bit mix to 32 bits before the
        modulo (`mix ^ (mix >> 32)`), matching the Rust kernels. The upstream
        `engram_demo_v1.py` baseline hashes the full 64-bit mix, so folded and
        un-folded hashes intentionally differ.
        """
        compressed = self.compressed_tokenizer(input_ids)
        return {
            lid: self._get_ngram_hashes_for_layer(
                compressed, lid, use_pq_hash=use_pq_hash, fold_32bit=fold_32bit
            )
            for lid in self.layer_ids
        }


def get_or_create_hash_mapping_pq(
    engram_vocab_size: Sequence[int],
    max_ngram_size: int,
    n_embed_per_ngram: int,
    n_head_per_ngram: int,
    layer_ids: Sequence[int],
    tokenizer_name_or_path: Any = "offline",
    pad_id: Optional[int] = 2,
    seed: int = 0,
    fallback_vocab_size: int = 4096,
) -> NgramHashMappingPq:
    tok_key = tokenizer_name_or_path if isinstance(tokenizer_name_or_path, str) else id(tokenizer_name_or_path)
    key = (
        tuple(engram_vocab_size),
        int(max_ngram_size),
        int(n_embed_per_ngram),
        int(n_head_per_ngram),
        tuple(layer_ids),
        tok_key,
        pad_id,
        int(seed),
        int(fallback_vocab_size),
    )
    cached = _HASH_MAPPING_PQ_CACHE.get(key)
    if cached is None:
        cached = NgramHashMappingPq(
            engram_vocab_size=engram_vocab_size,
            max_ngram_size=max_ngram_size,
            n_embed_per_ngram=n_embed_per_ngram,
            n_head_per_ngram=n_head_per_ngram,
            layer_ids=layer_ids,
            tokenizer_name_or_path=tokenizer_name_or_path,
            pad_id=pad_id,
            seed=seed,
            fallback_vocab_size=fallback_vocab_size,
        )
        _HASH_MAPPING_PQ_CACHE[key] = cached
    return cached


# ==============================================================================
# 4. Omni BFP-64 (65 B/row) & MEM|8 Adaptive (i8 -> i16) MultiHeadEmbedding
# ==============================================================================

@dataclass
class Bfp64HeadRow:
    """
    Single 64-dimensional Engram head row in Omni BFP-64 format:
    `64 x int8` mantissas in `[-127, 127]` + `1 x int8` shared dyadic exponent (`65 bytes/row`).
    """
    mantissas: List[int]  # length D (default 64), values in [-127, 127]
    exponent: int         # shared power-of-two exponent e in [-60, 60]

    @classmethod
    def from_floats(cls, values: Sequence[float]) -> "Bfp64HeadRow":
        max_abs = max((abs(v) for v in values), default=0.0)
        if max_abs < 1e-18:
            return cls(mantissas=[0] * len(values), exponent=-60)
        exp_val = max(-60, min(60, math.ceil(math.log2(max_abs / 127.0))))
        inv_scale = math.ldexp(1.0, -exp_val)
        m = [max(-127, min(127, int(round(v * inv_scale)))) for v in values]
        return cls(mantissas=m, exponent=exp_val)

    def dequantize(self) -> List[float]:
        scale = math.ldexp(1.0, self.exponent)
        return [m * scale for m in self.mantissas]


@dataclass
class Mem8AdaptiveHeadRow:
    """
    MEM|8 Adaptive Precision head row:
    - Stays in compact `BFP-i8` (`65 B/row`, `i16` accumulator) when relative error <= `tau`
    - Promotes to `BFP-i16` (`129 B/row`, `i32` accumulator) for high-salience/outlier N-grams
    """
    mantissas: List[int]
    exponent: int
    promoted_i16: bool

    @classmethod
    def from_floats(cls, values: Sequence[float], tau: float = 0.0052) -> "Mem8AdaptiveHeadRow":
        bfp8 = Bfp64HeadRow.from_floats(values)
        recon8 = bfp8.dequantize()
        err_sq = sum((a - b) * (a - b) for a, b in zip(values, recon8))
        sig_sq = max(1e-24, sum(a * a for a in values))
        rel_err = math.sqrt(err_sq / sig_sq)
        if rel_err <= tau:
            return cls(mantissas=bfp8.mantissas, exponent=bfp8.exponent, promoted_i16=False)

        max_abs = max((abs(v) for v in values), default=0.0)
        exp16 = max(-60, min(60, math.ceil(math.log2(max(1e-18, max_abs) / 32767.0))))
        inv_scale16 = math.ldexp(1.0, -exp16)
        m16 = [max(-32767, min(32767, int(round(v * inv_scale16)))) for v in values]
        return cls(mantissas=m16, exponent=exp16, promoted_i16=True)

    def dequantize(self) -> List[float]:
        scale = math.ldexp(1.0, self.exponent)
        return [m * scale for m in self.mantissas]


class Bfp64MultiHeadEmbeddingPq:
    """
    Omni BFP-64 MultiHeadEmbedding table (`65 bytes/row` vs `256 bytes/row` FP32 = 3.94x smaller).
    Supports both dequantized float forward and zero-float-op `gather_bfp64_blocks` for direct
    `i16 -> i64` integer-aligned ternary projection!
    """

    def __init__(self, list_of_N: Sequence[int], D: int = ENGRAM_HEAD_DIM, seed: int = 0):
        self.list_of_N = list(list_of_N)
        self.num_heads = len(self.list_of_N)
        self.embedding_dim = int(D)
        self.offsets: List[int] = [0]
        for n in self.list_of_N[:-1]:
            self.offsets.append(self.offsets[-1] + int(n))
        self.total_N = sum(self.list_of_N)

        rng = random.Random(seed + 17)
        self.rows: List[Bfp64HeadRow] = []
        for _ in range(self.total_N):
            floats = [(rng.random() - 0.5) * 0.2 for _ in range(self.embedding_dim)]
            self.rows.append(Bfp64HeadRow.from_floats(floats))

    @property
    def nbytes_bfp64(self) -> int:
        return self.total_N * (self.embedding_dim + 1)

    @property
    def nbytes_fp32(self) -> int:
        return self.total_N * self.embedding_dim * 4

    @property
    def compression_ratio_vs_fp32(self) -> float:
        return self.nbytes_fp32 / max(1, self.nbytes_bfp64)

    def gather_bfp64_blocks(
        self, head_ids_for_token: Sequence[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Zero-float-op O(1) gather of `num_heads` pre-quantized BFP-64 blocks:
        returns `(m_x, exp_x)` where `m_x` has length `num_heads * D` (`i8`)
        and `exp_x` has length `num_heads` (`i8`).
        """
        m_x: List[int] = []
        exp_x: List[int] = []
        for h, hid in enumerate(head_ids_for_token):
            row = self.rows[self.offsets[h] + int(hid)]
            m_x.extend(row.mantissas)
            exp_x.append(row.exponent)
        return m_x, exp_x

    def __call__(self, hash_input_ids: Sequence[Sequence[Sequence[int]]]) -> List[List[List[List[float]]]]:
        out: List[List[List[List[float]]]] = []
        for batch_seq in hash_input_ids:
            seq_out: List[List[List[float]]] = []
            for token_heads in batch_seq:
                heads_out: List[List[float]] = []
                for h, hid in enumerate(token_heads):
                    row = self.rows[self.offsets[h] + int(hid)]
                    heads_out.append(row.dequantize())
                seq_out.append(heads_out)
            out.append(seq_out)
        return out


class Mem8AdaptiveMultiHeadEmbeddingPq:
    """
    MEM|8 Adaptive Precision MultiHeadEmbedding (`BFP-i8` -> `BFP-i16` promotion).
    """

    def __init__(
        self,
        list_of_N: Sequence[int],
        D: int = ENGRAM_HEAD_DIM,
        tau: float = 0.0052,
        seed: int = 0,
    ):
        self.list_of_N = list(list_of_N)
        self.num_heads = len(self.list_of_N)
        self.embedding_dim = int(D)
        self.tau = float(tau)
        self.offsets: List[int] = [0]
        for n in self.list_of_N[:-1]:
            self.offsets.append(self.offsets[-1] + int(n))
        self.total_N = sum(self.list_of_N)

        rng = random.Random(seed + 17)
        self.rows: List[Mem8AdaptiveHeadRow] = []
        for _ in range(self.total_N):
            floats = [(rng.random() - 0.5) * 0.2 for _ in range(self.embedding_dim)]
            self.rows.append(Mem8AdaptiveHeadRow.from_floats(floats, tau=self.tau))

    @property
    def promoted_count(self) -> int:
        return sum(1 for r in self.rows if r.promoted_i16)

    def __call__(self, hash_input_ids: Sequence[Sequence[Sequence[int]]]) -> List[List[List[List[float]]]]:
        out: List[List[List[List[float]]]] = []
        for batch_seq in hash_input_ids:
            seq_out: List[List[List[float]]] = []
            for token_heads in batch_seq:
                heads_out = [
                    self.rows[self.offsets[h] + int(hid)].dequantize()
                    for h, hid in enumerate(token_heads)
                ]
                seq_out.append(heads_out)
            out.append(seq_out)
        return out


# ==============================================================================
# 5. Quantum 64-Trit Bitplane & Deferred-Scale Integer-Aligned Projection
# ==============================================================================

@dataclass(frozen=True)
class Quantum64TritBlock:
    """
    Packs 64 ternary weights `w_k in {-1, 0, +1}` into two 64-bit bitmasks
    (`pos_mask: u64`, `neg_mask: u64` = 16 bytes vs 256 bytes FP32 = 16x compression).
    """
    pos_mask: int
    neg_mask: int

    @classmethod
    def pack(cls, signs_64: Sequence[int]) -> "Quantum64TritBlock":
        pos = 0
        neg = 0
        for idx, s in enumerate(signs_64[:64]):
            if s > 0:
                pos |= (1 << idx)
            elif s < 0:
                neg |= (1 << idx)
        return cls(pos_mask=pos, neg_mask=neg)

    def dot_i16(self, mx_64: Sequence[int]) -> int:
        """
        Compute exact 64-element ternary x i8 dot product in i16 range (`[-8128, 8128]`).
        """
        acc = 0
        pos = self.pos_mask
        neg = self.neg_mask
        while pos:
            lsb = pos & -pos
            idx = lsb.bit_length() - 1
            acc += mx_64[idx]
            pos ^= lsb
        while neg:
            lsb = neg & -neg
            idx = lsb.bit_length() - 1
            acc -= mx_64[idx]
            neg ^= lsb
        return acc


class TernaryLinearPq:
    """
    Ternary `{-1, 0, +1}` linear projection with group-64 rational `p/q` (`q = 2^20`)
    scales and Deferred-Scale Integer-Aligned `BFP-64` (`i16 -> i64`) accumulation!
    """

    def __init__(self, in_features: int, out_features: int, seed: int = 0):
        assert in_features % ENGRAM_HEAD_DIM == 0, "in_features must be a multiple of 64"
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_groups = self.in_features // ENGRAM_HEAD_DIM

        rng = random.Random(seed + 101)
        self.blocks: List[List[Quantum64TritBlock]] = []
        self.scales_pq: List[List[int]] = []
        self.scales_f64: List[List[float]] = []

        for _ in range(self.out_features):
            row_blocks: List[Quantum64TritBlock] = []
            row_spq: List[int] = []
            row_sf: List[float] = []
            for _ in range(self.num_groups):
                signs = [rng.choice((-1, 0, 1)) for _ in range(ENGRAM_HEAD_DIM)]
                row_blocks.append(Quantum64TritBlock.pack(signs))
                scale = 0.015 + rng.random() * 0.01
                row_sf.append(scale)
                row_spq.append(int(round(scale * PQ_SCALE_Q20)))
            self.blocks.append(row_blocks)
            self.scales_pq.append(row_spq)
            self.scales_f64.append(row_sf)

    def forward_prequant_bfp64(self, m_x: Sequence[int], exp_x: Sequence[int]) -> List[float]:
        """
        Project pre-quantized BFP-64 Engram heads `(m_x, exp_x)` directly via
        `i16` block accumulation -> `i64` cross-head integer alignment!
        """
        min_exp = min(exp_x[:self.num_groups])
        head_mult = [
            1 << max(0, min(24, int(exp_x[g]) - int(min_exp)))
            for g in range(self.num_groups)
        ]
        boundary_scale = math.ldexp(1.0, int(min_exp) - 20)

        mx_groups = [
            m_x[g * ENGRAM_HEAD_DIM:(g + 1) * ENGRAM_HEAD_DIM]
            for g in range(self.num_groups)
        ]

        out = [0.0] * self.out_features
        for j in range(self.out_features):
            row_blocks = self.blocks[j]
            row_spq = self.scales_pq[j]
            row_i64 = 0
            for g in range(self.num_groups):
                block_sum_i16 = row_blocks[g].dot_i16(mx_groups[g])
                row_i64 += block_sum_i16 * row_spq[g] * head_mult[g]
            out[j] = row_i64 * boundary_scale
        return out

    def __call__(self, x_vec: Sequence[float]) -> List[float]:
        m_x: List[int] = []
        exp_x: List[int] = []
        for g in range(self.num_groups):
            blk = Bfp64HeadRow.from_floats(x_vec[g * ENGRAM_HEAD_DIM:(g + 1) * ENGRAM_HEAD_DIM])
            m_x.extend(blk.mantissas)
            exp_x.append(blk.exponent)
        return self.forward_prequant_bfp64(m_x, exp_x)


# ==============================================================================
# 6. Fused Bilinear RMSNorm Gate & Complete Pure-Stdlib Engram Layer
# ==============================================================================

def fused_rmsnorm_gate(
    key: Sequence[float],
    query: Sequence[float],
    norm_k_w: Sequence[float],
    norm_q_w: Sequence[float],
    eps: float = 1e-5,
) -> float:
    """
    Fused single-pass Bilinear RMSNorm hyper-connection gate:
    computes `<RMSNorm(k), RMSNorm(q)> / sqrt(D)` without allocating intermediate
    normalized `k` and `q` vectors, followed by signed-sqrt and sigmoid.
    """
    d = len(key)
    sum_k2 = 0.0
    sum_q2 = 0.0
    weighted_dot = 0.0
    for k_i, q_i, wk_i, wq_i in zip(key, query, norm_k_w, norm_q_w):
        sum_k2 += k_i * k_i
        sum_q2 += q_i * q_i
        weighted_dot += (wk_i * wq_i) * (k_i * q_i)

    rms_k = math.sqrt(sum_k2 / d + eps)
    rms_q = math.sqrt(sum_q2 / d + eps)
    g = weighted_dot / (math.sqrt(d) * rms_k * rms_q)
    abs_g = max(abs(g), 1e-6)
    signed_sqrt = math.sqrt(abs_g) if g >= 0.0 else -math.sqrt(abs_g)
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, signed_sqrt))))


class EngramPq:
    """
    Pure-stdlib DeepSeek Engram module powered by Omni BFP-64 + Ternary p/q projections.

    `fold_32bit` controls the mix folding used during hashing; the default
    (`True`) matches the Rust kernels, while `False` hashes the full 64-bit mix
    like the upstream demo.
    """

    def __init__(
        self,
        layer_id: int,
        engram_vocab_size: Sequence[int] = (2048, 2048),
        max_ngram_size: int = 3,
        n_embed_per_ngram: int = 512,
        n_head_per_ngram: int = 8,
        layer_ids: Sequence[int] = (1, 15),
        tokenizer_name_or_path: Any = "offline",
        pad_id: int = 2,
        seed: int = 0,
        hidden_size: int = 256,
        hc_mult: int = 4,
        embedding_mode: str = "bfp64",
        fold_32bit: bool = True,
    ):
        self.layer_id = int(layer_id)
        self.hidden_size = int(hidden_size)
        self.hc_mult = int(hc_mult)
        self.fold_32bit = bool(fold_32bit)
        self.engram_hidden_size = (int(max_ngram_size) - 1) * int(n_embed_per_ngram)
        head_dim = int(n_embed_per_ngram) // int(n_head_per_ngram)

        self.hash_mapping = get_or_create_hash_mapping_pq(
            engram_vocab_size=engram_vocab_size,
            max_ngram_size=max_ngram_size,
            n_embed_per_ngram=n_embed_per_ngram,
            n_head_per_ngram=n_head_per_ngram,
            layer_ids=layer_ids,
            tokenizer_name_or_path=tokenizer_name_or_path,
            pad_id=pad_id,
            seed=seed,
        )
        list_of_n = [
            x for y in self.hash_mapping.vocab_size_across_layers[self.layer_id] for x in y
        ]
        if embedding_mode == "mem8_adaptive":
            self.multi_head_embedding: Any = Mem8AdaptiveMultiHeadEmbeddingPq(
                list_of_n, D=head_dim, seed=seed + self.layer_id
            )
        else:
            self.multi_head_embedding = Bfp64MultiHeadEmbeddingPq(
                list_of_n, D=head_dim, seed=seed + self.layer_id
            )

        self.value_proj = TernaryLinearPq(
            self.engram_hidden_size, self.hidden_size, seed=seed + 100 + self.layer_id
        )
        self.key_projs = [
            TernaryLinearPq(
                self.engram_hidden_size,
                self.hidden_size,
                seed=seed + 200 + self.layer_id * 16 + hc,
            )
            for hc in range(self.hc_mult)
        ]
        self.norm1_w = [[1.0] * self.hidden_size for _ in range(self.hc_mult)]
        self.norm2_w = [[1.0] * self.hidden_size for _ in range(self.hc_mult)]
        self.short_conv_w = [[0.25] * self.hidden_size for _ in range(self.hc_mult)]

    def forward(
        self,
        hidden_states: List[List[List[List[float]]]],
        input_ids: Sequence[Sequence[int]],
    ) -> List[List[List[List[float]]]]:
        """
        hidden_states: [B][L][HC_MULT][D]
        input_ids:     [B][L]
        returns:       [B][L][HC_MULT][D]
        """
        hash_batches = self.hash_mapping.hash(
            input_ids, use_pq_hash=True, fold_32bit=self.fold_32bit
        )[self.layer_id]
        out_batches: List[List[List[List[float]]]] = []

        for b_idx, seq_hashes in enumerate(hash_batches):
            seq_out: List[List[List[float]]] = []
            for t_idx, token_heads in enumerate(seq_hashes):
                if isinstance(self.multi_head_embedding, Bfp64MultiHeadEmbeddingPq):
                    m_x, exp_x = self.multi_head_embedding.gather_bfp64_blocks(token_heads)
                    val_vec = self.value_proj.forward_prequant_bfp64(m_x, exp_x)
                    key_vecs = [
                        kp.forward_prequant_bfp64(m_x, exp_x) for kp in self.key_projs
                    ]
                else:
                    deq_heads = self.multi_head_embedding([[token_heads]])[0][0]
                    flat_emb = [v for head in deq_heads for v in head]
                    val_vec = self.value_proj(flat_emb)
                    key_vecs = [kp(flat_emb) for kp in self.key_projs]

                hc_out: List[List[float]] = []
                for hc in range(self.hc_mult):
                    query_hc = hidden_states[b_idx][t_idx][hc]
                    gate = fused_rmsnorm_gate(
                        key_vecs[hc],
                        query_hc,
                        self.norm1_w[hc],
                        self.norm2_w[hc],
                    )
                    conv_w = self.short_conv_w[hc]
                    stream_out: List[float] = []
                    for d in range(self.hidden_size):
                        v_gated = gate * val_vec[d]
                        conv_val = v_gated * conv_w[d]
                        silu = conv_val / (1.0 + math.exp(-max(-40.0, min(40.0, conv_val))))
                        stream_out.append(v_gated + silu)
                    hc_out.append(stream_out)
                seq_out.append(hc_out)
            out_batches.append(seq_out)
        return out_batches
