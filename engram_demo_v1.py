"""
================================================================================
[Engram Architecture Demo Implementation — 8b-is Sovereign p/q + BFP-64 Edition]

Includes:
1. Upstream DeepSeek Engram Conditional Memory architecture (`engram_demo_v1.py`)
2. Multi-layer `NgramHashMapping` cache (`_HASH_MAPPING_CACHE`, PR #13)
3. Vectorized Hyper-Connection Gating (`torch.einsum` across `hc_mult`, PR #14)
4. Bounds-safe `CompressedTokenizer` & device-aware CPU/GPU hashing + offline
   sovereign fallback tokenizer (PR #15)
5. Pluggable `embedding_factory` hook (`"dense"`, `"bfp64"`, `"mem8_adaptive"`,
   or custom callable, PR #27)
6. 8b-is GAIA-MLX-QUANT / deepsiper-enthea `p/q` ("Counting Divided") +
   Omni `BFP-64` (`65 B/row` = 3.94x table compression, zero-quant O(1) lookup,
   `i16 -> i64` integer-aligned ternary projection) + Lemire `p/q` (`q = 2^64`)
   reciprocal prime hashing (`engram_pq.py`).
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from engram_pq import (
    ENGRAM_HEAD_DIM,
    PQ_SCALE_Q20,
    U32_MASK,
    U64_MASK,
    Bfp64HeadRow,
    Bfp64MultiHeadEmbeddingPq,
    CompressedTokenizerPq,
    EngramPq,
    Mem8AdaptiveHeadRow,
    Mem8AdaptiveMultiHeadEmbeddingPq,
    NgramHashMappingPq,
    PrimeBucketMod,
    Quantum64TritBlock,
    SovereignOfflineTokenizer,
    TernaryLinearPq,
    clear_hash_mapping_pq_cache,
    find_next_prime as _find_next_prime_pq,
    fused_rmsnorm_gate,
    get_or_create_hash_mapping_pq,
    is_prime_u64,
    normalize_canonical_text,
)

try:
    import numpy as np  # type: ignore
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    HAS_TORCH = True
except ImportError:
    np = None  # type: ignore
    torch = None  # type: ignore
    nn = None  # type: ignore
    HAS_TORCH = False

try:
    from sympy import isprime as _sympy_isprime  # type: ignore
except ImportError:
    _sympy_isprime = is_prime_u64

try:
    from transformers import AutoTokenizer  # type: ignore
    from tokenizers import Regex, normalizers  # type: ignore
    HAS_HF_TOKENIZERS = True
except ImportError:
    AutoTokenizer = None  # type: ignore
    Regex = None  # type: ignore
    normalizers = None  # type: ignore
    HAS_HF_TOKENIZERS = False


@dataclass
class EngramConfig:
    tokenizer_name_or_path: Any = "deepseek-ai/DeepSeek-V3"
    engram_vocab_size: List[int] = field(default_factory=lambda: [129280 * 5, 129280 * 5])
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    layer_ids: List[int] = field(default_factory=lambda: [1, 15])
    pad_id: int = 2
    seed: int = 0
    kernel_size: int = 4
    embedding_mode: str = "dense"  # "dense", "bfp64", or "mem8_adaptive"
    proj_mode: str = "dense"       # "dense" or "ternary_pq"
    use_pq_hash: bool = False      # Set True for Lemire p/q (q=2^64) reciprocal hashing


@dataclass
class BackBoneConfig:
    hidden_size: int = 1024
    hc_mult: int = 4
    vocab_size: int = 129280
    num_layers: int = 30


engram_cfg = EngramConfig()
backbone_config = BackBoneConfig()

_HASH_MAPPING_CACHE: Dict[Tuple[Any, ...], "NgramHashMapping"] = {}


def clear_hash_mapping_cache() -> None:
    """Clear the cached NgramHashMapping instances (PR #13)."""
    _HASH_MAPPING_CACHE.clear()
    clear_hash_mapping_pq_cache()


def find_next_prime(start: int, seen_primes: set) -> int:
    candidate = start + 1
    while True:
        if _sympy_isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


if HAS_TORCH:

    def _fast_rem_u64_vec(x: np.ndarray, bucket: PrimeBucketMod) -> np.ndarray:
        """
        Vectorized Lemire p/q (q = 2^64) fast remainder, bit-exact with
        `PrimeBucketMod.fast_rem_u64` for `0 <= x < 2^32`.
        """
        xu = x.astype(np.uint64)
        inv = np.uint64(bucket.inv_q64)
        p = np.uint64(bucket.prime)
        hi = xu * (inv >> np.uint64(32))
        lo = xu * (inv & np.uint64(0xFFFFFFFF))
        carry = (
            (hi & np.uint64(0xFFFFFFFF)) + (lo >> np.uint64(32))
            >= (np.uint64(1) << np.uint64(32))
        ).astype(np.uint64)
        q = (hi >> np.uint64(32)) + carry
        rem = xu - q * p
        rem = np.where(rem >= p, rem - p, rem)
        rem = np.where(rem >= p, rem - p, rem)
        return rem.astype(np.int64)

    class CompressedTokenizer:
        """
        Canonicalizer + lookup table compressor for token IDs.
        Includes bounds-safe `_compress` (PR #15) and sovereign offline fallback
        when HuggingFace Hub is unreachable or a pre-built tokenizer object is passed.
        """

        def __init__(self, tokenizer_name_or_path: Any):
            if isinstance(tokenizer_name_or_path, str):
                if tokenizer_name_or_path == "offline" or not HAS_HF_TOKENIZERS:
                    self.tokenizer = SovereignOfflineTokenizer()
                else:
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            tokenizer_name_or_path, trust_remote_code=True
                        )
                    except Exception:
                        self.tokenizer = SovereignOfflineTokenizer()
            else:
                self.tokenizer = tokenizer_name_or_path

            if HAS_HF_TOKENIZERS and normalizers is not None and Regex is not None:
                sentinel = "\uE000"
                self.normalizer = normalizers.Sequence([
                    normalizers.NFKC(),
                    normalizers.NFD(),
                    normalizers.StripAccents(),
                    normalizers.Lowercase(),
                    normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
                    normalizers.Replace(Regex(r"^ $"), sentinel),
                    normalizers.Strip(),
                    normalizers.Replace(sentinel, " "),
                ])
            else:
                self.normalizer = None

            self.lookup_table, self.num_new_token = self._build_lookup_table()

        def __len__(self) -> int:
            return self.num_new_token

        def _normalize(self, text: str) -> str:
            if self.normalizer is not None:
                return self.normalizer.normalize_str(text)
            return normalize_canonical_text(text)

        def _build_lookup_table(self) -> Tuple[np.ndarray, int]:
            old2new: Dict[int, int] = {}
            key2new: Dict[str, int] = {}
            new_tokens: List[str] = []

            vocab_size = len(self.tokenizer)
            for tid in range(vocab_size):
                text = self.tokenizer.decode([tid], skip_special_tokens=False)
                if "\ufffd" in text:
                    key = str(self.tokenizer.convert_ids_to_tokens(tid))
                else:
                    norm = self._normalize(text)
                    key = norm if norm else text

                nid = key2new.get(key)
                if nid is None:
                    nid = len(new_tokens)
                    key2new[key] = nid
                    new_tokens.append(key)
                old2new[tid] = nid

            lookup = np.empty(vocab_size, dtype=np.int64)
            for tid in range(vocab_size):
                lookup[tid] = old2new[tid]
            return lookup, len(new_tokens)

        def _compress(self, input_ids: Any) -> np.ndarray:
            if isinstance(input_ids, torch.Tensor):
                arr = input_ids.detach().cpu().numpy().astype(np.int64, copy=False)
            else:
                arr = np.asarray(input_ids, dtype=np.int64)
            # Bounds-safe mask for both negative padding IDs and out-of-range IDs (PR #15)
            valid_mask = (arr >= 0) & (arr < len(self.lookup_table))
            out = arr.copy()
            valid_ids = arr[valid_mask]
            out[valid_mask] = self.lookup_table[valid_ids]
            return out

        def __call__(self, input_ids: Any) -> np.ndarray:
            return self._compress(input_ids)

    class ShortConv(nn.Module):
        def __init__(
            self,
            hidden_size: int,
            kernel_size: int = 4,
            dilation: int = 1,
            norm_eps: float = 1e-5,
            hc_mult: int = 4,
            activation: bool = True,
        ):
            super().__init__()
            self.hc_mult = hc_mult
            self.activation = activation

            total_channels = hidden_size * hc_mult
            self.conv = nn.Conv1d(
                in_channels=total_channels,
                out_channels=total_channels,
                kernel_size=kernel_size,
                groups=total_channels,
                bias=False,
                padding=(kernel_size - 1) * dilation,
                dilation=dilation,
            )
            self.norms = nn.ModuleList([
                nn.RMSNorm(hidden_size, eps=norm_eps)
                for _ in range(hc_mult)
            ])
            if self.activation:
                self.act_fn = nn.SiLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, T, G, C = x.shape
            assert G == self.hc_mult, f"Input groups {G} != hc_mult {self.hc_mult}"

            normed_chunks = [self.norms[i](x[:, :, i, :]) for i in range(G)]
            x_norm = torch.cat(normed_chunks, dim=-1)
            x_bct = x_norm.transpose(1, 2)
            y_bct = self.conv(x_bct)[..., :T]
            if self.activation:
                y_bct = self.act_fn(y_bct)
            return y_bct.transpose(1, 2).view(B, T, G, C).contiguous()

    class NgramHashMapping:
        def __init__(
            self,
            engram_vocab_size: Sequence[int],
            max_ngram_size: int,
            n_embed_per_ngram: int,
            n_head_per_ngram: int,
            layer_ids: Sequence[int],
            tokenizer_name_or_path: Any,
            pad_id: Optional[int],
            seed: int,
            use_pq_hash: bool = False,
        ):
            self.vocab_size_per_ngram = list(engram_vocab_size)
            self.max_ngram_size = max_ngram_size
            self.n_embed_per_ngram = n_embed_per_ngram
            self.n_head_per_ngram = n_head_per_ngram
            self.pad_id = pad_id
            self.layer_ids = list(layer_ids)
            self.use_pq_hash = bool(use_pq_hash)

            self.compressed_tokenizer = CompressedTokenizer(
                tokenizer_name_or_path=tokenizer_name_or_path
            )
            self.tokenizer_vocab_size = len(self.compressed_tokenizer)
            if self.pad_id is not None and 0 <= int(self.pad_id) < len(self.compressed_tokenizer.lookup_table):
                self.pad_id = int(self.compressed_tokenizer.lookup_table[int(self.pad_id)])

            max_long = np.iinfo(np.int64).max
            M_max = int(max_long // max(1, self.tokenizer_vocab_size))
            half_bound = max(1, M_max // 2)
            PRIME_1 = 10007

            self.layer_multipliers: Dict[int, np.ndarray] = {}
            for lid in self.layer_ids:
                base_seed = int(seed + PRIME_1 * int(lid))
                g = np.random.default_rng(base_seed)
                r = g.integers(
                    low=0,
                    high=half_bound,
                    size=(self.max_ngram_size,),
                    dtype=np.int64,
                )
                self.layer_multipliers[lid] = r * 2 + 1

            self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()
            self.pq_buckets_across_layers: Dict[int, List[List[PrimeBucketMod]]] = {}
            for lid in self.layer_ids:
                offset = 0
                layer_buckets: List[List[PrimeBucketMod]] = []
                for ngram_heads in self.vocab_size_across_layers[lid]:
                    head_mods = []
                    for p in ngram_heads:
                        head_mods.append(PrimeBucketMod.create(int(p), table_offset=offset))
                        offset += int(p)
                    layer_buckets.append(head_mods)
                self.pq_buckets_across_layers[lid] = layer_buckets

        def calculate_vocab_size_across_layers(self) -> Dict[int, List[List[int]]]:
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

        def _get_ngram_hashes(
            self,
            input_ids: np.ndarray,
            layer_id: int,
        ) -> np.ndarray:
            x = np.asarray(input_ids, dtype=np.int64)
            B, T = x.shape
            multipliers = self.layer_multipliers[layer_id]

            def shift_k(k: int) -> np.ndarray:
                if k == 0:
                    return x
                return np.pad(
                    x, ((0, 0), (k, 0)), mode="constant", constant_values=self.pad_id
                )[:, :T]

            base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]
            all_hashes = []

            for n in range(2, self.max_ngram_size + 1):
                n_gram_index = n - 2
                tokens = base_shifts[:n]
                mix = tokens[0] * multipliers[0]
                for k in range(1, n):
                    mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
                head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]
                pq_buckets = self.pq_buckets_across_layers[layer_id][n_gram_index]

                if self.use_pq_hash:
                    mix_u32 = ((mix.astype(np.uint64) ^ (mix.astype(np.uint64) >> np.uint64(32))) & np.uint64(0xFFFFFFFF))
                    for j in range(self.n_head_per_ngram):
                        b = pq_buckets[j]
                        rem = _fast_rem_u64_vec(mix_u32.reshape(-1), b).reshape(B, T)
                        all_hashes.append(rem)
                else:
                    for j in range(self.n_head_per_ngram):
                        mod = int(head_vocab_sizes[j])
                        head_hash = mix % mod
                        all_hashes.append(head_hash.astype(np.int64, copy=False))

            return np.stack(all_hashes, axis=2)

        def hash(self, input_ids: Any) -> Dict[int, np.ndarray]:
            compressed = self.compressed_tokenizer(input_ids)
            return {
                lid: self._get_ngram_hashes(compressed, layer_id=lid)
                for lid in self.layer_ids
            }

    def get_or_create_hash_mapping(
        engram_vocab_size: Sequence[int],
        max_ngram_size: int,
        n_embed_per_ngram: int,
        n_head_per_ngram: int,
        layer_ids: Sequence[int],
        tokenizer_name_or_path: Any,
        pad_id: Optional[int],
        seed: int,
        use_pq_hash: bool = False,
    ) -> NgramHashMapping:
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
            bool(use_pq_hash),
        )
        cached = _HASH_MAPPING_CACHE.get(key)
        if cached is None:
            cached = NgramHashMapping(
                engram_vocab_size=engram_vocab_size,
                max_ngram_size=max_ngram_size,
                n_embed_per_ngram=n_embed_per_ngram,
                n_head_per_ngram=n_head_per_ngram,
                layer_ids=layer_ids,
                tokenizer_name_or_path=tokenizer_name_or_path,
                pad_id=pad_id,
                seed=seed,
                use_pq_hash=use_pq_hash,
            )
            _HASH_MAPPING_CACHE[key] = cached
        return cached

    class MultiHeadEmbedding(nn.Module):
        def __init__(self, list_of_N: List[int], D: int):
            super().__init__()
            self.num_heads = len(list_of_N)
            self.embedding_dim = D

            offsets = [0]
            for n in list_of_N[:-1]:
                offsets.append(offsets[-1] + n)
            self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

            total_N = sum(list_of_N)
            self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            shifted_input_ids = input_ids + self.offsets
            return self.embedding(shifted_input_ids)

    class TorchBfp64MultiHeadEmbedding(nn.Module):
        """
        Omni BFP-64 MultiHeadEmbedding in PyTorch:
        Stores `total_N x D` head rows as `int8` mantissas `[total_N, D]` + `int8` shared
        exponent `[total_N, 1]` = `65 bytes/row` (for `D = 64`) vs `256 bytes/row` FP32
        = 3.94x memory reduction, with O(1) pre-quantized `(m_x, exp_x)` block gather!
        """

        def __init__(self, list_of_N: List[int], D: int = ENGRAM_HEAD_DIM):
            super().__init__()
            self.num_heads = len(list_of_N)
            self.embedding_dim = D
            offsets = [0]
            for n in list_of_N[:-1]:
                offsets.append(offsets[-1] + n)
            self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

            total_N = sum(list_of_N)
            self.total_N = total_N
            init_m = torch.randint(-64, 65, (total_N, D), dtype=torch.int8)
            init_e = torch.full((total_N, 1), -9, dtype=torch.int8)
            self.register_buffer("mantissas", init_m)
            self.register_buffer("exponents", init_e)

        @classmethod
        def from_float_embedding(cls, dense_emb: MultiHeadEmbedding) -> "TorchBfp64MultiHeadEmbedding":
            total_N, D = dense_emb.embedding.weight.shape
            obj = cls.__new__(cls)
            nn.Module.__init__(obj)
            obj.num_heads = dense_emb.num_heads
            obj.embedding_dim = D
            obj.total_N = total_N
            obj.register_buffer("offsets", dense_emb.offsets.clone())

            w = dense_emb.embedding.weight.detach().float()
            max_abs = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-18)
            exp = torch.ceil(torch.log2(max_abs / 127.0)).clamp(-60, 60).to(torch.int8)
            inv_scale = torch.ldexp(torch.ones_like(max_abs), -exp.to(torch.int32))
            mant = torch.round(w * inv_scale).clamp(-127, 127).to(torch.int8)
            obj.register_buffer("mantissas", mant)
            obj.register_buffer("exponents", exp)
            return obj

        @property
        def nbytes_bfp64(self) -> int:
            return int(self.mantissas.numel() + self.exponents.numel())

        @property
        def nbytes_fp32(self) -> int:
            return int(self.total_N * self.embedding_dim * 4)

        def gather_bfp64_blocks(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Zero-float-op O(1) gather returning `(m_x, exp_x)`:
            - `m_x`: `[B, L, num_heads, D]` (`torch.int8`)
            - `exp_x`: `[B, L, num_heads, 1]` (`torch.int8`)
            """
            shifted = input_ids + self.offsets
            return self.mantissas[shifted], self.exponents[shifted]

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            m_x, exp_x = self.gather_bfp64_blocks(input_ids)
            return torch.ldexp(m_x.float(), exp_x.to(torch.int32))

    class TorchMem8AdaptiveMultiHeadEmbedding(nn.Module):
        """
        MEM|8 Adaptive MultiHeadEmbedding in PyTorch: rows stay in `BFP-i8`
        (`65 B/row`) while their relative reconstruction error is `<= tau`, and
        promote to `BFP-i16` (`129 B/row`) when it exceeds `tau`.
        """

        def __init__(self, list_of_N: List[int], D: int = ENGRAM_HEAD_DIM, tau: float = 0.0052):
            super().__init__()
            self.num_heads = len(list_of_N)
            self.embedding_dim = D
            self.tau = float(tau)
            offsets = [0]
            for n in list_of_N[:-1]:
                offsets.append(offsets[-1] + n)
            self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

            total_N = sum(list_of_N)
            self.total_N = total_N
            self.register_buffer("mantissas_i8", torch.zeros(total_N, D, dtype=torch.int8))
            self.register_buffer("mantissas_i16", torch.zeros(total_N, D, dtype=torch.int16))
            self.register_buffer("exponents", torch.full((total_N, 1), -9, dtype=torch.int8))
            self.register_buffer("promoted", torch.zeros(total_N, dtype=torch.bool))

        @classmethod
        def from_float_embedding(
            cls, dense_emb: MultiHeadEmbedding, tau: float = 0.0052
        ) -> "TorchMem8AdaptiveMultiHeadEmbedding":
            total_N, D = dense_emb.embedding.weight.shape
            obj = cls.__new__(cls)
            nn.Module.__init__(obj)
            obj.num_heads = dense_emb.num_heads
            obj.embedding_dim = D
            obj.tau = float(tau)
            obj.total_N = total_N
            obj.register_buffer("offsets", dense_emb.offsets.clone())

            w = dense_emb.embedding.weight.detach().float()
            max_abs = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-18)

            exp8 = torch.ceil(torch.log2(max_abs / 127.0)).clamp(-60, 60).to(torch.int8)
            inv8 = torch.ldexp(torch.ones_like(max_abs), -exp8.to(torch.int32))
            m8 = torch.round(w * inv8).clamp(-127, 127).to(torch.int8)

            recon8 = torch.ldexp(m8.float(), exp8.to(torch.int32))
            err_sq = (w - recon8).pow(2).sum(dim=-1, keepdim=True)
            sig_sq = w.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-24)
            promote = ((err_sq / sig_sq).sqrt() > tau).squeeze(-1)

            exp16 = torch.ceil(torch.log2(max_abs / 32767.0)).clamp(-60, 60).to(torch.int8)
            inv16 = torch.ldexp(torch.ones_like(max_abs), -exp16.to(torch.int32))
            m16 = torch.round(w * inv16).clamp(-32767, 32767).to(torch.int16)

            obj.register_buffer("mantissas_i8", m8)
            obj.register_buffer("mantissas_i16", m16)
            obj.register_buffer("exponents", torch.where(promote.unsqueeze(-1), exp16, exp8))
            obj.register_buffer("promoted", promote)
            return obj

        @property
        def promoted_count(self) -> int:
            return int(self.promoted.sum().item())

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            shifted = input_ids + self.offsets
            promoted = self.promoted[shifted].unsqueeze(-1)
            m = torch.where(
                promoted,
                self.mantissas_i16[shifted].float(),
                self.mantissas_i8[shifted].float(),
            )
            return torch.ldexp(m, self.exponents[shifted].to(torch.int32))

    class TorchTernaryLinearPq(nn.Module):
        """
        Ternary `{-1, 0, +1}` linear projection with group-64 rational `p/q` (`q = 2^20`)
        scales and Deferred-Scale Integer-Aligned `BFP-64` (`i16 -> i64`) accumulation!
        """

        def __init__(self, in_features: int, out_features: int, group_size: int = ENGRAM_HEAD_DIM):
            super().__init__()
            assert in_features % group_size == 0
            self.in_features = in_features
            self.out_features = out_features
            self.group_size = group_size
            self.num_groups = in_features // group_size

            signs = torch.randint(-1, 2, (out_features, self.num_groups, group_size), dtype=torch.int8)
            scales_pq = torch.full((out_features, self.num_groups), int(0.02 * PQ_SCALE_Q20), dtype=torch.int64)
            self.register_buffer("w_signs", signs)
            self.register_buffer("scales_pq", scales_pq)

        def forward_prequant_bfp64(self, m_x: torch.Tensor, exp_x: torch.Tensor) -> torch.Tensor:
            """
            Execute integer-aligned BFP-64 projection directly on gathered `(m_x, exp_x)`:
            - `m_x`: `[B, L, num_groups, 64]` (`int8`)
            - `exp_x`: `[B, L, num_groups, 1]` (`int8`)
            """
            exp_g = exp_x.squeeze(-1).to(torch.int32)  # [B, L, G]
            min_exp = exp_g.amin(dim=-1, keepdim=True)  # [B, L, 1]
            shift = (exp_g - min_exp).clamp(0, 24)
            head_mult = (1 << shift).to(torch.int64)    # [B, L, G]

            # Block sum in i16 range (|S_64| <= 64 * 127 = 8,128 <= 32,767)
            block_sums = torch.einsum(
                "blgk,ogk->blgo",
                m_x.to(torch.int32),
                self.w_signs.to(torch.int32),
            ).to(torch.int64)

            # Pure i64 cross-head accumulation with rational p/q (q = 2^20) group scales
            aligned = block_sums * head_mult.unsqueeze(-1) * self.scales_pq.T.unsqueeze(0).unsqueeze(0)
            row_i64 = aligned.sum(dim=2)  # [B, L, out_features]
            return torch.ldexp(row_i64.float(), min_exp - 20)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, L, Din = x.shape
            xg = x.view(B, L, self.num_groups, self.group_size)
            max_abs = xg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-18)
            exp_x = torch.ceil(torch.log2(max_abs / 127.0)).clamp(-60, 60).to(torch.int8)
            inv_scale = torch.ldexp(torch.ones_like(max_abs), -exp_x.to(torch.int32))
            m_x = torch.round(xg * inv_scale).clamp(-127, 127).to(torch.int8)
            return self.forward_prequant_bfp64(m_x, exp_x)

    class Engram(nn.Module):
        def __init__(
            self,
            layer_id: int,
            config: Optional[EngramConfig] = None,
            backbone_cfg: Optional[BackBoneConfig] = None,
            embedding_factory: Optional[Union[str, Callable[[List[int], int, int], nn.Module]]] = None,
            hash_mapping: Optional[NgramHashMapping] = None,
        ):
            super().__init__()
            cfg = config if config is not None else engram_cfg
            b_cfg = backbone_cfg if backbone_cfg is not None else backbone_config

            self.layer_id = layer_id
            self.cfg = cfg
            self.b_cfg = b_cfg

            # Reuse cached NgramHashMapping across layers (PR #13)
            self.hash_mapping = (
                hash_mapping
                if hash_mapping is not None
                else get_or_create_hash_mapping(
                    engram_vocab_size=cfg.engram_vocab_size,
                    max_ngram_size=cfg.max_ngram_size,
                    n_embed_per_ngram=cfg.n_embed_per_ngram,
                    n_head_per_ngram=cfg.n_head_per_ngram,
                    layer_ids=cfg.layer_ids,
                    tokenizer_name_or_path=cfg.tokenizer_name_or_path,
                    pad_id=cfg.pad_id,
                    seed=cfg.seed,
                    use_pq_hash=cfg.use_pq_hash,
                )
            )

            list_of_N = [
                x for y in self.hash_mapping.vocab_size_across_layers[self.layer_id] for x in y
            ]
            head_dim = cfg.n_embed_per_ngram // cfg.n_head_per_ngram
            mode = embedding_factory if embedding_factory is not None else cfg.embedding_mode

            if callable(mode):
                self.multi_head_embedding = mode(list_of_N, head_dim, self.layer_id)
            elif mode == "dense":
                self.multi_head_embedding = MultiHeadEmbedding(list_of_N=list_of_N, D=head_dim)
            elif mode == "bfp64":
                dense_emb = MultiHeadEmbedding(list_of_N=list_of_N, D=head_dim)
                self.multi_head_embedding = TorchBfp64MultiHeadEmbedding.from_float_embedding(
                    dense_emb
                )
            elif mode == "mem8_adaptive":
                dense_emb = MultiHeadEmbedding(list_of_N=list_of_N, D=head_dim)
                self.multi_head_embedding = (
                    TorchMem8AdaptiveMultiHeadEmbedding.from_float_embedding(dense_emb)
                )
            else:
                raise ValueError(
                    f"Unknown embedding mode {mode!r}; expected 'dense', 'bfp64', "
                    "'mem8_adaptive', or a callable (list_of_N, D, layer_id) -> nn.Module"
                )

            self.short_conv = ShortConv(
                hidden_size=b_cfg.hidden_size,
                kernel_size=cfg.kernel_size,
                dilation=cfg.max_ngram_size,
                hc_mult=b_cfg.hc_mult,
            )

            engram_hidden_size = (cfg.max_ngram_size - 1) * cfg.n_embed_per_ngram
            if cfg.proj_mode == "ternary_pq":
                self.value_proj: nn.Module = TorchTernaryLinearPq(engram_hidden_size, b_cfg.hidden_size)
                self.key_projs = nn.ModuleList([
                    TorchTernaryLinearPq(engram_hidden_size, b_cfg.hidden_size)
                    for _ in range(b_cfg.hc_mult)
                ])
            else:
                self.value_proj = nn.Linear(engram_hidden_size, b_cfg.hidden_size)
                self.key_projs = nn.ModuleList([
                    nn.Linear(engram_hidden_size, b_cfg.hidden_size)
                    for _ in range(b_cfg.hc_mult)
                ])

            self.norm1 = nn.ModuleList([
                nn.RMSNorm(b_cfg.hidden_size) for _ in range(b_cfg.hc_mult)
            ])
            self.norm2 = nn.ModuleList([
                nn.RMSNorm(b_cfg.hidden_size) for _ in range(b_cfg.hc_mult)
            ])

        def forward(self, hidden_states: torch.Tensor, input_ids: Any) -> torch.Tensor:
            """
            hidden_states: [B, L, HC_MULT, D]
            input_ids:     [B, L]
            """
            # Device-aware placement (PR #15)
            hash_np = self.hash_mapping.hash(input_ids)[self.layer_id]
            hash_input_ids = torch.as_tensor(
                hash_np, dtype=torch.long, device=hidden_states.device
            )

            # Fast zero-quantization BFP-64 + Ternary p/q path when both are enabled
            if (
                isinstance(self.multi_head_embedding, TorchBfp64MultiHeadEmbedding)
                and isinstance(self.value_proj, TorchTernaryLinearPq)
            ):
                m_x, exp_x = self.multi_head_embedding.gather_bfp64_blocks(hash_input_ids)
                val_proj_out = self.value_proj.forward_prequant_bfp64(m_x, exp_x)
                normed_keys = torch.stack(
                    [
                        self.norm1[hc_idx](
                            self.key_projs[hc_idx].forward_prequant_bfp64(m_x, exp_x)  # type: ignore[operator]
                        )
                        for hc_idx in range(self.b_cfg.hc_mult)
                    ],
                    dim=2,
                )
            else:
                embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)
                val_proj_out = self.value_proj(embeddings)
                normed_keys = torch.stack(
                    [
                        self.norm1[hc_idx](self.key_projs[hc_idx](embeddings))
                        for hc_idx in range(self.b_cfg.hc_mult)
                    ],
                    dim=2,
                )

            # Vectorized hyper-connection gating across all `hc_mult` streams (PR #14)
            normed_queries = torch.stack(
                [
                    self.norm2[hc_idx](hidden_states[:, :, hc_idx, :])
                    for hc_idx in range(self.b_cfg.hc_mult)
                ],
                dim=2,
            )
            gate = torch.einsum("bshd,bshd->bsh", normed_keys, normed_queries) / math.sqrt(
                self.b_cfg.hidden_size
            )
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gates = gate.sigmoid().unsqueeze(-1)

            value = gates * val_proj_out.unsqueeze(2)
            output = value + self.short_conv(value)
            return output

    class TransformerBlock(nn.Module):
        def __init__(
            self,
            layer_id: int,
            config: Optional[EngramConfig] = None,
            backbone_cfg: Optional[BackBoneConfig] = None,
            embedding_factory: Optional[Union[str, Callable[[List[int], int, int], nn.Module]]] = None,
        ):
            super().__init__()
            cfg = config if config is not None else engram_cfg
            b_cfg = backbone_cfg if backbone_cfg is not None else backbone_config
            self.attn = lambda x: x
            self.moe = lambda x: x
            self.engram = None
            if layer_id in cfg.layer_ids:
                self.engram = Engram(
                    layer_id=layer_id,
                    config=cfg,
                    backbone_cfg=b_cfg,
                    embedding_factory=embedding_factory,
                )

        def forward(self, input_ids: Any, hidden_states: torch.Tensor) -> torch.Tensor:
            if self.engram is not None:
                hidden_states = (
                    self.engram(hidden_states=hidden_states, input_ids=input_ids) + hidden_states
                )
            hidden_states = self.attn(hidden_states) + hidden_states
            hidden_states = self.moe(hidden_states) + hidden_states
            return hidden_states

else:
    # Pure-stdlib sovereign bindings when PyTorch is not installed
    CompressedTokenizer = CompressedTokenizerPq  # type: ignore
    NgramHashMapping = NgramHashMappingPq        # type: ignore
    MultiHeadEmbedding = Bfp64MultiHeadEmbeddingPq  # type: ignore
    Engram = EngramPq                            # type: ignore


if __name__ == "__main__":
    text = "Only Alexander the Great could tame the horse Bucephalus."

    if HAS_TORCH:
        LLM = [
            nn.Embedding(backbone_config.vocab_size, backbone_config.hidden_size),
            *[
                TransformerBlock(layer_id=layer_id)
                for layer_id in range(backbone_config.num_layers)
            ],
            nn.Linear(backbone_config.hidden_size, backbone_config.vocab_size),
        ]

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                engram_cfg.tokenizer_name_or_path, trust_remote_code=True
            )
            input_ids = tokenizer(text, return_tensors="pt").input_ids
        except Exception:
            offline_tok = SovereignOfflineTokenizer(vocab_size=backbone_config.vocab_size)
            input_ids = torch.tensor([offline_tok.encode(text)], dtype=torch.long)

        for idx, layer in enumerate(LLM):
            if idx == 0:
                hidden_states = LLM[0](input_ids)
                hidden_states = hidden_states.unsqueeze(2).expand(
                    -1, -1, backbone_config.hc_mult, -1
                )
            elif idx == len(LLM) - 1:
                hidden_states = hidden_states[:, :, 0, :]
                output = layer(hidden_states)
            else:
                hidden_states = layer(input_ids=input_ids, hidden_states=hidden_states)

        print("✅ Forward Complete! (PyTorch Engine)")
        print(f"{input_ids.shape=}\n{output.shape=}")
    else:
        offline_tok = SovereignOfflineTokenizer(vocab_size=4096)
        input_ids_list = [offline_tok.encode(text)]
        B = len(input_ids_list)
        L = len(input_ids_list[0])
        hidden_size = 128
        hc_mult = 4

        engram_layer = EngramPq(
            layer_id=1,
            engram_vocab_size=(512, 512),
            max_ngram_size=3,
            n_embed_per_ngram=512,
            n_head_per_ngram=8,
            layer_ids=(1, 15),
            tokenizer_name_or_path="offline",
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            embedding_mode="bfp64",
        )
        hidden_states_pq = [
            [[[0.05 * ((t + hc + d) % 7 - 3) for d in range(hidden_size)] for hc in range(hc_mult)] for t in range(L)]
            for _ in range(B)
        ]
        out_pq = engram_layer.forward(hidden_states_pq, input_ids_list)
        emb_table = engram_layer.multi_head_embedding
        print("✅ Forward Complete! (8b-is Sovereign Pure-Stdlib p/q + BFP-64 Engine)")
        print(f"input_ids.shape=({B}, {L})")
        print(f"output.shape=({len(out_pq)}, {len(out_pq[0])}, {len(out_pq[0][0])}, {len(out_pq[0][0][0])})")
        print(
            f"BFP-64 Table Compression: {emb_table.nbytes_fp32:,} B (FP32) -> "
            f"{emb_table.nbytes_bfp64:,} B (BFP-64) = {emb_table.compression_ratio_vs_fp32:.2f}x smaller"
        )
