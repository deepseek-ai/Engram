"""
================================================================================
[Engram Architecture - Hybrid Optimized Implementation v3]

Optimizations:
1. Hybrid CPU/GPU path selection based on input size
2. Small batches (< threshold): NumPy path (lower overhead)
3. Large batches (>= threshold): GPU path (better throughput)
4. No logging in forward pass
5. Pre-computed constants and buffers

Compatible with benchmark.py and test_correctness.py
================================================================================
"""

from typing import List, Tuple
from dataclasses import dataclass, field
import math

from sympy import isprime
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tokenizers import normalizers, Regex


@dataclass
class EngramConfig:
    tokenizer_name_or_path: str = "deepseek-ai/DeepSeek-V3"
    engram_vocab_size: List[int] = field(default_factory=lambda: [129280*5, 129280*5])
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    layer_ids: List[int] = field(default_factory=lambda: [1, 15])
    pad_id: int = 2
    seed: int = 0
    kernel_size: int = 4
    # Threshold for GPU vs CPU path (batch_size * seq_len)
    gpu_threshold: int = 1024
    

@dataclass
class BackBoneConfig:
    hidden_size: int = 1024
    hc_mult: int = 4
    vocab_size: int = 129280
    num_layers: int = 30
    

engram_cfg = EngramConfig()
backbone_config = BackBoneConfig()


class CompressedTokenizer:
    """
    Tokenizer compression via NFKC normalization.
    Reduces vocabulary ~23% by merging semantically equivalent tokens.
    """
    
    def __init__(self, tokenizer_name_or_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, 
            trust_remote_code=True
        )
        
        SENTINEL = "\uE000"
        self.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), SENTINEL),
            normalizers.Strip(),
            normalizers.Replace(SENTINEL, " "),
        ])
        
        self.lookup_table, self.num_new_token = self._build_lookup_table()
        self._lookup_tensor = None  # Lazy init for GPU
        self._lookup_tensor_device = None
    
    def __len__(self) -> int:
        return self.num_new_token
    
    def _build_lookup_table(self) -> Tuple[np.ndarray, int]:
        old2new = {}
        key2new = {}          
        new_tokens = []

        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)
            
            if "�" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.normalizer.normalize_str(text)
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
    
    def compress_cpu(self, input_ids: np.ndarray) -> np.ndarray:
        """Fast CPU path using NumPy indexing"""
        clamped = np.clip(input_ids, 0, len(self.lookup_table) - 1)
        return self.lookup_table[clamped]
    
    def compress_gpu(self, input_ids: torch.Tensor) -> torch.Tensor:
        """GPU path with lazy tensor initialization"""
        device = input_ids.device
        
        # Lazy init / device transfer
        if self._lookup_tensor is None or self._lookup_tensor_device != device:
            self._lookup_tensor = torch.from_numpy(self.lookup_table).to(device)
            self._lookup_tensor_device = device
        
        clamped = input_ids.clamp(0, len(self.lookup_table) - 1)
        return self._lookup_tensor[clamped]
    
    def __call__(self, input_ids):
        """CPU fallback for compatibility"""
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.cpu().numpy()
        arr = np.asarray(input_ids, dtype=np.int64)
        return self.compress_cpu(arr)


class ShortConv(nn.Module):
    """Depthwise causal convolution with RMSNorm and SiLU activation"""
    
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

        normed_chunks = [self.norms[i](x[:, :, i, :]) for i in range(G)]
        x_norm = torch.cat(normed_chunks, dim=-1)
        
        x_bct = x_norm.transpose(1, 2)
        y_bct = self.conv(x_bct)[..., :T]

        if self.activation:
            y_bct = self.act_fn(y_bct)
            
        return y_bct.transpose(1, 2).view(B, T, G, C).contiguous()


def find_next_prime(start: int, seen_primes: set) -> int:
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


class HybridNgramHashMapping(nn.Module):
    """
    Hybrid CPU/GPU N-gram hashing module.
    
    Automatically selects optimal path based on input size:
    - Small inputs: CPU/NumPy (lower overhead)
    - Large inputs: GPU (better throughput)
    """
    
    def __init__(
        self, 
        engram_vocab_size: List[int],
        max_ngram_size: int,
        n_embed_per_ngram: int,
        n_head_per_ngram: int,
        layer_ids: List[int],
        tokenizer_name_or_path: str,
        pad_id: int,
        seed: int,
        gpu_threshold: int = 1024,
    ):
        super().__init__()
        
        self.vocab_size_per_ngram = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.pad_id = pad_id
        self.layer_ids = layer_ids
        self.gpu_threshold = gpu_threshold

        # Initialize compressed tokenizer
        self.compressed_tokenizer = CompressedTokenizer(tokenizer_name_or_path)
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        
        if self.pad_id is not None:
            self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])
        
        # Compute layer-specific multipliers
        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007
        
        self.layer_multipliers = {}
        for layer_id in self.layer_ids:
            base_seed = int(seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(low=0, high=half_bound, size=(self.max_ngram_size,), dtype=np.int64)
            self.layer_multipliers[layer_id] = r * 2 + 1

        # Calculate prime modulo sizes
        self.vocab_size_across_layers = self._calculate_vocab_size_across_layers()
        
        # Register GPU buffers
        for layer_id in self.layer_ids:
            mult_tensor = torch.tensor(self.layer_multipliers[layer_id], dtype=torch.long)
            self.register_buffer(f"multipliers_layer_{layer_id}", mult_tensor)
            
            all_primes = [p for ngram_heads in self.vocab_size_across_layers[layer_id] for p in ngram_heads]
            self.register_buffer(f"prime_mods_layer_{layer_id}", torch.tensor(all_primes, dtype=torch.long))
            
            # Also store as numpy for CPU path
            setattr(self, f"_np_primes_layer_{layer_id}", np.array(all_primes, dtype=np.int64))

    def _calculate_vocab_size_across_layers(self) -> dict:
        seen_primes = set()
        vocab_size_across_layers = {}
        
        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []
                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                current_prime_search_start = vocab_size - 1
                
                for _ in range(self.n_head_per_ngram):
                    found_prime = find_next_prime(current_prime_search_start, seen_primes)
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime
                
                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes
            
        return vocab_size_across_layers

    def _hash_cpu(self, input_ids: np.ndarray, layer_id: int) -> np.ndarray:
        """CPU path: NumPy-based hashing (optimal for small inputs)"""
        B, T = input_ids.shape
        
        multipliers = self.layer_multipliers[layer_id]
        prime_mods = getattr(self, f"_np_primes_layer_{layer_id}")
        
        # Build shifted versions
        shifts = [input_ids]
        for k in range(1, self.max_ngram_size):
            padding = np.full((B, k), self.pad_id, dtype=np.int64)
            shifts.append(np.concatenate([padding, input_ids[:, :-k]], axis=1))
        
        # Compute hashes
        all_hashes = []
        hash_idx = 0
        
        for n in range(2, self.max_ngram_size + 1):
            mix = shifts[0] * multipliers[0]
            for k in range(1, n):
                mix = np.bitwise_xor(mix, shifts[k] * multipliers[k])
            
            for _ in range(self.n_head_per_ngram):
                all_hashes.append((mix % prime_mods[hash_idx]).astype(np.int64))
                hash_idx += 1
        
        return np.stack(all_hashes, axis=2)

    def _hash_gpu(self, input_ids: torch.Tensor, layer_id: int) -> torch.Tensor:
        """GPU path: PyTorch-based hashing (optimal for large inputs)"""
        B, T = input_ids.shape
        device = input_ids.device
        
        multipliers = getattr(self, f"multipliers_layer_{layer_id}")
        prime_mods = getattr(self, f"prime_mods_layer_{layer_id}")
        
        # Build shifted versions
        shifts = [input_ids]
        for k in range(1, self.max_ngram_size):
            padding = torch.full((B, k), self.pad_id, dtype=torch.long, device=device)
            shifts.append(torch.cat([padding, input_ids[:, :-k]], dim=1))
        
        # Compute hashes
        all_hashes = []
        hash_idx = 0
        
        for n in range(2, self.max_ngram_size + 1):
            mix = shifts[0] * multipliers[0]
            for k in range(1, n):
                mix = torch.bitwise_xor(mix, shifts[k] * multipliers[k])
            
            for _ in range(self.n_head_per_ngram):
                all_hashes.append(mix % prime_mods[hash_idx])
                hash_idx += 1
        
        return torch.stack(all_hashes, dim=2)

    def hash(self, input_ids: torch.Tensor) -> dict:
        """
        Main hashing interface with automatic path selection.
        
        Args:
            input_ids: Raw token IDs [B, T]
            
        Returns:
            Dictionary mapping layer_id -> hash indices [B, T, num_heads]
        """
        B, T = input_ids.shape
        input_size = B * T
        device = input_ids.device
        
        if input_size < self.gpu_threshold:
            # CPU path for small inputs
            input_np = input_ids.cpu().numpy()
            compressed_np = self.compressed_tokenizer.compress_cpu(input_np)
            
            result = {}
            for layer_id in self.layer_ids:
                hash_np = self._hash_cpu(compressed_np, layer_id)
                result[layer_id] = torch.from_numpy(hash_np).to(device)
            return result
        else:
            # GPU path for large inputs
            compressed = self.compressed_tokenizer.compress_gpu(input_ids)
            return {
                layer_id: self._hash_gpu(compressed, layer_id)
                for layer_id in self.layer_ids
            }


class MultiHeadEmbedding(nn.Module):
    """Multi-head embedding lookup with concatenated tables"""
    
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
        return self.embedding(input_ids + self.offsets)


class Engram(nn.Module):
    """
    Engram conditional memory module with hybrid optimization.
    
    Retrieves static n-gram embeddings and fuses them with dynamic hidden states
    via context-aware gating.
    """
    
    def __init__(self, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        
        # Hybrid hash mapping
        self.hash_mapping = HybridNgramHashMapping(
            engram_vocab_size=engram_cfg.engram_vocab_size,
            max_ngram_size=engram_cfg.max_ngram_size,
            n_embed_per_ngram=engram_cfg.n_embed_per_ngram,
            n_head_per_ngram=engram_cfg.n_head_per_ngram,
            layer_ids=engram_cfg.layer_ids,
            tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
            pad_id=engram_cfg.pad_id,
            seed=engram_cfg.seed,
            gpu_threshold=engram_cfg.gpu_threshold,
        )
        
        # Multi-head embedding
        vocab_sizes = [x for y in self.hash_mapping.vocab_size_across_layers[self.layer_id] for x in y]
        embed_dim = engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram
        self.multi_head_embedding = MultiHeadEmbedding(list_of_N=vocab_sizes, D=embed_dim)
        
        # Short convolution
        self.short_conv = ShortConv(
            hidden_size=backbone_config.hidden_size,
            kernel_size=engram_cfg.kernel_size,
            dilation=engram_cfg.max_ngram_size,
            hc_mult=backbone_config.hc_mult,
        )
        
        # Projections
        engram_hidden_size = (engram_cfg.max_ngram_size - 1) * engram_cfg.n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size, backbone_config.hidden_size)
        self.key_projs = nn.ModuleList([
            nn.Linear(engram_hidden_size, backbone_config.hidden_size) 
            for _ in range(backbone_config.hc_mult)
        ])
        
        # Normalizations
        self.norm1 = nn.ModuleList([
            nn.RMSNorm(backbone_config.hidden_size) 
            for _ in range(backbone_config.hc_mult)
        ])
        self.norm2 = nn.ModuleList([
            nn.RMSNorm(backbone_config.hidden_size) 
            for _ in range(backbone_config.hc_mult)
        ])
        
        # Pre-computed constant
        self._inv_sqrt_d = 1.0 / math.sqrt(backbone_config.hidden_size)
    
    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, HC_MULT, D]
            input_ids: [B, T]
            
        Returns:
            output: [B, T, HC_MULT, D]
        """
        # Hash and embed
        hash_result = self.hash_mapping.hash(input_ids)
        hash_indices = hash_result[self.layer_id]
        embeddings = self.multi_head_embedding(hash_indices).flatten(start_dim=-2)
        
        # Context-aware gating
        gates = []
        for hc_idx in range(backbone_config.hc_mult):
            key = self.key_projs[hc_idx](embeddings)
            normed_key = self.norm1[hc_idx](key)
            query = hidden_states[:, :, hc_idx, :]
            normed_query = self.norm2[hc_idx](query)
            
            gate = (normed_key * normed_query).sum(dim=-1) * self._inv_sqrt_d
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gates.append(gate.sigmoid().unsqueeze(-1))
        
        gates = torch.stack(gates, dim=2)
        
        # Gated value + convolution
        value = gates * self.value_proj(embeddings).unsqueeze(2)
        return value + self.short_conv(value)


class TransformerBlock(nn.Module):
    """Transformer block with optional Engram module"""
    
    def __init__(self, layer_id: int):
        super().__init__()
        self.attn = lambda x: x
        self.moe = lambda x: x
        self.engram = Engram(layer_id=layer_id) if layer_id in engram_cfg.layer_ids else None
    
    def forward(self, input_ids: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.engram is not None:
            hidden_states = self.engram(hidden_states=hidden_states, input_ids=input_ids) + hidden_states
        hidden_states = self.attn(hidden_states) + hidden_states
        hidden_states = self.moe(hidden_states) + hidden_states
        return hidden_states


if __name__ == '__main__':
    import time
    
    print("=" * 60)
    print("Engram Hybrid Optimized v3 - Demo")
    print("=" * 60)
    print(f"GPU threshold: {engram_cfg.gpu_threshold} elements")
    
    # Build model
    LLM = [
        nn.Embedding(backbone_config.vocab_size, backbone_config.hidden_size),
        *[TransformerBlock(layer_id=layer_id) for layer_id in range(backbone_config.num_layers)],
        nn.Linear(backbone_config.hidden_size, backbone_config.vocab_size)
    ]
    
    # Tokenize input
    text = "Only Alexander the Great could tame the horse Bucephalus."
    tokenizer = AutoTokenizer.from_pretrained(engram_cfg.tokenizer_name_or_path, trust_remote_code=True)
    input_ids = tokenizer(text, return_tensors='pt').input_ids
    
    print(f"\nInput: {text}")
    print(f"Tokenized shape: {input_ids.shape}")
    print(f"Input size: {input_ids.shape[0] * input_ids.shape[1]} -> {'CPU' if input_ids.shape[0] * input_ids.shape[1] < engram_cfg.gpu_threshold else 'GPU'} path")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    input_ids = input_ids.to(device)
    for layer in LLM:
        layer.to(device)
    
    # Forward pass
    print("\nRunning forward pass...")
    start = time.time()
    
    for idx, layer in enumerate(LLM):
        if idx == 0:
            hidden_states = LLM[0](input_ids)
            hidden_states = hidden_states.unsqueeze(2).expand(-1, -1, backbone_config.hc_mult, -1)
        elif idx == len(LLM) - 1:
            hidden_states = hidden_states[:, :, 0, :]
            output = layer(hidden_states)
        else:
            hidden_states = layer(input_ids=input_ids, hidden_states=hidden_states)
    
    elapsed = (time.time() - start) * 1000
    
    print(f"\n Forward pass complete in {elapsed:.2f}ms")
    print(f"Output shape: {output.shape}")
    
    # Test both paths explicitly
    print("\n" + "=" * 60)
    print("Testing both paths explicitly:")
    print("=" * 60)
    
    engram = LLM[2].engram  # Layer 1 has Engram
    
    # Small input (CPU path)
    small_ids = torch.randint(0, 1000, (2, 64), device=device)
    small_hidden = torch.randn(2, 64, backbone_config.hc_mult, backbone_config.hidden_size, device=device)
    
    start = time.time()
    for _ in range(10):
        _ = engram(small_hidden, small_ids)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    small_time = (time.time() - start) * 100
    print(f"Small input (2x64={2*64}): {small_time:.2f}ms avg -> CPU path")
    
    # Large input (GPU path)
    large_ids = torch.randint(0, 1000, (8, 512), device=device)
    large_hidden = torch.randn(8, 512, backbone_config.hc_mult, backbone_config.hidden_size, device=device)
    
    start = time.time()
    for _ in range(10):
        _ = engram(large_hidden, large_ids)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    large_time = (time.time() - start) * 100
    print(f"Large input (8x512={8*512}): {large_time:.2f}ms avg -> GPU path")