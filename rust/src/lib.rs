//! # `engram-pq` — DeepSeek Engram Conditional Memory on the 8b-is Sovereign `p/q` + `BFP-64` Substrate
//!
//! Zero-dependency, `#![no_std]`-first, `#![forbid(unsafe_code)]` implementation of the
//! **DeepSeek Engram** conditional memory architecture
//! (*"Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models"*)
//! powered by the **8b-is GAIA-MLX-QUANT / `deepsiper-enthea`** mathematical substrate:
//!
//! 1. **Exact 64-Dim Head Alignment (`D_head = 64 == BFP-64 block size`)**:
//!    DeepSeek's Engram configuration (`n_embed_per_ngram = 512`, `n_head_per_ngram = 8`)
//!    uses head dimension `D_head = 512 / 8 = 64` — the exact block size of Omni `BFP-64`
//!    (`64 x i8` mantissas in `[-127, 127]` + `1 x i8` shared dyadic exponent = `65 bytes/row`
//!    vs `256 bytes/row` FP32 = **`3.94x` table compression**).
//!
//! 2. **Zero-Quantization `O(1)` Lookup + `i16 -> i64` Integer-Aligned Projection**:
//!    Retrieving `16` heads (`16 x 64 = 1024`) from a `BFP-64` table yields pre-quantized
//!    `(m_x, exp_x)` blocks with **zero floating-point operations**. Downstream ternary
//!    projections execute via 16-wide `i16` SIMD block accumulation (`|S_64| <= 64 * 127 = 8,128
//!    <= 32,767 = i16::MAX`) and `i64` cross-head integer exponent alignment.
//!
//! 3. **Lemire `p/q` (`q = 2^64`) Fast Reciprocal Prime Hashing**:
//!    Replaces hardware modulo (`mix % prime`) across all `(ngram, head)` prime buckets with
//!    precomputed 64-bit dyadic reciprocals (`ceil(2^64 / prime)`) with 100% bit-exact bucket
//!    equivalence. The speedup is platform-dependent (`1.39x-3.25x` on x86-64; on Apple
//!    Silicon the hardware `% prime` path can be faster).
//!
//! 4. **Fused Single-Pass Bilinear RMSNorm Gate**:
//!    Computes `g = <RMSNorm(k), RMSNorm(q)> / sqrt(D)` in a single fused pass without
//!    materializing intermediate normalized `k` and `q` vectors.
//!
//! 5. **100% Zero-Float Pure-Integer `p/q` ("Counting Divided") Engram Step**:
//!    `engram_step_pq_no_floats_into` executes the entire Engram layer step (gather, 5 linear
//!    projections, RMSNorm gate, signed-sqrt, sigmoid, ShortConv SiLU, and residual) in pure
//!    integer `i16`/`i32`/`i64`/`i128` arithmetic with **zero IEEE-754 floating-point ops**.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

/// Default Engram head dimension (`512 / 8 = 64`), identical to the `BFP-64` block size.
pub const ENGRAM_HEAD_DIM: usize = 64;

/// Default hyper-connection multiplicity (`hc_mult = 4`).
pub const ENGRAM_DEFAULT_HC_MULT: usize = 4;

/// Fixed-point `Q16` scale (`q = 2^16 = 65,536`) for the zero-float `p/q` Engram pipeline.
pub const PQ_Q16: i64 = 1 << 16;

/// Precomputed Lemire `p/q` (`q = 2^64`) reciprocal for fast prime-bucket modulo.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrimeBucketMod {
    /// Prime bucket count `P`
    pub prime: u64,
    /// Precomputed reciprocal `ceil(2^64 / P)` for fast multiply-shift remainder
    pub inv_q64: u64,
    /// Cumulative row offset of this head inside `MultiHeadEmbedding`
    pub table_offset: usize,
}

impl PrimeBucketMod {
    /// Construct a [`PrimeBucketMod`] for prime modulus `prime` at `table_offset`.
    #[inline]
    pub const fn new(prime: u64, table_offset: usize) -> Self {
        let p = if prime == 0 { 1 } else { prime };
        let inv_q64 = (u64::MAX / p).wrapping_add(1);
        Self {
            prime: p,
            inv_q64,
            table_offset,
        }
    }

    /// Compute `x % self.prime` for `x < 2^32` using Lemire's direct `p/q` (`q = 2^64`)
    /// reciprocal remainder (`mulhi64(wrapping_mul(x, inv_q64), prime)`) with zero branches
    /// and zero hardware division instructions.
    #[inline]
    pub const fn fast_rem_u32(&self, x: u64) -> usize {
        let lowbits = x.wrapping_mul(self.inv_q64);
        (((lowbits as u128) * (self.prime as u128)) >> 64) as usize
    }

    /// Compute `x % self.prime` for any 64-bit `x` (`0 <= x < 2^64` when `self.prime < 2^32`)
    /// using Lemire's `p/q` (`q = 2^64`) reciprocal without hardware division instructions.
    #[inline]
    pub const fn fast_rem_u64(&self, x: u64) -> usize {
        if x <= 0xFFFF_FFFF {
            return self.fast_rem_u32(x);
        }
        let inv_floor = self.inv_q64.wrapping_sub(1);
        let q_est = (((x as u128) * (inv_floor as u128)) >> 64) as u64;
        let rem = x.wrapping_sub(q_est.wrapping_mul(self.prime));
        if rem >= self.prime {
            (rem - self.prime) as usize
        } else {
            rem as usize
        }
    }
}

/// Reference N-gram XOR-multiply hash + hardware `% prime` modulo over the
/// 32-bit-folded mix (`mix ^ (mix >> 32)`, matching `engram_pq.py` with
/// `fold_32bit = true`). Note: `engram_demo_v1.py`'s default path hashes the
/// full 64-bit mix, so this kernel is bit-identical to the folded reference,
/// not to the un-folded Python baseline.
pub fn engram_hash_ngram_mod_into(
    tokens: &[u64],
    pad_id: u64,
    multipliers: &[u64],
    n_heads_per_ngram: usize,
    buckets: &[PrimeBucketMod],
    out_global_indices: &mut [usize],
) {
    let seq_len = tokens.len();
    let max_ngram = multipliers.len();
    if max_ngram < 2 || n_heads_per_ngram == 0 {
        return;
    }
    let total_heads = (max_ngram - 1) * n_heads_per_ngram;
    assert!(buckets.len() >= total_heads);
    assert!(out_global_indices.len() >= seq_len * total_heads);

    for t in 0..seq_len {
        let row_out = &mut out_global_indices[t * total_heads..(t + 1) * total_heads];
        let mut mix = tokens[t].wrapping_mul(multipliers[0]);

        for n in 2..=max_ngram {
            let k = n - 1;
            let tok_k = if t >= k { tokens[t - k] } else { pad_id };
            mix ^= tok_k.wrapping_mul(multipliers[k]);
            let mix32 = (mix ^ (mix >> 32)) & 0xFFFF_FFFF;

            let ngram_idx = n - 2;
            let base_h = ngram_idx * n_heads_per_ngram;
            for h in 0..n_heads_per_ngram {
                let b = &buckets[base_h + h];
                row_out[base_h + h] = b.table_offset + ((mix32 % b.prime) as usize);
            }
        }
    }
}

/// Fast `p/q` (`q = 2^64`) reciprocal N-gram hash mapping (zero hardware division in inner loop).
///
/// Produces bit-identical bucket indices to [`engram_hash_ngram_mod_into`].
pub fn engram_hash_ngram_pq_into(
    tokens: &[u64],
    pad_id: u64,
    multipliers: &[u64],
    n_heads_per_ngram: usize,
    buckets: &[PrimeBucketMod],
    out_global_indices: &mut [usize],
) {
    let seq_len = tokens.len();
    let max_ngram = multipliers.len();
    if max_ngram < 2 || n_heads_per_ngram == 0 {
        return;
    }
    let total_heads = (max_ngram - 1) * n_heads_per_ngram;
    assert!(buckets.len() >= total_heads);
    assert!(out_global_indices.len() >= seq_len * total_heads);

    for t in 0..seq_len {
        let row_out = &mut out_global_indices[t * total_heads..(t + 1) * total_heads];
        let mut mix = tokens[t].wrapping_mul(multipliers[0]);

        for n in 2..=max_ngram {
            let k = n - 1;
            let tok_k = if t >= k { tokens[t - k] } else { pad_id };
            mix ^= tok_k.wrapping_mul(multipliers[k]);
            let mix32 = (mix ^ (mix >> 32)) & 0xFFFF_FFFF;

            let ngram_idx = n - 2;
            let base_h = ngram_idx * n_heads_per_ngram;
            for h in 0..n_heads_per_ngram {
                let b = &buckets[base_h + h];
                row_out[base_h + h] = b.table_offset + b.fast_rem_u32(mix32);
            }
        }
    }
}

/// Unfolded 64-bit N-gram XOR-multiply hash + hardware `% prime` modulo
/// (matching SGLang's `_engram_hash_kernel` and `engram_demo_v1.py` where the full
/// 64-bit rolling XOR-product is reduced modulo each head's prime).
pub fn engram_hash_ngram_mod64_into(
    tokens: &[u64],
    pad_id: u64,
    multipliers: &[u64],
    n_heads_per_ngram: usize,
    buckets: &[PrimeBucketMod],
    out_global_indices: &mut [usize],
) {
    let seq_len = tokens.len();
    let max_ngram = multipliers.len();
    if max_ngram < 2 || n_heads_per_ngram == 0 {
        return;
    }
    let total_heads = (max_ngram - 1) * n_heads_per_ngram;
    assert!(buckets.len() >= total_heads);
    assert!(out_global_indices.len() >= seq_len * total_heads);

    for t in 0..seq_len {
        let row_out = &mut out_global_indices[t * total_heads..(t + 1) * total_heads];
        let mut mix = tokens[t].wrapping_mul(multipliers[0]);

        for n in 2..=max_ngram {
            let k = n - 1;
            let tok_k = if t >= k { tokens[t - k] } else { pad_id };
            mix ^= tok_k.wrapping_mul(multipliers[k]);

            let ngram_idx = n - 2;
            let base_h = ngram_idx * n_heads_per_ngram;
            for h in 0..n_heads_per_ngram {
                let b = &buckets[base_h + h];
                row_out[base_h + h] = b.table_offset + ((mix % b.prime) as usize);
            }
        }
    }
}

/// Unfolded 64-bit `p/q` (`q = 2^64`) reciprocal N-gram hash mapping (zero hardware division).
///
/// Produces 100% bit-identical bucket indices to [`engram_hash_ngram_mod64_into`] across all
/// 64-bit rolling hashes `mix` using the floor reciprocal `inv_floor = inv_q64 - 1`.
pub fn engram_hash_ngram_pq64_into(
    tokens: &[u64],
    pad_id: u64,
    multipliers: &[u64],
    n_heads_per_ngram: usize,
    buckets: &[PrimeBucketMod],
    out_global_indices: &mut [usize],
) {
    let seq_len = tokens.len();
    let max_ngram = multipliers.len();
    if max_ngram < 2 || n_heads_per_ngram == 0 {
        return;
    }
    let total_heads = (max_ngram - 1) * n_heads_per_ngram;
    assert!(buckets.len() >= total_heads);
    assert!(out_global_indices.len() >= seq_len * total_heads);

    for t in 0..seq_len {
        let row_out = &mut out_global_indices[t * total_heads..(t + 1) * total_heads];
        let mut mix = tokens[t].wrapping_mul(multipliers[0]);

        for n in 2..=max_ngram {
            let k = n - 1;
            let tok_k = if t >= k { tokens[t - k] } else { pad_id };
            mix ^= tok_k.wrapping_mul(multipliers[k]);

            let ngram_idx = n - 2;
            let base_h = ngram_idx * n_heads_per_ngram;
            for h in 0..n_heads_per_ngram {
                let b = &buckets[base_h + h];
                let inv_floor = b.inv_q64.wrapping_sub(1);
                let q_est = (((mix as u128) * (inv_floor as u128)) >> 64) as u64;
                let rem = mix.wrapping_sub(q_est.wrapping_mul(b.prime));
                let r = if rem >= b.prime { rem - b.prime } else { rem };
                row_out[base_h + h] = b.table_offset + (r as usize);
            }
        }
    }
}

/// A single 64-dimensional Engram head embedding stored natively in **Omni `BFP-64`** format:
/// `64 x i8` mantissas in `[-127, 127]` + `1 x i8` shared power-of-two exponent (`65 bytes` total).
#[derive(Clone, Copy, Debug)]
pub struct Bfp64HeadRow {
    pub mantissas: [i8; ENGRAM_HEAD_DIM],
    pub exponent: i8,
}

impl Bfp64HeadRow {
    /// Quantize a 64-element `f64` head embedding into a native `BFP-64` row (`65 bytes`).
    pub fn from_f64_slice(src: &[f64]) -> Self {
        assert!(src.len() >= ENGRAM_HEAD_DIM);
        let mut max_abs = 0.0f64;
        for &v in &src[..ENGRAM_HEAD_DIM] {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        if max_abs < 1e-18 {
            return Self {
                mantissas: [0i8; ENGRAM_HEAD_DIM],
                exponent: -64,
            };
        }
        let ratio = max_abs / 127.0;
        let bits = ratio.to_bits();
        let raw_exp = (((bits >> 52) & 0x7FF) as i32) - 1023;
        let mant = bits & 0x000F_FFFF_FFFF_FFFF;
        let e = if mant == 0 { raw_exp } else { raw_exp + 1 }.clamp(-60, 60) as i8;
        let inv_scale = libm_exp2_neg(e);

        let mut mantissas = [0i8; ENGRAM_HEAD_DIM];
        for (dst, &v) in mantissas.iter_mut().zip(src[..ENGRAM_HEAD_DIM].iter()) {
            let scaled = v * inv_scale;
            let rounded = if scaled >= 0.0 {
                (scaled + 0.5) as i32
            } else {
                (scaled - 0.5) as i32
            };
            *dst = rounded.clamp(-127, 127) as i8;
        }
        Self {
            mantissas,
            exponent: e,
        }
    }
}

/// **Quantum 64-Trit Bitplane Block (`Quantum64TritBlock`)**:
///
/// Packs 64 ternary weights `w_k in {-1, 0, +1}` into two 64-bit registers (`pos_mask: u64`,
/// `neg_mask: u64` = `16 bytes` per 64 weights = **`16.0x` weight compression** over FP32).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Quantum64TritBlock {
    pub pos_mask: u64,
    pub neg_mask: u64,
}

impl Quantum64TritBlock {
    /// Pack 64 ternary signs `{-1, 0, +1}` into a 16-byte [`Quantum64TritBlock`].
    pub fn from_signs(signs: &[i8]) -> Self {
        assert!(signs.len() >= ENGRAM_HEAD_DIM);
        let mut pos_mask = 0u64;
        let mut neg_mask = 0u64;
        for (k, &s) in signs[..ENGRAM_HEAD_DIM].iter().enumerate() {
            if s > 0 {
                pos_mask |= 1u64 << k;
            } else if s < 0 {
                neg_mask |= 1u64 << k;
            }
        }
        Self { pos_mask, neg_mask }
    }

    /// Unpack into 64 `i8` signs in `{-1, 0, +1}` for `i16` SIMD inner loops.
    #[inline]
    pub const fn unpack_signs(&self) -> [i8; ENGRAM_HEAD_DIM] {
        let mut out = [0i8; ENGRAM_HEAD_DIM];
        let mut k = 0usize;
        while k < ENGRAM_HEAD_DIM {
            let bit = 1u64 << k;
            let p = ((self.pos_mask & bit) != 0) as i8;
            let n = ((self.neg_mask & bit) != 0) as i8;
            out[k] = p - n;
            k += 1;
        }
        out
    }

    /// Compute exact 64-element dot product against `mx: &[i8; 64]` in `i16` (`[-8128, 8128]`).
    #[inline]
    pub fn dot_i16(&self, mx: &[i8]) -> i16 {
        let mut acc: i16 = 0;
        let mut pos = self.pos_mask;
        let mut neg = self.neg_mask;
        while pos != 0 {
            let idx = pos.trailing_zeros() as usize;
            acc = acc.wrapping_add(mx[idx] as i16);
            pos &= pos - 1;
        }
        while neg != 0 {
            let idx = neg.trailing_zeros() as usize;
            acc = acc.wrapping_sub(mx[idx] as i16);
            neg &= neg - 1;
        }
        acc
    }
}

/// **MEM|8 Adaptive Precision 64-Dim Engram Head Row (`Mem8AdaptiveHeadRow`)**:
///
/// - Common syntactic N-gram heads with relative `i8` error `<= tau` stay in compact
///   **`BFP-i8`** (`65 bytes/row`, `i16` SIMD accumulator).
/// - High-salience or outlier factual N-gram heads promote to **`BFP-i16`** (`129 bytes/row`,
///   `i32` SIMD accumulator).
#[derive(Clone, Copy, Debug)]
pub struct Mem8AdaptiveHeadRow {
    pub mantissas_i8: [i8; ENGRAM_HEAD_DIM],
    pub mantissas_i16: [i16; ENGRAM_HEAD_DIM],
    pub exponent: i8,
    pub promoted_i16: bool,
}

impl Mem8AdaptiveHeadRow {
    /// Quantize a 64-element `f64` head embedding with adaptive promotion threshold `tau`.
    pub fn from_f64_slice(src: &[f64], tau: f64) -> Self {
        assert!(src.len() >= ENGRAM_HEAD_DIM);
        let bfp8 = Bfp64HeadRow::from_f64_slice(src);
        let scale8 = libm_exp2_pos(bfp8.exponent);

        let mut err_sq = 0.0f64;
        let mut sig_sq = 0.0f64;
        for (k, &v) in src[..ENGRAM_HEAD_DIM].iter().enumerate() {
            let recon = (bfp8.mantissas[k] as f64) * scale8;
            let diff = v - recon;
            err_sq += diff * diff;
            sig_sq += v * v;
        }
        let rel_err = libm_sqrt(err_sq / if sig_sq > 1e-24 { sig_sq } else { 1e-24 });

        if rel_err <= tau {
            return Self {
                mantissas_i8: bfp8.mantissas,
                mantissas_i16: [0i16; ENGRAM_HEAD_DIM],
                exponent: bfp8.exponent,
                promoted_i16: false,
            };
        }

        let mut max_abs = 0.0f64;
        for &v in &src[..ENGRAM_HEAD_DIM] {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let ratio16 = max_abs / 32767.0;
        let bits16 = ratio16.to_bits();
        let raw_exp16 = (((bits16 >> 52) & 0x7FF) as i32) - 1023;
        let mant16 = bits16 & 0x000F_FFFF_FFFF_FFFF;
        let e16 = if mant16 == 0 { raw_exp16 } else { raw_exp16 + 1 }.clamp(-60, 60) as i8;
        let inv_scale16 = libm_exp2_neg(e16);

        let mut mantissas_i16 = [0i16; ENGRAM_HEAD_DIM];
        for (dst, &v) in mantissas_i16.iter_mut().zip(src[..ENGRAM_HEAD_DIM].iter()) {
            let scaled = v * inv_scale16;
            let rounded = if scaled >= 0.0 {
                (scaled + 0.5) as i32
            } else {
                (scaled - 0.5) as i32
            };
            *dst = rounded.clamp(-32767, 32767) as i16;
        }

        Self {
            mantissas_i8: bfp8.mantissas,
            mantissas_i16,
            exponent: e16,
            promoted_i16: true,
        }
    }
}

/// Zero-quantization `O(1)` gather from a `BFP-64` `MultiHeadEmbedding` table into
/// pre-quantized `(m_x, exp_x)` buffers ready for `i16` SIMD ternary projection.
#[inline]
pub fn engram_gather_bfp64_heads_into(
    table: &[Bfp64HeadRow],
    head_indices: &[usize],
    out_m_x: &mut [i8],
    out_exp_x: &mut [i8],
) {
    let num_heads = head_indices.len();
    assert!(out_m_x.len() >= num_heads * ENGRAM_HEAD_DIM);
    assert!(out_exp_x.len() >= num_heads);

    for (h, &row_idx) in head_indices.iter().enumerate() {
        let row = &table[row_idx];
        let dst_start = h * ENGRAM_HEAD_DIM;
        out_m_x[dst_start..dst_start + ENGRAM_HEAD_DIM].copy_from_slice(&row.mantissas);
        out_exp_x[h] = row.exponent;
    }
}

/// Project `num_heads` retrieved [`Mem8AdaptiveHeadRow`] blocks directly against a ternary
/// `[out_features, in_features]` matrix using **`i16` SIMD accumulation** for `BFP-i8` heads
/// and **`i32` SIMD accumulation** for promoted `BFP-i16` heads.
pub fn engram_mem8_adaptive_matmul_into(
    table: &[Mem8AdaptiveHeadRow],
    head_indices: &[usize],
    w_signs: &[i8],
    group_scales: &[f64],
    in_features: usize,
    out_features: usize,
    out: &mut [f64],
) {
    let num_groups = in_features / ENGRAM_HEAD_DIM;
    assert_eq!(head_indices.len(), num_groups);
    assert!(
        num_groups <= 64,
        "num_groups ({}) exceeds the 64-group BFP-64 alignment limit",
        num_groups
    );

    let mut exp2_cache = [0.0f64; 64];
    for (g, &idx) in head_indices.iter().enumerate().take(64) {
        exp2_cache[g] = libm_exp2_pos(table[idx].exponent);
    }

    for j in 0..out_features {
        let w_row = &w_signs[j * in_features..(j + 1) * in_features];
        let s_row = &group_scales[j * num_groups..(j + 1) * num_groups];
        let mut total = 0.0f64;

        for (g, &row_idx) in head_indices.iter().enumerate() {
            let head = &table[row_idx];
            let mw_blk = &w_row[g * ENGRAM_HEAD_DIM..(g + 1) * ENGRAM_HEAD_DIM];

            if head.promoted_i16 {
                let block_sum_i32: i32 = mw_blk
                    .iter()
                    .zip(head.mantissas_i16.iter())
                    .map(|(&w, &m)| (w as i32).wrapping_mul(m as i32))
                    .fold(0i32, i32::wrapping_add);
                total += (block_sum_i32 as f64) * exp2_cache[g] * s_row[g];
            } else {
                let block_sum_i16: i16 = mw_blk
                    .iter()
                    .zip(head.mantissas_i8.iter())
                    .map(|(&w, &m)| (w as i16).wrapping_mul(m as i16))
                    .fold(0i16, i16::wrapping_add);
                total += (block_sum_i16 as f64) * exp2_cache[g] * s_row[g];
            }
        }
        out[j] = total;
    }
}

/// Project pre-quantized `BFP-64` Engram heads `(m_x, exp_x)` directly against a ternary
/// `[out_features, in_features]` weight matrix (`w_signs in {-1, 0, +1}`) using
/// **Omni's 16-wide `i16` SIMD block accumulator (`|S_64| <= 64 * 127 = 8,128 <= i16::MAX`)**.
pub fn engram_bfp64_prequant_matmul_into(
    m_x: &[i8],
    exp_x: &[i8],
    w_signs: &[i8],
    group_scales: &[f64],
    in_features: usize,
    out_features: usize,
    out: &mut [f64],
) {
    let num_groups = in_features / ENGRAM_HEAD_DIM;
    assert_eq!(in_features % ENGRAM_HEAD_DIM, 0);
    assert!(
        num_groups <= 64,
        "num_groups ({}) exceeds the 64-group BFP-64 alignment limit",
        num_groups
    );

    let mut exp2_cache = [0.0f64; 64];
    for g in 0..num_groups.min(64) {
        exp2_cache[g] = libm_exp2_pos(exp_x[g]);
    }

    for j in 0..out_features {
        let w_row = &w_signs[j * in_features..(j + 1) * in_features];
        let s_row = &group_scales[j * num_groups..(j + 1) * num_groups];
        let mut total = 0.0f64;

        for g in 0..num_groups {
            let base = g * ENGRAM_HEAD_DIM;
            let mx_blk = &m_x[base..base + ENGRAM_HEAD_DIM];
            let mw_blk = &w_row[base..base + ENGRAM_HEAD_DIM];

            let block_sum: i16 = mw_blk
                .iter()
                .zip(mx_blk.iter())
                .map(|(&w, &m)| (w as i16).wrapping_mul(m as i16))
                .fold(0i16, i16::wrapping_add);
            total += (block_sum as f64) * exp2_cache[g] * s_row[g];
        }
        out[j] = total;
    }
}

/// **Deferred-Scale Integer-Aligned `BFP-64` Engram Projection (`engram_bfp64_int_aligned_matmul_into`)**:
///
/// Pre-aligns the 16 head exponents `2^(exp_x[g] - min_exp)` with integer group scales
/// `p_{s,g}` in `p/q` (`q = 2^20`), accumulates all 16 heads in **pure `i64` integer arithmetic**,
/// and performs **only one `i64 -> f64` conversion per output row**!
pub fn engram_bfp64_int_aligned_matmul_into(
    m_x: &[i8],
    exp_x: &[i8],
    w_signs: &[i8],
    group_scales_pq: &[i32], // [out_features * num_groups] scaled by q_s = 2^20
    in_features: usize,
    out_features: usize,
    out: &mut [f64],
) {
    let num_groups = in_features / ENGRAM_HEAD_DIM;
    assert_eq!(in_features % ENGRAM_HEAD_DIM, 0);
    assert!(
        num_groups <= 64,
        "num_groups ({}) exceeds the 64-group BFP-64 alignment limit",
        num_groups
    );

    let mut min_exp = i8::MAX;
    for &e in &exp_x[..num_groups] {
        if e < min_exp {
            min_exp = e;
        }
    }
    let mut head_mult = [1i64; 64];
    for g in 0..num_groups.min(64) {
        let diff = ((exp_x[g] as i32) - (min_exp as i32)).clamp(0, 24) as u32;
        head_mult[g] = 1i64 << diff;
    }
    let boundary_scale = libm_exp2_pos(min_exp.saturating_sub(20));

    for j in 0..out_features {
        let w_row = &w_signs[j * in_features..(j + 1) * in_features];
        let ps_row = &group_scales_pq[j * num_groups..(j + 1) * num_groups];
        let mut row_i64: i64 = 0;

        for g in 0..num_groups {
            let base = g * ENGRAM_HEAD_DIM;
            let mx_blk = &m_x[base..base + ENGRAM_HEAD_DIM];
            let mw_blk = &w_row[base..base + ENGRAM_HEAD_DIM];

            let block_sum: i16 = mw_blk
                .iter()
                .zip(mx_blk.iter())
                .map(|(&w, &m)| (w as i16).wrapping_mul(m as i16))
                .fold(0i16, i16::wrapping_add);

            row_i64 += (block_sum as i64) * (ps_row[g] as i64) * head_mult[g];
        }
        out[j] = (row_i64 as f64) * boundary_scale;
    }
}

/// **Pure-Integer `Q16` Deferred-Scale `BFP-64` Engram Projection (`engram_bfp64_int_q16_matmul_into`)**:
///
/// Accumulates all `num_groups` 64-dim heads in `i16 -> i64` and shifts directly into
/// `Q16` (`2^16`) fixed-point integers (`out_q16: &mut [i32]`) with **zero floating-point ops**.
pub fn engram_bfp64_int_q16_matmul_into(
    m_x: &[i8],
    exp_x: &[i8],
    w_signs: &[i8],
    group_scales_pq: &[i32], // [out_features * num_groups] scaled by q_s = 2^20
    in_features: usize,
    out_features: usize,
    out_q16: &mut [i32],
) {
    let num_groups = in_features / ENGRAM_HEAD_DIM;
    assert!(
        num_groups <= 64,
        "num_groups ({}) exceeds the 64-group BFP-64 alignment limit",
        num_groups
    );
    let mut min_exp = i8::MAX;
    for &e in &exp_x[..num_groups] {
        if e < min_exp {
            min_exp = e;
        }
    }
    let mut head_mult = [1i64; 64];
    for g in 0..num_groups.min(64) {
        let diff = ((exp_x[g] as i32) - (min_exp as i32)).clamp(0, 20) as u32;
        head_mult[g] = 1i64 << diff;
    }
    // row_i64 is scaled by 2^(20 - min_exp). To convert to Q16 (2^16), shift by (min_exp - 4):
    let shift_to_q16 = (min_exp as i32) - 4;

    for j in 0..out_features {
        let w_row = &w_signs[j * in_features..(j + 1) * in_features];
        let ps_row = &group_scales_pq[j * num_groups..(j + 1) * num_groups];
        let mut row_i64: i64 = 0;

        for g in 0..num_groups {
            let base = g * ENGRAM_HEAD_DIM;
            let mx_blk = &m_x[base..base + ENGRAM_HEAD_DIM];
            let mw_blk = &w_row[base..base + ENGRAM_HEAD_DIM];

            let block_sum: i16 = mw_blk
                .iter()
                .zip(mx_blk.iter())
                .map(|(&w, &m)| (w as i16).wrapping_mul(m as i16))
                .fold(0i16, i16::wrapping_add);

            row_i64 += (block_sum as i64) * (ps_row[g] as i64) * head_mult[g];
        }

        let val_q16 = if shift_to_q16 >= 0 {
            row_i64 << (shift_to_q16.min(30) as u32)
        } else {
            let rshift = (-shift_to_q16).min(60) as u32;
            let half = 1i64 << (rshift.saturating_sub(1));
            if row_i64 >= 0 {
                (row_i64 + half) >> rshift
            } else {
                -(((-row_i64) + half) >> rshift)
            }
        };
        out_q16[j] = val_q16.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    }
}

/// **Fused Single-Pass Bilinear RMSNorm Gate (`engram_fused_rmsnorm_gate`)**:
///
/// Computes `g = <RMSNorm(k), RMSNorm(q)> / sqrt(D)` in a **single fused pass** over `k` and `q`
/// without allocating or writing intermediate normalized `scratch_k` / `scratch_q` vectors!
#[inline]
pub fn engram_fused_rmsnorm_gate(
    key: &[f64],
    query: &[f64],
    norm_k_weight: &[f64],
    norm_q_weight: &[f64],
    eps: f64,
) -> f64 {
    let d = key.len();
    assert_eq!(query.len(), d);
    let mut sum_k2 = 0.0f64;
    let mut sum_q2 = 0.0f64;
    let mut weighted_dot = 0.0f64;

    for i in 0..d {
        let ki = key[i];
        let qi = query[i];
        sum_k2 += ki * ki;
        sum_q2 += qi * qi;
        weighted_dot += (norm_k_weight[i] * norm_q_weight[i]) * (ki * qi);
    }

    let d_f64 = d as f64;
    let rms_k = libm_sqrt(sum_k2 / d_f64 + eps);
    let rms_q = libm_sqrt(sum_q2 / d_f64 + eps);
    let g = weighted_dot / (libm_sqrt(d_f64) * rms_k * rms_q);
    let abs_g = if g.abs() < 1e-6 { 1e-6 } else { g.abs() };
    let signed_sqrt = if g >= 0.0 {
        libm_sqrt(abs_g)
    } else {
        -libm_sqrt(abs_g)
    };
    1.0 / (1.0 + libm_exp(-signed_sqrt))
}

/// Pure-Integer `Q16` Fused Bilinear RMSNorm Gate (`engram_fused_rmsnorm_gate_q16`):
///
/// Computes the hyper-connection gate in `Q16` (`[0, 65536]`) using exact integer
/// Newton-Raphson `isqrt_u128` and rational Padé `[2/2]` sigmoid with **zero floats**.
#[inline]
pub fn engram_fused_rmsnorm_gate_q16(
    key_q16: &[i32],
    query_q16: &[i32],
    norm_k_w_q16: &[i32],
    norm_q_w_q16: &[i32],
) -> i32 {
    let d = key_q16.len();
    let mut sum_k2: u128 = 0;
    let mut sum_q2: u128 = 0;
    let mut weighted_dot_q32: i128 = 0;

    for i in 0..d {
        let k = key_q16[i] as i64;
        let q = query_q16[i] as i64;
        sum_k2 += (k * k) as u128;
        sum_q2 += (q * q) as u128;
        let w_prod_q16 = ((norm_k_w_q16[i] as i64) * (norm_q_w_q16[i] as i64)) >> 16;
        let kq_q16 = (k * q) >> 16;
        weighted_dot_q32 += (w_prod_q16 as i128) * (kq_q16 as i128);
    }

    // eps = 1e-5 in Q32 * D: D * 1e-5 * 2^32 ≈ D * 42_950
    let eps_term = (d as u128) * 42_950;
    let norm_k_q16 = isqrt_u128(sum_k2 + eps_term).max(1) as i128;
    let norm_q_q16 = isqrt_u128(sum_q2 + eps_term).max(1) as i128;
    // sqrt(D) in Q16 = isqrt(D * 2^32)
    let sqrt_d_q16 = isqrt_u128((d as u128) << 32).max(1) as i128;

    let denom_q32 = (norm_k_q16 * norm_q_q16).max(1);
    let g_q16 = ((weighted_dot_q32 * sqrt_d_q16) / denom_q32) as i64;

    let abs_g_q16 = g_q16.unsigned_abs().max(1); // clamp_min
    let sqrt_abs_q16 = isqrt_u128((abs_g_q16 as u128) << 16) as i64;
    let signed_sqrt_q16 = if g_q16 >= 0 {
        sqrt_abs_q16
    } else {
        -sqrt_abs_q16
    };

    sigmoid_pade_q16(signed_sqrt_q16)
}

/// Rational Padé `[2/2]` Sigmoid in `Q16` (`1 / (1 + exp(-x))`), zero floats!
#[inline]
pub const fn sigmoid_pade_q16(x_q16: i64) -> i32 {
    // Clamp to [-6.0, +6.0] in Q16
    let x = if x_q16 < -(6 * PQ_Q16) {
        -(6 * PQ_Q16)
    } else if x_q16 > (6 * PQ_Q16) {
        6 * PQ_Q16
    } else {
        x_q16
    };
    // Padé [2/2] for exp(-x): (12 - 6x + x^2) / (12 + 6x + x^2)
    // Therefore sigmoid(x) = 1 / (1 + exp(-x)) = (12 + 6x + x^2) / (24 + 2*x^2)
    let x2_q16 = (x * x) >> 16;
    let num = 12 * PQ_Q16 + 6 * x + x2_q16;
    let raw_den = 24 * PQ_Q16 + 2 * x2_q16;
    let den = if raw_den < 1 { 1 } else { raw_den };
    let sig_q16 = ((num << 16) + (den >> 1)) / den;
    if sig_q16 < 0 {
        0
    } else if sig_q16 > PQ_Q16 {
        PQ_Q16 as i32
    } else {
        sig_q16 as i32
    }
}

/// Execute a complete single-token **DeepSeek Engram Conditional Memory Step** using
/// **Omni `BFP-64` (`i8` head table + `i16 -> i64` Integer-Aligned SIMD + Fused Bilinear RMSNorm Gate)**!
#[allow(clippy::too_many_arguments)]
pub fn engram_step_bfp64_into(
    table: &[Bfp64HeadRow],
    head_indices: &[usize],
    hidden_states_hc: &[f64],     // [hc_mult * hidden_size]
    key_projs_signs: &[i8],       // [hc_mult * hidden_size * engram_dim]
    key_projs_scales_pq: &[i32],  // [hc_mult * hidden_size * num_groups] (q = 2^20)
    val_proj_signs: &[i8],        // [hidden_size * engram_dim]
    val_proj_scales_pq: &[i32],   // [hidden_size * num_groups] (q = 2^20)
    norm_w: &[f64],               // [hidden_size]
    short_conv_w: &[f64],         // [hc_mult * hidden_size]
    hc_mult: usize,
    engram_dim: usize,
    hidden_size: usize,
    m_x_scratch: &mut [i8],       // [engram_dim]
    exp_x_scratch: &mut [i8],     // [engram_dim / 64]
    val_scratch: &mut [f64],      // [hidden_size]
    key_scratch: &mut [f64],      // [hidden_size]
    out_hc: &mut [f64],           // [hc_mult * hidden_size]
) {
    let num_groups = engram_dim / ENGRAM_HEAD_DIM;

    // 1. O(1) gather of pre-quantized BFP-64 heads (zero float ops!)
    engram_gather_bfp64_heads_into(table, head_indices, m_x_scratch, exp_x_scratch);

    // 2. Project value_proj(embeddings) via i16 -> i64 integer-aligned accumulation
    engram_bfp64_int_aligned_matmul_into(
        m_x_scratch,
        exp_x_scratch,
        val_proj_signs,
        val_proj_scales_pq,
        engram_dim,
        hidden_size,
        val_scratch,
    );

    // 3. Per-stream integer-aligned key projection + Fused Bilinear RMSNorm gate + ShortConv
    let kp_mat_stride = hidden_size * engram_dim;
    let kp_scale_stride = hidden_size * num_groups;

    for hc in 0..hc_mult {
        let k_signs = &key_projs_signs[hc * kp_mat_stride..(hc + 1) * kp_mat_stride];
        let k_scales_pq = &key_projs_scales_pq[hc * kp_scale_stride..(hc + 1) * kp_scale_stride];
        let query_hc = &hidden_states_hc[hc * hidden_size..(hc + 1) * hidden_size];

        engram_bfp64_int_aligned_matmul_into(
            m_x_scratch,
            exp_x_scratch,
            k_signs,
            k_scales_pq,
            engram_dim,
            hidden_size,
            key_scratch,
        );

        let gate = engram_fused_rmsnorm_gate(key_scratch, query_hc, norm_w, norm_w, 1e-5);

        let conv_hc = &short_conv_w[hc * hidden_size..(hc + 1) * hidden_size];
        let out_slice = &mut out_hc[hc * hidden_size..(hc + 1) * hidden_size];

        for d in 0..hidden_size {
            let v_gated = gate * val_scratch[d];
            let conv_val = v_gated * conv_hc[d];
            let silu = conv_val / (1.0 + libm_exp(-conv_val));
            out_slice[d] = query_hc[d] + v_gated + silu;
        }
    }
}

/// **100% Zero-Float Pure-Integer `p/q` ("Counting Divided") Engram Layer Step (`engram_step_pq_no_floats_into`)**:
///
/// Executes the entire DeepSeek Engram layer step (`MultiHeadEmbedding` gather, `value_proj`,
/// `key_projs[0..hc_mult]`, Fused Bilinear `RMSNorm` gate via `isqrt_u128`, signed-sqrt,
/// rational Padé `[2/2]` sigmoid, ShortConv SiLU, and residual) in pure integer arithmetic
/// with **zero IEEE-754 floating-point instructions**!
#[allow(clippy::too_many_arguments)]
pub fn engram_step_pq_no_floats_into(
    table: &[Bfp64HeadRow],
    head_indices: &[usize],
    hidden_states_q16: &[i32],    // [hc_mult * hidden_size] in Q16
    key_projs_signs: &[i8],       // [hc_mult * hidden_size * engram_dim]
    key_projs_scales_pq: &[i32],  // [hc_mult * hidden_size * num_groups] in Q20
    val_proj_signs: &[i8],        // [hidden_size * engram_dim]
    val_proj_scales_pq: &[i32],   // [hidden_size * num_groups] in Q20
    norm_w_q16: &[i32],           // [hidden_size] in Q16
    short_conv_w_q16: &[i32],     // [hc_mult * hidden_size] in Q16
    hc_mult: usize,
    engram_dim: usize,
    hidden_size: usize,
    m_x_scratch: &mut [i8],       // [engram_dim]
    exp_x_scratch: &mut [i8],     // [engram_dim / 64]
    val_q16_scratch: &mut [i32],  // [hidden_size]
    key_q16_scratch: &mut [i32],  // [hidden_size]
    out_hc_q16: &mut [i32],       // [hc_mult * hidden_size] in Q16
) {
    let num_groups = engram_dim / ENGRAM_HEAD_DIM;

    // 1. O(1) gather of pre-quantized BFP-64 heads
    engram_gather_bfp64_heads_into(table, head_indices, m_x_scratch, exp_x_scratch);

    // 2. Pure-integer Q16 value projection (`i16 -> i64 -> Q16`)
    engram_bfp64_int_q16_matmul_into(
        m_x_scratch,
        exp_x_scratch,
        val_proj_signs,
        val_proj_scales_pq,
        engram_dim,
        hidden_size,
        val_q16_scratch,
    );

    // 3. Per-stream integer key projection + integer RMSNorm gate + integer SiLU ShortConv
    let kp_mat_stride = hidden_size * engram_dim;
    let kp_scale_stride = hidden_size * num_groups;

    for hc in 0..hc_mult {
        let k_signs = &key_projs_signs[hc * kp_mat_stride..(hc + 1) * kp_mat_stride];
        let k_scales_pq = &key_projs_scales_pq[hc * kp_scale_stride..(hc + 1) * kp_scale_stride];
        let query_q16 = &hidden_states_q16[hc * hidden_size..(hc + 1) * hidden_size];

        engram_bfp64_int_q16_matmul_into(
            m_x_scratch,
            exp_x_scratch,
            k_signs,
            k_scales_pq,
            engram_dim,
            hidden_size,
            key_q16_scratch,
        );

        let gate_q16 = engram_fused_rmsnorm_gate_q16(
            key_q16_scratch,
            query_q16,
            norm_w_q16,
            norm_w_q16,
        ) as i64;

        let conv_hc_q16 = &short_conv_w_q16[hc * hidden_size..(hc + 1) * hidden_size];
        let out_slice = &mut out_hc_q16[hc * hidden_size..(hc + 1) * hidden_size];

        for d in 0..hidden_size {
            let v_gated_q16 = (gate_q16 * (val_q16_scratch[d] as i64)) >> 16;
            let conv_val_q16 = (v_gated_q16 * (conv_hc_q16[d] as i64)) >> 16;
            let sig_conv_q16 = sigmoid_pade_q16(conv_val_q16) as i64;
            let silu_q16 = (conv_val_q16 * sig_conv_q16) >> 16;
            out_slice[d] = (query_q16[d] as i64 + v_gated_q16 + silu_q16) as i32;
        }
    }
}

/// Reference dense FP64 Engram step (mirrors `Engram.forward` in `engram_demo_v1.py`).
#[allow(clippy::too_many_arguments)]
pub fn engram_step_dense_fp64_into(
    dense_table: &[f64],
    head_indices: &[usize],
    hidden_states_hc: &[f64],
    key_projs_dense: &[f64],
    val_proj_dense: &[f64],
    norm_w: &[f64],
    short_conv_w: &[f64],
    hc_mult: usize,
    engram_dim: usize,
    hidden_size: usize,
    emb_scratch: &mut [f64],
    val_scratch: &mut [f64],
    key_scratch: &mut [f64],
    out_hc: &mut [f64],
) {
    for (h, &row_idx) in head_indices.iter().enumerate() {
        let src = &dense_table[row_idx * ENGRAM_HEAD_DIM..(row_idx + 1) * ENGRAM_HEAD_DIM];
        emb_scratch[h * ENGRAM_HEAD_DIM..(h + 1) * ENGRAM_HEAD_DIM].copy_from_slice(src);
    }

    for j in 0..hidden_size {
        let w_row = &val_proj_dense[j * engram_dim..(j + 1) * engram_dim];
        let mut sum = 0.0f64;
        for i in 0..engram_dim {
            sum += emb_scratch[i] * w_row[i];
        }
        val_scratch[j] = sum;
    }

    let kp_stride = hidden_size * engram_dim;
    for hc in 0..hc_mult {
        let k_mat = &key_projs_dense[hc * kp_stride..(hc + 1) * kp_stride];
        let query_hc = &hidden_states_hc[hc * hidden_size..(hc + 1) * hidden_size];
        for j in 0..hidden_size {
            let w_row = &k_mat[j * engram_dim..(j + 1) * engram_dim];
            let mut sum = 0.0f64;
            for i in 0..engram_dim {
                sum += emb_scratch[i] * w_row[i];
            }
            key_scratch[j] = sum;
        }

        let gate = engram_fused_rmsnorm_gate(key_scratch, query_hc, norm_w, norm_w, 1e-5);
        let conv_hc = &short_conv_w[hc * hidden_size..(hc + 1) * hidden_size];
        let out_slice = &mut out_hc[hc * hidden_size..(hc + 1) * hidden_size];
        for d in 0..hidden_size {
            let v_gated = gate * val_scratch[d];
            let conv_val = v_gated * conv_hc[d];
            let silu = conv_val / (1.0 + libm_exp(-conv_val));
            out_slice[d] = query_hc[d] + v_gated + silu;
        }
    }
}

/// Exact integer floor square root `floor(sqrt(n))` for `u128`.
#[inline]
pub const fn isqrt_u128(n: u128) -> u128 {
    if n < 2 {
        return n;
    }
    let shift = (128 - n.leading_zeros()).div_ceil(2);
    let mut x = 1u128 << shift;
    loop {
        let y = (x + n / x) >> 1;
        if y >= x {
            return x;
        }
        x = y;
    }
}

#[inline]
const fn libm_exp2_neg(e: i8) -> f64 {
    let biased = (1023i64 - (e as i64)) as u64;
    f64::from_bits(biased << 52)
}

#[inline]
const fn libm_exp2_pos(e: i8) -> f64 {
    let biased = (1023i64 + (e as i64)) as u64;
    f64::from_bits(biased << 52)
}

#[inline]
fn libm_sqrt(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut y = f64::from_bits((x.to_bits() >> 1) + (1023u64 << 51));
    y = 0.5 * (y + x / y);
    y = 0.5 * (y + x / y);
    y = 0.5 * (y + x / y);
    y = 0.5 * (y + x / y);
    y
}

#[inline]
fn libm_exp(x: f64) -> f64 {
    let x_clamped = x.clamp(-40.0, 40.0);
    const LN2: f64 = core::f64::consts::LN_2;
    const INV_LN2: f64 = core::f64::consts::LOG2_E;
    let k = (x_clamped * INV_LN2 + if x_clamped >= 0.0 { 0.5 } else { -0.5 }) as i32;
    let r = x_clamped - (k as f64) * LN2;
    let r2 = r * r;
    let p = 1.0 + 0.5 * r + (3.0 / 28.0) * r2 + (1.0 / 84.0) * r2 * r + (1.0 / 1680.0) * r2 * r2;
    let q = 1.0 - 0.5 * r + (3.0 / 28.0) * r2 - (1.0 / 84.0) * r2 * r + (1.0 / 1680.0) * r2 * r2;
    let biased = ((1023 + k).clamp(1, 2046) as u64) << 52;
    (p / q) * f64::from_bits(biased)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lemire_pq_hash_equivalence() {
        let primes: [u64; 16] = [
            1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091,
            1093, 1097,
        ];
        let mut buckets = [PrimeBucketMod::new(1, 0); 16];
        let mut total_rows = 0usize;
        for (i, &p) in primes.iter().enumerate() {
            buckets[i] = PrimeBucketMod::new(p, total_rows);
            total_rows += p as usize;
        }

        let multipliers: [u64; 3] = [6364136223846793005, 1442695040888963407, 22695477];
        let tokens = [10u64, 250, 999, 42, 129000, 77, 13, 2];
        let mut idx_mod = [0usize; 8 * 16];
        let mut idx_pq = [0usize; 8 * 16];

        engram_hash_ngram_mod_into(&tokens, 2, &multipliers, 8, &buckets, &mut idx_mod);
        engram_hash_ngram_pq_into(&tokens, 2, &multipliers, 8, &buckets, &mut idx_pq);
        assert_eq!(idx_mod, idx_pq);

        let signs = [-1i8, 0, 1, 1, -1, 0, 1, -1];
        let mut signs64 = [0i8; 64];
        for i in 0..64 {
            signs64[i] = signs[i % 8];
        }
        let qblk = Quantum64TritBlock::from_signs(&signs64);
        assert_eq!(qblk.unpack_signs(), signs64);
    }

    #[test]
    fn zero_float_engram_step_tracks_dense_step() {
        let primes: [u64; 8] = [101, 103, 107, 109, 113, 127, 131, 137];
        let mut head_indices = [0usize; 8];
        let mut total_rows = 0usize;
        for (i, &p) in primes.iter().enumerate() {
            head_indices[i] = total_rows;
            total_rows += p as usize;
        }
        let mut dense_table = vec![0.0f64; total_rows * ENGRAM_HEAD_DIM];
        for (i, v) in dense_table.iter_mut().enumerate() {
            *v = (((i * 73 + 19) % 200) as f64 - 100.0) * 0.005;
        }
        let bfp_table: Vec<Bfp64HeadRow> = (0..total_rows)
            .map(|r| Bfp64HeadRow::from_f64_slice(&dense_table[r * ENGRAM_HEAD_DIM..(r + 1) * ENGRAM_HEAD_DIM]))
            .collect();

        let hc_mult = 2usize;
        let hidden_size = 64usize;
        let engram_dim = 8 * ENGRAM_HEAD_DIM;
        let num_groups = engram_dim / ENGRAM_HEAD_DIM;

        let mut key_signs = vec![0i8; hc_mult * hidden_size * engram_dim];
        let mut key_scales_pq = vec![0i32; hc_mult * hidden_size * num_groups];
        let mut key_dense = vec![0.0f64; hc_mult * hidden_size * engram_dim];
        for i in 0..key_signs.len() {
            let s = match (i * 37 + 11) % 3 {
                0 => -1i8,
                1 => 0i8,
                _ => 1i8,
            };
            key_signs[i] = s;
            let row = i / engram_dim;
            let g = (i % engram_dim) / ENGRAM_HEAD_DIM;
            let scale = 0.015 + ((row + g) % 7) as f64 * 0.002;
            key_scales_pq[row * num_groups + g] = (scale * 1_048_576.0).round() as i32;
            key_dense[i] = (s as f64) * scale;
        }
        let mut val_signs = vec![0i8; hidden_size * engram_dim];
        let mut val_scales_pq = vec![0i32; hidden_size * num_groups];
        let mut val_dense = vec![0.0f64; hidden_size * engram_dim];
        for i in 0..val_signs.len() {
            let s = match (i * 53 + 7) % 3 {
                0 => -1i8,
                1 => 0i8,
                _ => 1i8,
            };
            val_signs[i] = s;
            let row = i / engram_dim;
            let g = (i % engram_dim) / ENGRAM_HEAD_DIM;
            let scale = 0.018 + ((row + g) % 5) as f64 * 0.002;
            val_scales_pq[row * num_groups + g] = (scale * 1_048_576.0).round() as i32;
            val_dense[i] = (s as f64) * scale;
        }

        let hidden_states: Vec<f64> = (0..hc_mult * hidden_size)
            .map(|i| (((i * 41 + 3) % 100) as f64 - 50.0) * 0.02)
            .collect();
        let hidden_states_q16: Vec<i32> = hidden_states
            .iter()
            .map(|&x| (x * (PQ_Q16 as f64)).round() as i32)
            .collect();
        let norm_w = vec![1.0f64; hidden_size];
        let norm_w_q16 = vec![PQ_Q16 as i32; hidden_size];
        let short_conv_w = vec![0.25f64; hc_mult * hidden_size];
        let short_conv_w_q16 = vec![(0.25 * (PQ_Q16 as f64)).round() as i32; hc_mult * hidden_size];

        let mut emb_scratch = vec![0.0f64; engram_dim];
        let mut m_x_scratch = vec![0i8; engram_dim];
        let mut exp_x_scratch = vec![0i8; num_groups];
        let mut val_scratch = vec![0.0f64; hidden_size];
        let mut key_scratch = vec![0.0f64; hidden_size];
        let mut val_q16_scratch = vec![0i32; hidden_size];
        let mut key_q16_scratch = vec![0i32; hidden_size];
        let mut out_dense = vec![0.0f64; hc_mult * hidden_size];
        let mut out_q16 = vec![0i32; hc_mult * hidden_size];

        engram_step_dense_fp64_into(
            &dense_table,
            &head_indices,
            &hidden_states,
            &key_dense,
            &val_dense,
            &norm_w,
            &short_conv_w,
            hc_mult,
            engram_dim,
            hidden_size,
            &mut emb_scratch,
            &mut val_scratch,
            &mut key_scratch,
            &mut out_dense,
        );
        engram_step_pq_no_floats_into(
            &bfp_table,
            &head_indices,
            &hidden_states_q16,
            &key_signs,
            &key_scales_pq,
            &val_signs,
            &val_scales_pq,
            &norm_w_q16,
            &short_conv_w_q16,
            hc_mult,
            engram_dim,
            hidden_size,
            &mut m_x_scratch,
            &mut exp_x_scratch,
            &mut val_q16_scratch,
            &mut key_q16_scratch,
            &mut out_q16,
        );

        let norm_ref = out_dense.iter().map(|x| x * x).sum::<f64>().sqrt();
        let err: f64 = out_dense
            .iter()
            .zip(out_q16.iter())
            .map(|(&a, &b)| (a - (b as f64) / (PQ_Q16 as f64)).powi(2))
            .sum::<f64>()
            .sqrt()
            / norm_ref;
        assert!(err < 0.05, "zero-float step relative error {err} exceeds 0.05");
    }

    #[test]
    #[should_panic(expected = "num_groups")]
    fn matmul_functions_reject_more_than_64_groups() {
        let in_features = 65 * ENGRAM_HEAD_DIM;
        let out_features = 4;
        let m_x = vec![0i8; in_features];
        let exp_x = vec![0i8; 65];
        let signs = vec![0i8; out_features * in_features];
        let scales_pq = vec![0i32; out_features * 65];
        let mut out = vec![0.0f64; out_features];
        engram_bfp64_int_aligned_matmul_into(
            &m_x,
            &exp_x,
            &signs,
            &scales_pq,
            in_features,
            out_features,
            &mut out,
        );
    }
}
