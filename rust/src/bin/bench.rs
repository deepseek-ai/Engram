use engram_pq::{
    Bfp64HeadRow, ENGRAM_DEFAULT_HC_MULT, ENGRAM_HEAD_DIM, Mem8AdaptiveHeadRow, PQ_Q16,
    PrimeBucketMod, engram_bfp64_int_aligned_matmul_into, engram_bfp64_int_q16_matmul_into,
    engram_gather_bfp64_heads_into, engram_hash_ngram_mod_into, engram_hash_ngram_mod64_into,
    engram_hash_ngram_pq_into, engram_hash_ngram_pq64_into, engram_mem8_adaptive_matmul_into,
    engram_step_bfp64_into, engram_step_dense_fp64_into, engram_step_pq_no_floats_into,
};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let max_ngram_size = 3;
    let n_heads_per_ngram = 8;
    let total_heads = (max_ngram_size - 1) * n_heads_per_ngram; // 16 heads
    let engram_dim = total_heads * ENGRAM_HEAD_DIM; // 16 * 64 = 1024
    let hidden_size = 512;
    let hc_mult = ENGRAM_DEFAULT_HC_MULT; // 4

    let primes: [u64; 16] = [
        1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093,
        1097,
    ];
    let mut buckets = Vec::with_capacity(16);
    let mut total_table_rows = 0usize;
    for &p in &primes {
        buckets.push(PrimeBucketMod::new(p, total_table_rows));
        total_table_rows += p as usize;
    }

    let multipliers: [u64; 3] = [6364136223846793005, 1442695040888963407, 22695477];
    let seq_len = 1024;
    let tokens: Vec<u64> = (0..seq_len)
        .map(|i| ((i as u64) * 131 + 17) % 129_280)
        .collect();

    let mut idx_mod = vec![0usize; seq_len * total_heads];
    let mut idx_pq = vec![0usize; seq_len * total_heads];
    let mut idx_mod64 = vec![0usize; seq_len * total_heads];
    let mut idx_pq64 = vec![0usize; seq_len * total_heads];

    engram_hash_ngram_mod_into(&tokens, 2, &multipliers, n_heads_per_ngram, &buckets, &mut idx_mod);
    engram_hash_ngram_pq_into(&tokens, 2, &multipliers, n_heads_per_ngram, &buckets, &mut idx_pq);
    assert_eq!(idx_mod, idx_pq, "Lemire p/q 32-bit prime hashing must be bit-exact with % prime");

    engram_hash_ngram_mod64_into(&tokens, 2, &multipliers, n_heads_per_ngram, &buckets, &mut idx_mod64);
    engram_hash_ngram_pq64_into(&tokens, 2, &multipliers, n_heads_per_ngram, &buckets, &mut idx_pq64);
    assert_eq!(idx_mod64, idx_pq64, "Lemire p/q 64-bit prime hashing must be bit-exact with % prime");

    let hash_iters = 500;
    let t_h0 = Instant::now();
    for _ in 0..hash_iters {
        engram_hash_ngram_mod_into(
            black_box(&tokens),
            2,
            black_box(&multipliers),
            n_heads_per_ngram,
            black_box(&buckets),
            &mut idx_mod,
        );
    }
    let hash_mod_us = t_h0.elapsed().as_secs_f64() * 1e6 / (hash_iters as f64);

    let t_h1 = Instant::now();
    for _ in 0..hash_iters {
        engram_hash_ngram_pq_into(
            black_box(&tokens),
            2,
            black_box(&multipliers),
            n_heads_per_ngram,
            black_box(&buckets),
            &mut idx_pq,
        );
    }
    let hash_pq_us = t_h1.elapsed().as_secs_f64() * 1e6 / (hash_iters as f64);

    let t_h2 = Instant::now();
    for _ in 0..hash_iters {
        engram_hash_ngram_mod64_into(
            black_box(&tokens),
            2,
            black_box(&multipliers),
            n_heads_per_ngram,
            black_box(&buckets),
            &mut idx_mod64,
        );
    }
    let hash_mod64_us = t_h2.elapsed().as_secs_f64() * 1e6 / (hash_iters as f64);

    let t_h3 = Instant::now();
    for _ in 0..hash_iters {
        engram_hash_ngram_pq64_into(
            black_box(&tokens),
            2,
            black_box(&multipliers),
            n_heads_per_ngram,
            black_box(&buckets),
            &mut idx_pq64,
        );
    }
    let hash_pq64_us = t_h3.elapsed().as_secs_f64() * 1e6 / (hash_iters as f64);

    // Build MultiHeadEmbedding table (`total_table_rows x 64`)
    let mut dense_table = vec![0.0f64; total_table_rows * ENGRAM_HEAD_DIM];
    for (i, v) in dense_table.iter_mut().enumerate() {
        *v = (((i * 73 + 19) % 200) as f64 - 100.0) * 0.005;
    }
    let bfp_table: Vec<Bfp64HeadRow> = (0..total_table_rows)
        .map(|r| Bfp64HeadRow::from_f64_slice(&dense_table[r * ENGRAM_HEAD_DIM..(r + 1) * ENGRAM_HEAD_DIM]))
        .collect();
    let mem8_table: Vec<Mem8AdaptiveHeadRow> = (0..total_table_rows)
        .map(|r| {
            Mem8AdaptiveHeadRow::from_f64_slice(
                &dense_table[r * ENGRAM_HEAD_DIM..(r + 1) * ENGRAM_HEAD_DIM],
                0.0052,
            )
        })
        .collect();

    let num_groups = engram_dim / ENGRAM_HEAD_DIM; // 16
    let kp_len = hc_mult * hidden_size * engram_dim;
    let vp_len = hidden_size * engram_dim;

    let mut key_projs_signs = vec![0i8; kp_len];
    let mut key_projs_scales = vec![0.02f64; hc_mult * hidden_size * num_groups];
    let mut key_projs_dense = vec![0.0f64; kp_len];
    for i in 0..kp_len {
        let s = match (i * 37 + 11) % 3 {
            0 => -1i8,
            1 => 0i8,
            _ => 1i8,
        };
        key_projs_signs[i] = s;
        let row = i / engram_dim;
        let g = (i % engram_dim) / ENGRAM_HEAD_DIM;
        let scale = 0.015 + ((row + g) % 7) as f64 * 0.002;
        key_projs_scales[row * num_groups + g] = scale;
        key_projs_dense[i] = (s as f64) * scale;
    }
    let key_projs_scales_pq: Vec<i32> = key_projs_scales
        .iter()
        .map(|&s| (s * 1_048_576.0).round() as i32)
        .collect();

    let mut val_proj_signs = vec![0i8; vp_len];
    let mut val_proj_scales = vec![0.02f64; hidden_size * num_groups];
    let mut val_proj_dense = vec![0.0f64; vp_len];
    for i in 0..vp_len {
        let s = match (i * 53 + 7) % 3 {
            0 => -1i8,
            1 => 0i8,
            _ => 1i8,
        };
        val_proj_signs[i] = s;
        let row = i / engram_dim;
        let g = (i % engram_dim) / ENGRAM_HEAD_DIM;
        let scale = 0.018 + ((row + g) % 5) as f64 * 0.002;
        val_proj_scales[row * num_groups + g] = scale;
        val_proj_dense[i] = (s as f64) * scale;
    }
    let val_proj_scales_pq: Vec<i32> = val_proj_scales
        .iter()
        .map(|&s| (s * 1_048_576.0).round() as i32)
        .collect();

    let hidden_states_hc: Vec<f64> = (0..hc_mult * hidden_size)
        .map(|i| (((i * 41 + 3) % 100) as f64 - 50.0) * 0.02)
        .collect();
    let hidden_states_q16: Vec<i32> = hidden_states_hc
        .iter()
        .map(|&x| (x * (PQ_Q16 as f64)).round() as i32)
        .collect();
    let norm_w = vec![1.0f64; hidden_size];
    let norm_w_q16 = vec![PQ_Q16 as i32; hidden_size];
    let short_conv_w = vec![0.25f64; hc_mult * hidden_size];
    let short_conv_w_q16 = vec![(0.25 * (PQ_Q16 as f64)).round() as i32; hc_mult * hidden_size];

    let head_indices = &idx_pq[0..total_heads];
    let mut emb_scratch = vec![0.0f64; engram_dim];
    let mut m_x_scratch = vec![0i8; engram_dim];
    let mut exp_x_scratch = vec![0i8; num_groups];
    let mut val_scratch = vec![0.0f64; hidden_size];
    let mut key_scratch = vec![0.0f64; hidden_size];
    let mut val_q16_scratch = vec![0i32; hidden_size];
    let mut key_q16_scratch = vec![0i32; hidden_size];
    let mut out_dense = vec![0.0f64; hc_mult * hidden_size];
    let mut out_bfp64 = vec![0.0f64; hc_mult * hidden_size];
    let mut out_pq_q16 = vec![0i32; hc_mult * hidden_size];

    engram_gather_bfp64_heads_into(&bfp_table, head_indices, &mut m_x_scratch, &mut exp_x_scratch);

    // 1. Isolated Projection Benchmark (`1024 -> 512`)
    for (h, &row_idx) in head_indices.iter().enumerate() {
        let src = &dense_table[row_idx * ENGRAM_HEAD_DIM..(row_idx + 1) * ENGRAM_HEAD_DIM];
        emb_scratch[h * ENGRAM_HEAD_DIM..(h + 1) * ENGRAM_HEAD_DIM].copy_from_slice(src);
    }
    let mut val_dense_out = vec![0.0f64; hidden_size];
    let mut val_int_out = vec![0.0f64; hidden_size];
    let mut val_q16_out = vec![0i32; hidden_size];
    let mut val_mem8_out = vec![0.0f64; hidden_size];

    let proj_iters = 500;
    let t_p0 = Instant::now();
    for _ in 0..proj_iters {
        for j in 0..hidden_size {
            let w_row = &val_proj_dense[j * engram_dim..(j + 1) * engram_dim];
            let mut sum = 0.0f64;
            for i in 0..engram_dim {
                sum += emb_scratch[i] * w_row[i];
            }
            val_dense_out[j] = sum;
        }
        black_box(&val_dense_out);
    }
    let dense_proj_us = t_p0.elapsed().as_secs_f64() * 1e6 / (proj_iters as f64);

    let t_p1 = Instant::now();
    for _ in 0..proj_iters {
        engram_bfp64_int_aligned_matmul_into(
            black_box(&m_x_scratch),
            black_box(&exp_x_scratch),
            black_box(&val_proj_signs),
            black_box(&val_proj_scales_pq),
            engram_dim,
            hidden_size,
            &mut val_int_out,
        );
        black_box(&val_int_out);
    }
    let int_aligned_proj_us = t_p1.elapsed().as_secs_f64() * 1e6 / (proj_iters as f64);

    let t_p2 = Instant::now();
    for _ in 0..proj_iters {
        engram_bfp64_int_q16_matmul_into(
            black_box(&m_x_scratch),
            black_box(&exp_x_scratch),
            black_box(&val_proj_signs),
            black_box(&val_proj_scales_pq),
            engram_dim,
            hidden_size,
            &mut val_q16_out,
        );
        black_box(&val_q16_out);
    }
    let int_q16_proj_us = t_p2.elapsed().as_secs_f64() * 1e6 / (proj_iters as f64);

    let t_p3 = Instant::now();
    for _ in 0..proj_iters {
        engram_mem8_adaptive_matmul_into(
            black_box(&mem8_table),
            black_box(head_indices),
            black_box(&val_proj_signs),
            black_box(&val_proj_scales),
            engram_dim,
            hidden_size,
            &mut val_mem8_out,
        );
        black_box(&val_mem8_out);
    }
    let mem8_proj_us = t_p3.elapsed().as_secs_f64() * 1e6 / (proj_iters as f64);

    let proj_norm_ref = val_dense_out.iter().map(|x| x * x).sum::<f64>().sqrt();
    let proj_err = val_dense_out
        .iter()
        .zip(val_int_out.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
        / proj_norm_ref;
    let q16_proj_err = val_dense_out
        .iter()
        .zip(val_q16_out.iter())
        .map(|(&a, &b)| (a - (b as f64) / (PQ_Q16 as f64)).powi(2))
        .sum::<f64>()
        .sqrt()
        / proj_norm_ref;
    let mem8_err = val_dense_out
        .iter()
        .zip(val_mem8_out.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
        / proj_norm_ref;
    let promoted_heads = head_indices
        .iter()
        .filter(|&&idx| mem8_table[idx].promoted_i16)
        .count();

    // 2. Full Engram Layer Step Benchmark (`2.62M` weights + HC Gate + ShortConv)
    let step_iters = 200;
    let t_s0 = Instant::now();
    for _ in 0..step_iters {
        engram_step_dense_fp64_into(
            black_box(&dense_table),
            black_box(head_indices),
            black_box(&hidden_states_hc),
            black_box(&key_projs_dense),
            black_box(&val_proj_dense),
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
    }
    let dense_step_us = t_s0.elapsed().as_secs_f64() * 1e6 / (step_iters as f64);

    let t_s1 = Instant::now();
    for _ in 0..step_iters {
        engram_step_bfp64_into(
            black_box(&bfp_table),
            black_box(head_indices),
            black_box(&hidden_states_hc),
            black_box(&key_projs_signs),
            black_box(&key_projs_scales_pq),
            black_box(&val_proj_signs),
            black_box(&val_proj_scales_pq),
            &norm_w,
            &short_conv_w,
            hc_mult,
            engram_dim,
            hidden_size,
            &mut m_x_scratch,
            &mut exp_x_scratch,
            &mut val_scratch,
            &mut key_scratch,
            &mut out_bfp64,
        );
    }
    let bfp64_step_us = t_s1.elapsed().as_secs_f64() * 1e6 / (step_iters as f64);

    let t_s2 = Instant::now();
    for _ in 0..step_iters {
        engram_step_pq_no_floats_into(
            black_box(&bfp_table),
            black_box(head_indices),
            black_box(&hidden_states_q16),
            black_box(&key_projs_signs),
            black_box(&key_projs_scales_pq),
            black_box(&val_proj_signs),
            black_box(&val_proj_scales_pq),
            &norm_w_q16,
            &short_conv_w_q16,
            hc_mult,
            engram_dim,
            hidden_size,
            &mut m_x_scratch,
            &mut exp_x_scratch,
            &mut val_q16_scratch,
            &mut key_q16_scratch,
            &mut out_pq_q16,
        );
    }
    let pq_no_floats_step_us = t_s2.elapsed().as_secs_f64() * 1e6 / (step_iters as f64);

    let step_norm_ref = out_dense.iter().map(|x| x * x).sum::<f64>().sqrt();
    let step_bfp_err = out_dense
        .iter()
        .zip(out_bfp64.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
        / step_norm_ref;
    let step_pq_err = out_dense
        .iter()
        .zip(out_pq_q16.iter())
        .map(|(&a, &b)| (a - (b as f64) / (PQ_Q16 as f64)).powi(2))
        .sum::<f64>()
        .sqrt()
        / step_norm_ref;

    let demo_rows = 10_342_604usize;
    let v41_rows = 384_006_168usize;
    let demo_fp32_gb = (demo_rows as f64 * 64.0 * 4.0) / 1e9;
    let demo_bfp64_gb = (demo_rows as f64 * 65.0) / 1e9;
    let v41_fp32_gb = (v41_rows as f64 * 64.0 * 4.0) / 1e9;
    let v41_bfp64_gb = (v41_rows as f64 * 65.0) / 1e9;

    println!("=========================================================================================");
    println!("  8b-is / Engram — Sovereign p/q + Omni BFP-64 + MEM|8 + Quantum 64-Trit Benchmark");
    println!("=========================================================================================");
    println!("  1. Multi-Head N-Gram Prime Hashing (1024 tokens x 16 prime heads = 16,384 hashes):");
    println!("     - 32-bit Folded Hardware `% prime`:        {:>8.2} us (1.00x)", hash_mod_us);
    println!(
        "     - 32-bit Folded Lemire p/q (q=2^64):       {:>8.2} us ({:.2}x speedup, bit-exact!)",
        hash_pq_us,
        hash_mod_us / hash_pq_us
    );
    println!("     - 64-bit Unfolded Hardware `% prime`:      {:>8.2} us (1.00x)", hash_mod64_us);
    println!(
        "     - 64-bit Unfolded Lemire p/q (q=2^64):     {:>8.2} us ({:.2}x speedup, bit-exact!)",
        hash_pq64_us,
        hash_mod64_us / hash_pq64_us
    );
    println!("  2. Native 64-Dim Head Table Compression (`D_head = 64 == BFP-64 block size`):");
    println!(
        "     - Engram Demo Table (10.34M rows):         {:.2} GB (FP32) -> {:.2} GB (BFP-64) = {:.2}x smaller",
        demo_fp32_gb,
        demo_bfp64_gb,
        demo_fp32_gb / demo_bfp64_gb
    );
    println!(
        "     - DeepSeek-V4.1 Table (384.0M rows):       {:.2} GB (FP32) -> {:.2} GB (BFP-64) = {:.2}x smaller",
        v41_fp32_gb,
        v41_bfp64_gb,
        v41_fp32_gb / v41_bfp64_gb
    );
    println!("  3. Isolated Engram Head Projection (`1024 -> 512`, 16 Pre-Quantized 64-Dim Heads):");
    println!("     - Dense FP64 Projection:                   {:>8.2} us (1.00x)", dense_proj_us);
    println!(
        "     - Integer-Aligned BFP-64 (i16 -> i64):     {:>8.2} us ({:.2}x speedup) | rel_err: {:.5}",
        int_aligned_proj_us,
        dense_proj_us / int_aligned_proj_us,
        proj_err
    );
    println!(
        "     - Zero-Float Q16 BFP-64 (i16->i64->Q16):   {:>8.2} us ({:.2}x speedup) | rel_err: {:.5}",
        int_q16_proj_us,
        dense_proj_us / int_q16_proj_us,
        q16_proj_err
    );
    println!(
        "     - MEM|8 Adaptive ({:>2}/16 prom i8->i16):     {:>8.2} us ({:.2}x speedup) | rel_err: {:.5}",
        promoted_heads,
        mem8_proj_us,
        dense_proj_us / mem8_proj_us,
        mem8_err
    );
    println!(
        "  4. Full Engram Layer Step (16x64 Gather + 4 KeyProjs + 1 ValProj [2.62M weights] + Fused Gate + ShortConv):"
    );
    println!("     - Dense FP64 Engram Step:                  {:>8.2} us (1.00x)", dense_step_us);
    println!(
        "     - Int-Aligned BFP-64 + Fused Gate Step:    {:>8.2} us ({:.2}x speedup) | rel_err: {:.5}",
        bfp64_step_us,
        dense_step_us / bfp64_step_us,
        step_bfp_err
    );
    println!(
        "     - 100% Zero-Float p/q Engram Step (Q16):   {:>8.2} us ({:.2}x speedup) | rel_err: {:.5}",
        pq_no_floats_step_us,
        dense_step_us / pq_no_floats_step_us,
        step_pq_err
    );
    println!("=========================================================================================");
}
