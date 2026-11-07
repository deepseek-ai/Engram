import math
import random
import unittest

import engram_demo_v1 as demo
import engram_pq as pq


class EngramSovereignTest(unittest.TestCase):
    def setUp(self) -> None:
        demo.clear_hash_mapping_cache()

    def test_primality_and_distinct_prime_buckets(self) -> None:
        known_primes = [2, 3, 5, 7, 11, 1009, 10007, 129289, 646403]
        for p in known_primes:
            self.assertTrue(pq.is_prime_u64(p), f"Expected {p} to be prime")
        for comp in [0, 1, 4, 9, 15, 1000, 129280, 646400]:
            self.assertFalse(pq.is_prime_u64(comp), f"Expected {comp} to be composite")

        mapping = pq.NgramHashMappingPq(
            engram_vocab_size=[100, 120],
            max_ngram_size=3,
            n_embed_per_ngram=64,
            n_head_per_ngram=4,
            layer_ids=[1, 15],
            tokenizer_name_or_path="offline",
            pad_id=2,
            seed=0,
            fallback_vocab_size=512,
        )
        all_primes = [
            prime
            for layer_buckets in mapping.vocab_size_across_layers.values()
            for ngram_buckets in layer_buckets
            for prime in ngram_buckets
        ]
        self.assertEqual(len(all_primes), len(set(all_primes)))
        self.assertTrue(all(pq.is_prime_u64(p) for p in all_primes))

    def test_lemire_pq_reciprocal_is_bit_exact_with_modulo(self) -> None:
        primes = [101, 1009, 10007, 65537, 646403, 1292809]
        rng = random.Random(42)
        for p in primes:
            bucket = pq.PrimeBucketMod.create(p, table_offset=0)
            for _ in range(500):
                x = rng.getrandbits(32)
                self.assertEqual(
                    bucket.fast_rem_u64(x),
                    x % p,
                    f"Mismatch for x={x}, p={p}",
                )

        mapping = pq.NgramHashMappingPq(
            engram_vocab_size=[500, 700],
            max_ngram_size=3,
            n_embed_per_ngram=512,
            n_head_per_ngram=8,
            layer_ids=[1, 15],
            tokenizer_name_or_path="offline",
            pad_id=2,
            seed=7,
            fallback_vocab_size=512,
        )
        sample_ids = [[0, 15, 260, 261, 300, 412, 19, 2]]
        hashes_mod = mapping.hash(sample_ids, use_pq_hash=False, fold_32bit=True)
        hashes_pq = mapping.hash(sample_ids, use_pq_hash=True, fold_32bit=True)
        self.assertEqual(hashes_mod, hashes_pq)

    def test_compressed_tokenizer_bounds_and_canonical_merge(self) -> None:
        comp = pq.CompressedTokenizerPq("offline", fallback_vocab_size=1536)
        # In SovereignOfflineTokenizer, 260 -> "A000" and 260 + 512 = 772 -> "a000"
        self.assertEqual(comp.lookup_table[260], comp.lookup_table[772])

        # Bounds check (PR #15): negative IDs and out-of-range IDs are preserved safely
        out = comp([[-1, -100, 260, 772, 999999]])
        self.assertEqual(out[0][0], -1)
        self.assertEqual(out[0][1], -100)
        self.assertEqual(out[0][2], out[0][3])
        self.assertEqual(out[0][4], 999999)

    def test_hash_mapping_cache_reused_across_layers(self) -> None:
        e1 = pq.EngramPq(layer_id=1, engram_vocab_size=(256, 256), layer_ids=(1, 15))
        e15 = pq.EngramPq(layer_id=15, engram_vocab_size=(256, 256), layer_ids=(1, 15))
        self.assertIs(e1.hash_mapping, e15.hash_mapping)

    def test_bfp64_and_mem8_adaptive_embedding_compression_and_accuracy(self) -> None:
        emb_bfp = pq.Bfp64MultiHeadEmbeddingPq([128, 128, 128, 128], D=64, seed=11)
        self.assertAlmostEqual(emb_bfp.compression_ratio_vs_fp32, 256.0 / 65.0, places=5)

        rng = random.Random(99)
        vec = [(rng.random() - 0.5) * 2.0 for _ in range(64)]
        row8 = pq.Bfp64HeadRow.from_floats(vec)
        recon8 = row8.dequantize()
        norm_v = math.sqrt(sum(v * v for v in vec))
        err8 = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec, recon8))) / norm_v
        self.assertLess(err8, 0.01)

        # Force promotion in MEM|8 Adaptive with strict tau
        row_prom = pq.Mem8AdaptiveHeadRow.from_floats(vec, tau=0.0001)
        self.assertTrue(row_prom.promoted_i16)
        recon16 = row_prom.dequantize()
        err16 = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec, recon16))) / norm_v
        self.assertLess(err16, err8 / 50.0)

    def test_quantum_64trit_and_integer_aligned_projection(self) -> None:
        proj = pq.TernaryLinearPq(in_features=256, out_features=64, seed=23)
        rng = random.Random(23)
        m_x = [rng.randint(-127, 127) for _ in range(256)]
        exp_x = [-8, -7, -9, -6]
        out = proj.forward_prequant_bfp64(m_x, exp_x)
        self.assertEqual(len(out), 64)

        # Compare against explicit float accumulation
        for j in range(64):
            expected = 0.0
            for g in range(4):
                blk_m = m_x[g * 64:(g + 1) * 64]
                dot_i16 = proj.blocks[j][g].dot_i16(blk_m)
                scale_g = proj.scales_pq[j][g] / float(pq.PQ_SCALE_Q20)
                expected += dot_i16 * math.ldexp(1.0, exp_x[g]) * scale_g
            self.assertAlmostEqual(out[j], expected, places=9)

    def test_fused_rmsnorm_gate_and_engram_forward(self) -> None:
        rng = random.Random(77)
        D = 128
        k = [rng.uniform(-1.0, 1.0) for _ in range(D)]
        q = [rng.uniform(-1.0, 1.0) for _ in range(D)]
        wk = [rng.uniform(0.5, 1.5) for _ in range(D)]
        wq = [rng.uniform(0.5, 1.5) for _ in range(D)]
        eps = 1e-5

        rms_k = math.sqrt(sum(x * x for x in k) / D + eps)
        rms_q = math.sqrt(sum(x * x for x in q) / D + eps)
        nk = [(x / rms_k) * w for x, w in zip(k, wk)]
        nq = [(x / rms_q) * w for x, w in zip(q, wq)]
        dot = sum(a * b for a, b in zip(nk, nq)) / math.sqrt(D)
        signed_sqrt = math.copysign(math.sqrt(max(abs(dot), 1e-6)), dot)
        expected_gate = 1.0 / (1.0 + math.exp(-signed_sqrt))

        actual_gate = pq.fused_rmsnorm_gate(k, q, wk, wq, eps=eps)
        self.assertAlmostEqual(actual_gate, expected_gate, places=12)

        layer = pq.EngramPq(
            layer_id=1,
            engram_vocab_size=(128, 128),
            max_ngram_size=3,
            n_embed_per_ngram=128,
            n_head_per_ngram=2,
            layer_ids=(1,),
            tokenizer_name_or_path="offline",
            hidden_size=64,
            hc_mult=4,
            embedding_mode="bfp64",
        )
        hidden = [[[[0.1 * ((t + hc + d) % 5 - 2) for d in range(64)] for hc in range(4)] for t in range(6)]]
        out = layer.forward(hidden, [[10, 20, 30, 40, 50, 60]])
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), 6)
        self.assertEqual(len(out[0][0]), 4)
        self.assertEqual(len(out[0][0][0]), 64)

    def test_numpy_vectorized_fast_rem_matches_scalar(self) -> None:
        if not demo.HAS_TORCH:
            self.skipTest("torch not installed")
        import numpy as np

        rng = random.Random(1234)
        for p in [101, 1009, 10007, 65537, 646403, 1292809]:
            bucket = pq.PrimeBucketMod.create(p, table_offset=0)
            xs = np.array([rng.getrandbits(32) for _ in range(4096)], dtype=np.uint64)
            got = demo._fast_rem_u64_vec(xs, bucket)
            want = np.array([bucket.fast_rem_u64(int(v)) for v in xs], dtype=np.int64)
            np.testing.assert_array_equal(got, want)

    def test_torch_embedding_modes_dispatch(self) -> None:
        if not demo.HAS_TORCH or not hasattr(demo.nn, "RMSNorm"):
            self.skipTest("torch >= 2.4 required (nn.RMSNorm)")

        def build(mode: str) -> demo.Engram:
            cfg = demo.EngramConfig(
                engram_vocab_size=[100, 120],
                n_embed_per_ngram=64,
                n_head_per_ngram=4,
                layer_ids=[1],
                tokenizer_name_or_path="offline",
                embedding_mode=mode,
            )
            b_cfg = demo.BackBoneConfig(hidden_size=32, hc_mult=2)
            return demo.Engram(layer_id=1, config=cfg, backbone_cfg=b_cfg)

        self.assertIsInstance(build("dense").multi_head_embedding, demo.MultiHeadEmbedding)
        self.assertIsInstance(
            build("bfp64").multi_head_embedding, demo.TorchBfp64MultiHeadEmbedding
        )
        mem8 = build("mem8_adaptive").multi_head_embedding
        self.assertIsInstance(mem8, demo.TorchMem8AdaptiveMultiHeadEmbedding)
        self.assertGreaterEqual(mem8.promoted_count, 0)
        self.assertEqual(mem8.total_N, mem8.mantissas_i8.shape[0])
        with self.assertRaises(ValueError):
            build("bogus_mode")

    def test_engram_pq_fold_32bit_parameter(self) -> None:
        base = dict(
            engram_vocab_size=(256, 256),
            max_ngram_size=3,
            n_embed_per_ngram=512,
            n_head_per_ngram=8,
            layer_ids=(1,),
            tokenizer_name_or_path="offline",
            pad_id=2,
            seed=3,
            hidden_size=64,
            hc_mult=2,
        )
        folded = pq.EngramPq(layer_id=1, fold_32bit=True, **base)
        unfolded = pq.EngramPq(layer_id=1, fold_32bit=False, **base)
        self.assertTrue(folded.fold_32bit)
        self.assertFalse(unfolded.fold_32bit)

        ids = [[0, 15, 260, 261, 300, 412, 19, 2]]
        hidden = [[[[0.1 for _ in range(64)] for _ in range(2)] for _ in range(8)]]
        out_folded = folded.forward(hidden, ids)
        out_unfolded = unfolded.forward(hidden, ids)
        self.assertNotEqual(out_folded, out_unfolded)


if __name__ == "__main__":
    unittest.main()
