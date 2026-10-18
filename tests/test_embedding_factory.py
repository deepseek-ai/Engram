"""Offline full-Engram checks; no tokenizer/checkpoint download.

Run: python -m unittest discover -s tests
Dependencies are the same as engram_demo_v1.py.
"""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import engram_demo_v1 as demo

ORIGINAL_HASHER = demo.NgramHashMapping


class LocalHashMapping:
    """Tiny fixed-ID fixture replacing tokenizer construction, not hash logic."""
    def __init__(self, **kwargs):
        self.max_ngram_size = kwargs["max_ngram_size"]
        self.n_head_per_ngram = kwargs["n_head_per_ngram"]
        self.pad_id = 0
        self.layer_multipliers = {1: np.array([17, 19, 23], dtype=np.int64)}
        self.vocab_size_across_layers = {1: [[11, 13], [17, 19]]}

    def hash(self, input_ids):
        return {1: ORIGINAL_HASHER._get_ngram_hashes(self, input_ids, 1)}


class GatherBackend(nn.Module):
    """A parameter-free frozen gather that satisfies the factory contract."""
    def __init__(self, source):
        super().__init__()
        self.register_buffer("rows", source.embedding.weight.detach().clone())
        self.register_buffer("offsets", source.offsets.clone())
        self.calls = 0

    def forward(self, ids):
        self.calls += 1
        return torch.nn.functional.embedding(ids + self.offsets, self.rows)


class EmbeddingFactoryTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = demo.EngramConfig(engram_vocab_size=[10, 16], max_ngram_size=3,
                                       n_embed_per_ngram=8, n_head_per_ngram=2,
                                       layer_ids=[1], pad_id=0, kernel_size=3)
        self.backbone = demo.BackBoneConfig(hidden_size=8, hc_mult=2, vocab_size=32, num_layers=2)
        self.patches = [patch.object(demo, "engram_cfg", self.config),
                        patch.object(demo, "backbone_config", self.backbone),
                        patch.object(demo, "NgramHashMapping", LocalHashMapping)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_default_and_explicit_original_factory_are_identical(self):
        torch.manual_seed(3)
        default = demo.Engram(1)
        torch.manual_seed(3)
        explicit = demo.Engram(1, embedding_factory=demo.MultiHeadEmbedding)
        self.assertEqual(set(default.state_dict()), set(explicit.state_dict()))
        for key, value in default.state_dict().items():
            self.assertTrue(torch.equal(value, explicit.state_dict()[key]))
        self.assertTrue(default.multi_head_embedding.embedding.weight.requires_grad)

    def test_injected_backend_matches_full_gated_convolution_output(self):
        baseline = demo.Engram(1).eval()
        backend = GatherBackend(baseline.multi_head_embedding)
        calls = []
        def factory(list_of_N, D):
            calls.append((list_of_N, D))
            return backend
        # Construction must never allocate the default table for this path.
        with patch.object(demo, "MultiHeadEmbedding", side_effect=AssertionError("default allocation")):
            mapped = demo.Engram(1, embedding_factory=factory).eval()
        downstream = {k: v for k, v in baseline.state_dict().items() if not k.startswith("multi_head_embedding.")}
        status = mapped.load_state_dict(downstream, strict=False)
        self.assertFalse(status.unexpected_keys)
        self.assertTrue(all(k.startswith("multi_head_embedding.") for k in status.missing_keys))
        self.assertEqual(calls, [([11, 13, 17, 19], 4)])
        ids = np.array([[0, 1, 2, 3, 0], [3, 3, 2, 1, 0]], dtype=np.int64)
        hidden = torch.randn(2, 5, 2, 8)
        with torch.inference_mode():
            before, after = baseline(hidden, ids), mapped(hidden, ids)
        self.assertTrue(torch.equal(before.view(torch.uint8), after.view(torch.uint8)))
        self.assertEqual(backend.calls, 1)

    def test_factory_must_return_module(self):
        with self.assertRaisesRegex(TypeError, "nn.Module"):
            demo.Engram(1, embedding_factory=lambda **kwargs: object())


if __name__ == "__main__":
    unittest.main()
