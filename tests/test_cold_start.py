"""Regression checks for cold-start handling in Steps 5.2 and 5.3."""

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from src.model.mmgcf import MMGCF
from src.model.late_fusion_mmgcf import LateFusionMMGCF
from src.model.train_eval import NegativeSampler


class ColdStartTests(unittest.TestCase):
    def test_isolated_item_keeps_base_embedding_after_propagation(self):
        adjacency = torch.sparse_coo_tensor(
            torch.tensor([[0, 1], [1, 0]]),
            torch.ones(2),
            (3, 3),
            check_invariants=True,
        ).coalesce()
        inputs = SimpleNamespace(
            num_users=1,
            num_items=2,
            normalized_adj=adjacency,
            node_degree=torch.tensor([1, 1, 0]),
            text_features=torch.ones(2, 4),
            image_features=torch.ones(2, 4),
        )
        model = MMGCF(inputs, embedding_dim=4, num_layers=2)
        initial = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        propagated = model._propagate(initial)
        torch.testing.assert_close(propagated[2], initial[2])
        output = model()
        self.assertEqual(output.view_weights[2, 0].item(), 0.0)
        self.assertTrue(torch.isfinite(output.item_embeddings).all())

    def test_cold_items_are_not_sampled_as_train_negatives(self):
        sampler = NegativeSampler(
            train_seen=[{0}, {1}],
            num_items=3,
            rng=np.random.default_rng(42),
            candidate_items=np.array([0, 1]),
        )
        samples = sampler.sample(np.array([0, 1] * 10), count=3)
        self.assertTrue(np.all(samples[0::2] == 1))
        self.assertTrue(np.all(samples[1::2] == 0))

    def test_late_fusion_keeps_cold_and_warm_item_norms_equal(self):
        adjacency = torch.sparse_coo_tensor(
            torch.tensor([[0, 1], [1, 0]]),
            torch.ones(2),
            (3, 3),
            check_invariants=True,
        ).coalesce()
        inputs = SimpleNamespace(
            num_users=1,
            num_items=2,
            normalized_adj=adjacency,
            node_degree=torch.tensor([1, 1, 0]),
            text_features=torch.randn(2, 4),
            image_features=torch.randn(2, 4),
        )
        model = LateFusionMMGCF(inputs, embedding_dim=4, num_layers=2)
        output = model()
        torch.testing.assert_close(output.item_embeddings.norm(dim=1), torch.ones(2))
        self.assertEqual(output.view_weights[2, 0].item(), 0.0)
        loss = model.bpr_loss(torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))
        loss.backward()
        self.assertIsNotNone(model.text_projection.weight.grad)


if __name__ == "__main__":
    unittest.main()
