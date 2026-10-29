"""Recommendation ranking must never return a user's known listings."""

from types import SimpleNamespace
import unittest

import pandas as pd
import torch

from src.model.recommend import recommend_user, resolve_user_idx


class DummyModel:
    cold_item_mask = torch.tensor([False, True, False])

    def __call__(self):
        return None

    def score_all_items(self, user_idx, output=None):
        # The two highest scores belong to already observed items.
        return torch.tensor([[1.0, 8.0, 10.0]])


class RecommendationTests(unittest.TestCase):
    def test_unseen_listing_is_returned_even_when_seen_scores_are_higher(self):
        data = SimpleNamespace(
            num_users=1,
            num_items=3,
            device=torch.device("cpu"),
            user_mapping=pd.DataFrame({"user_idx": [0], "reviewer_id": ["reviewer-1"]}),
            item_mapping=pd.DataFrame(
                {"item_idx": [0, 1, 2], "listing_id": ["listing-0", "listing-1", "listing-2"]}
            ),
            train_edges=pd.DataFrame({"user_idx": [0], "item_idx": [2]}),
            val_edges=pd.DataFrame({"user_idx": [0], "item_idx": [1]}),
            test_edges=pd.DataFrame({"user_idx": pd.Series(dtype="int64"), "item_idx": pd.Series(dtype="int64")}),
            paths={"train_interactions": "data/processed/example/graph/train_interactions.parquet"},
        )
        self.assertEqual(resolve_user_idx(data, "reviewer-1", None), 0)
        result = recommend_user(DummyModel(), data, 0, top_k=10)
        self.assertEqual(result["item_idx"].tolist(), [0])
        self.assertEqual(result["listing_id"].tolist(), ["listing-0"])


if __name__ == "__main__":
    unittest.main()
