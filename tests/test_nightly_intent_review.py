from __future__ import annotations

import unittest

from gnaniassignment.pipeline.nightly_intent_review import ClassifiedTurn, _telegram_chunks


class NightlyIntentReviewTests(unittest.TestCase):
    def test_digest_contains_candidate_labels_and_turn_ids(self) -> None:
        messages = _telegram_chunks(
            [
                ClassifiedTurn(12, "Where is my order?", "track_order", 0.91),
                ClassifiedTurn(13, "Cancel it please", "cancel_order", 0.84),
            ],
            max_items=20,
        )
        self.assertEqual(len(messages), 1)
        self.assertIn("#12  track_order (91.0%)", messages[0])
        self.assertIn("#13  cancel_order (84.0%)", messages[0])
        self.assertIn("candidate labels only", messages[0])


if __name__ == "__main__":
    unittest.main()
