"""Checks for time leakage, missing quarters, and invalid raw observations."""
import unittest

import numpy as np
import pandas as pd

from src.config import Config
from src.data import clean
from src.features import FEATURES, build_dataset
from src.prepare import chronological_split
from src.evaluate import add_regimes


class PreparationTests(unittest.TestCase):
    def test_regime_boundaries_match_documented_definitions(self):
        rows = pd.DataFrame({"t": range(5), "actual": [-0.1, 0, 3, 7, 7.1]})
        result = add_regimes(rows, Config())
        self.assertEqual(result.regime.tolist(),
                         ["decline", "steady", "steady", "steady", "boom"])

    def test_holdout_purges_targets_at_and_after_cutoff(self):
        cfg = Config(first_test_year=2020)
        rows = pd.DataFrame({"t": range(8068, 8088)})
        train, test, purged = chronological_split(rows, cfg)
        self.assertTrue((train.t + cfg.horizon_q < 8080).all())
        self.assertTrue((test.t >= 8080).all())
        self.assertEqual(purged.t.tolist(), [8076, 8077, 8078, 8079])
        self.assertEqual(len(train) + len(test) + len(purged), len(rows))

    def test_clean_rejects_invalid_rows_and_conflicting_duplicates(self):
        raw = pd.DataFrame({
            "place_id": ["AA"] * 7, "place_name": ["A"] * 7,
            "yr": [2000, 2000, 2000, 2000.5, 2000, 2000, 2000],
            "period": [1, 1, 2, 3, 3, 4, 2.5],
            "index_nsa": [100, 100, np.inf, 103, -1, 104, 105],
        })
        cfg = Config(min_years=0)
        panel = clean(raw, cfg)
        self.assertEqual(panel.q.tolist(), [1, 4])
        conflict = pd.concat([raw, raw.iloc[[0]].assign(index_nsa=999)])
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            clean(conflict, cfg)

    def test_features_do_not_change_when_future_prices_change(self):
        t = np.arange(8000, 8040)
        panel = pd.DataFrame({"place_id": "AA", "place_name": "A", "t": t,
                              "index": 100 * 1.01 ** np.arange(len(t))})
        original, _ = build_dataset(panel, Config())
        changed_panel = panel.copy()
        changed_panel.loc[changed_panel.t > 8030, "index"] *= 2
        changed, _ = build_dataset(changed_panel, Config())
        pd.testing.assert_frame_equal(original.loc[original.t <= 8030, FEATURES],
                                      changed.loc[changed.t <= 8030, FEATURES])
        gap_panel = panel[panel.t != 8028]
        _, labeled = build_dataset(gap_panel, Config())
        self.assertNotIn(8024, labeled.t.tolist())  # Missing t+4 cannot supply a target.
        self.assertNotIn(8029, labeled.t.tolist())  # Missing last quarter cannot supply features.


if __name__ == "__main__":
    unittest.main()
