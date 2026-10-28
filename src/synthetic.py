"""Synthetic data in the FHFA Master-HPI layout. ONLY for testing that the pipeline runs.
Results on this data say nothing about real housing markets."""
import os

import numpy as np
import pandas as pd

STATES = ("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM "
          "NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY").split()


def make_synthetic(path: str, seed: int = 0, start: int = 1975, end: int = 2025) -> str:
    rng = np.random.default_rng(seed)
    T = (end - start + 1) * 4
    mu = {0: 1.1, 1: 2.3, 2: -1.4}                      # steady, boom, bust (% per quarter)
    P = np.array([[0.965, 0.02, 0.015], [0.08, 0.89, 0.03], [0.12, 0.0, 0.88]])
    state, g = 0, np.zeros(T)
    for i in range(1, T):
        state = rng.choice(3, p=P[state])
        g[i] = 0.85 * g[i - 1] + 0.15 * mu[state] + rng.normal(0, 0.3)

    rows = []
    for s in STATES:
        beta, off = rng.uniform(0.6, 1.5), rng.normal(0, 0.15)
        idio = np.zeros(T)
        for i in range(1, T):
            idio[i] = 0.7 * idio[i - 1] + rng.normal(0, 0.5)
        idx = 100 * np.exp(np.cumsum(beta * g + idio + off) / 100)
        for i in range(T):
            rows.append(("traditional", "all-transactions", "quarterly", "State", s, s,
                         start + i // 4, i % 4 + 1, idx[i], np.nan))
    df = pd.DataFrame(rows, columns=["hpi_type", "hpi_flavor", "frequency", "level", "place_name",
                                     "place_id", "yr", "period", "index_nsa", "index_sa"])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    return path
