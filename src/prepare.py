"""Save cleaned FHFA observations and a purged chronological holdout."""
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from .config import Config
from .data import clean, load_raw
from .features import FEATURES, build_dataset


def chronological_split(labeled, cfg):
    """Hold out all origins from first_test_year; purge overlapping targets."""
    cutoff = cfg.first_test_year * 4
    train = labeled[labeled["t"] + cfg.horizon_q < cutoff].copy()
    test = labeled[labeled["t"] >= cutoff].copy()
    purged = labeled[(labeled["t"] < cutoff) &
                     (labeled["t"] + cfg.horizon_q >= cutoff)].copy()
    if train.empty or test.empty:
        raise ValueError("Empty train/test split; choose a cutoff within the labeled history.")
    return train, test, purged


def prepare(cfg: Config, out_dir="data/processed"):
    panel = clean(load_raw(cfg.raw_path), cfg)
    all_rows, labeled = build_dataset(panel, cfg)
    train, test, purged = chronological_split(labeled, cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in [("cleaned_panel", panel), ("features", all_rows),
                        ("train", train), ("test", test), ("purged", purged)]:
        frame.to_csv(out / f"{name}.csv", index=False)
    metadata = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_sha256": hashlib.sha256(Path(cfg.raw_path).read_bytes()).hexdigest(),
        "source": "https://www.fhfa.gov/hpi/download/monthly/hpi_master.csv",
        "raw_path": str(cfg.raw_path), "level": cfg.level,
        "hpi_flavor": cfg.hpi_flavor, "index_col": cfg.index_col,
        "test_start": f"{cfg.first_test_year}Q1", "horizon_quarters": cfg.horizon_q,
        "features": FEATURES, "target": "next-horizon HPI percent change",
        "rows": {"panel": len(panel), "train": len(train), "test": len(test),
                 "purged": len(purged)},
        "origin_ranges": {
            name: {"start": f"{int(frame.t.min()) // 4}Q{int(frame.t.min()) % 4 + 1}",
                   "end": f"{int(frame.t.max()) // 4}Q{int(frame.t.max()) % 4 + 1}"}
            for name, frame in [("train", train), ("test", test)]
        },
        "scaling": "Raw features; fit scalers on training data only (see models.py).",
        "missing": "No interpolation; incomplete feature/target rows excluded.",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[split] train={len(train):,}, test={len(test):,}, purged={len(purged):,}; {out}")
    return panel, all_rows, labeled
