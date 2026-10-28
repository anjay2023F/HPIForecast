# FHFA data artifacts

Keep `raw/`, `processed/`, and model `outputs/` out of Git: these are generated
artifacts. Commit the downloader, preparation code, tests, and
`source_manifest.json` instead. The manifest records the verified raw snapshot
hash and selection settings; it does not pin the contents of FHFA's live URL.

Rebuild the data with:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_data.py
```

Then run and audit the full default pipeline:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py
.\.venv\Scripts\python.exe scripts/verify_pipeline.py
```

FHFA revises historical data. For a frozen paper, course submission, or release,
archive the exact raw CSV and generated metadata alongside the release or in
artifact storage. Confirm the SHA-256 matches `source_manifest.json` to reproduce
this snapshot. A new download may legitimately produce different values and
results. Do not update the manifest silently when refreshing the data.
