"""Download the FHFA Master HPI CSV. Run this on your own machine.

If the default URL ever changes, copy the "Data [CSV]" link under "Master HPI Data" at
https://www.fhfa.gov/data/hpi/datasets and pass it with --url, or just save the file manually to
data/raw/hpi_master.csv.
"""
import argparse
import os
import urllib.request

DEFAULT_URL = "https://www.fhfa.gov/hpi/download/monthly/hpi_master.csv"  # verify against the datasets page

ap = argparse.ArgumentParser()
ap.add_argument("--url", default=DEFAULT_URL)
ap.add_argument("--out", default="data/raw/hpi_master.csv")
args = ap.parse_args()

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
req = urllib.request.Request(args.url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=120) as r, open(args.out, "wb") as f:
    f.write(r.read())
print(f"saved {args.out} ({os.path.getsize(args.out):,} bytes)")
