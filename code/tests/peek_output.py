"""Spot-check rows of the written output.csv (structural verification only)."""

import csv
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

path = Path(r"D:\LocalProjects\hackerrank-orchestrate-august26\dataset\output.csv")
with path.open("r", encoding="utf-8", newline="") as fh:
    rows = list(csv.DictReader(fh))

print(f"total rows: {len(rows)}\n")
for r in rows[:8]:
    print(
        f"{r['message_id']:<9} {r['action']:<8} {r['message_type']:<15} "
        f"conf={r['confidence']:<6} ev={r['evidence_message_ids'][:26]:<26} "
        f"{r['reason'][:55]}"
    )
