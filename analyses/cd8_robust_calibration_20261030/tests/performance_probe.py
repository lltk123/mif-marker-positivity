from pathlib import Path
import json, time

r = Path(
    "/mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_robust_calibration_20261030"
)
items = [
    (json.loads(p.read_text()), p.stat().st_mtime)
    for p in (r / "shards").glob("*.json")
]
if items:
    rows = sum(x["rows"] for x, t in items)
    start = min(t - x["seconds"] for x, t in items)
    end = max(t for x, t in items)
    rate = rows / (end - start)
    print(
        json.dumps(
            dict(
                completed_shards=len(items),
                completed_cells=rows,
                elapsed_extraction_seconds=end - start,
                observed_cells_per_second=rate,
                rough_remaining_minutes=(1817783 - rows) / rate / 60,
                mean_worker_seconds_per_2000=sum(x["seconds"] for x, t in items)
                / len(items),
            ),
            indent=2,
        )
    )
for line in Path("/proc/meminfo").read_text().splitlines():
    if line.startswith("MemAvailable:"):
        print(line)
