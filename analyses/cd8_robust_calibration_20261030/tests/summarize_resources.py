"""Convert completed resource logs to a small, explicitly defined CSV table."""
from pathlib import Path
import argparse, json, os
import pandas as pd

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--run-dir", type=Path, required=True)
a = p.parse_args()
r = a.run_dir
manifest = json.loads((r / "extraction_manifest.json").read_text())
assert manifest["status"] == "complete"
info = dict(
    line.split(":", 1)
    for line in Path("/proc/meminfo").read_text().splitlines()
    if ":" in line
)
cpu = next(
    line.split(":", 1)[1].strip()
    for line in Path("/proc/cpuinfo").read_text().splitlines()
    if line.startswith("model name")
)
rows = []
for stage in ["extract", "merge"]:
    source = r / "logs" / (stage + "_001_resources.json")
    record = json.loads(source.read_text())
    assert record["returncode"] == 0
    rows.append(
        dict(
            stage=stage,
            rows=manifest["rows"],
            wall_seconds=record["wall_seconds"],
            wall_minutes=record["wall_seconds"] / 60,
            includes_input_hash_audit=record["includes_input_hash_audit"],
            configured_workers=record["workers"],
            parallel_feature_workers=record["workers"] if stage == "extract" else 0,
            max_observed_process_hwm_GiB=record["max_observed_process_hwm_bytes"]
            / 2**30,
            sampled_peak_sum_rss_GiB=record["sampled_peak_sum_rss_bytes"] / 2**30,
            rss_sum_note="Shared mmap pages counted repeatedly; not unique physical RAM consumed",
            sample_interval_seconds=record["sample_interval_seconds"],
            cpu_model=cpu,
            cpu_logical_cores=os.cpu_count(),
            host_total_memory_GiB=int(info["MemTotal"].split()[0]) / 1024**2,
            source_json=str(source.relative_to(r)),
        )
    )
output = r / "RUN_RESOURCES.csv"
assert not output.exists()
pd.DataFrame(rows).to_csv(output, index=False)
print(
    pd.DataFrame(rows)[
        [
            "stage",
            "wall_minutes",
            "max_observed_process_hwm_GiB",
            "sampled_peak_sum_rss_GiB",
        ]
    ].to_string(index=False)
)
