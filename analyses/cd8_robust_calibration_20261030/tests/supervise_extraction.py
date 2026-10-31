"""Observe extraction resources without changing frozen analysis code or settings."""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def descendants(pid):
    ids = [pid]
    children = Path("/proc/%s/task/%s/children" % (pid, pid))
    try:
        for child in children.read_text().split():
            ids.extend(descendants(int(child)))
    except (FileNotFoundError, ProcessLookupError):
        pass
    return ids


def memory(pid):
    try:
        values = {
            line.split(":", 1)[0]: line.split(":", 1)[1].strip()
            for line in Path("/proc/%s/status" % pid).read_text().splitlines()
            if ":" in line
        }
        return tuple(
            int(values.get(k, "0 kB").split()[0]) * 1024 for k in ["VmRSS", "VmHWM"]
        )
    except (FileNotFoundError, ProcessLookupError):
        return (0, 0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--run-id", default="001")
    a = p.parse_args()
    for stage in ["extract", "merge"]:
        stem = stage + "_" + a.run_id
        log = a.run_dir / "logs" / (stem + ".log")
        csv_path = a.run_dir / "logs" / (stem + "_resource_samples.csv")
        summary = a.run_dir / "logs" / (stem + "_resources.json")
        if any(x.exists() for x in [log, csv_path, summary]):
            raise FileExistsError("Resource run already exists: " + stem)
        command = [
            sys.executable,
            "-u",
            str(a.run_dir / "scripts/run_features.py"),
            "--run-dir",
            str(a.run_dir),
            "--stage",
            stage,
            "--workers",
            str(a.workers),
        ]
        start = time.monotonic()
        peak_rss = peak_hwm = 0
        with log.open("w") as out, csv_path.open("w", newline="") as raw:
            writer = csv.DictWriter(
                raw,
                fieldnames=[
                    "elapsed_seconds",
                    "processes",
                    "sum_rss_bytes",
                    "max_process_hwm_bytes",
                ],
            )
            writer.writeheader()
            process = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT)
            print(
                json.dumps(dict(stage=stage, pid=process.pid, command=command)),
                flush=True,
            )
            while True:
                ids = descendants(process.pid)
                stats = [memory(pid) for pid in ids]
                rss = sum(x[0] for x in stats)
                hwm = max(x[1] for x in stats)
                peak_rss = max(peak_rss, rss)
                peak_hwm = max(peak_hwm, hwm)
                writer.writerow(
                    dict(
                        elapsed_seconds=round(time.monotonic() - start, 3),
                        processes=len(ids),
                        sum_rss_bytes=rss,
                        max_process_hwm_bytes=hwm,
                    )
                )
                raw.flush()
                if process.poll() is not None:
                    break
                time.sleep(5)
        result = dict(
            stage=stage,
            returncode=process.returncode,
            wall_seconds=time.monotonic() - start,
            sampled_peak_sum_rss_bytes=peak_rss,
            max_observed_process_hwm_bytes=peak_hwm,
            sample_interval_seconds=5,
            memory_definition="sum of parent/worker RSS, including shared/mmap pages counted in each process; sampled upper-bound proxy for resident usage, not unique physical memory",
            includes_input_hash_audit=True,
            command=command,
            workers=a.workers,
            supervisor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            completed_utc=datetime.datetime.utcnow().isoformat() + "Z",
        )
        summary.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if process.returncode:
            raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
