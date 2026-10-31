from pathlib import Path

p = Path("/mnt/disk3/mIHC/mif_marker_positivity_20261022")
lines = (
    (p / "analyses/cd8_e1_e4_stability_20261023/audit/input_manifest.csv")
    .read_text()
    .splitlines()
)
print("\n".join(lines[2:5]))
print((p / "analyses/cd8_morphospatial_20261023/.venv/bin/python").exists())
