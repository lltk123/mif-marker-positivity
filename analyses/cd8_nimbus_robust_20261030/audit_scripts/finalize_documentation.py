"""Correct review-file scope and add completion links without changing numerical outputs."""
from pathlib import Path
import datetime
import hashlib
import json

R = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main():
    marker = R / "provenance/documentation_finalization.json"
    complete_path = R / "provenance/delivery_complete.json"
    report_path = R / "ANALYSIS_REPORT.md"
    complete = json.loads(complete_path.read_text())
    assert complete["status"] in ["complete", "complete_with_unscorable_cells"]
    assert sha(report_path) == complete["report_sha256"]
    if marker.exists():
        old = json.loads(marker.read_text())
        assert old["script_sha256"] == sha(Path(__file__))
        if old["report_sha256_after"] == sha(report_path):
            print("VERIFIED_DOCUMENTATION_FINALIZATION")
            return
        assert old["report_sha256_before"] == sha(
            report_path
        ), "report changed beyond the known generator output"
    integrity = json.loads(
        (R / "provenance/prediction_integrity_passed.json").read_text()
    )
    for name in ["local_real_crop_numeric_checks.csv", "offline_numeric_checks.csv"]:
        assert (R / "tests" / name).is_file(), name
    old_text = report_path.read_text(encoding="utf-8")
    old_sentence = "tests/offline_numeric_checks.csv 的人工bounds crop检查仅验证算术，不代表完整单位校准。"
    assert old_sentence in old_text
    new_sentence = (
        "tests/local_real_crop_numeric_checks.csv 的人工bounds真实crop检查仅验证局部算术，"
        "不代表完整单位校准；tests/offline_numeric_checks.csv 为服务器受控数值检查。"
    )
    updated = old_text.replace(old_sentence, new_sentence)
    updated += (
        "\n## 审计补充\n\n"
        "预推理实现修订及兼容缓存核验见 REPRODUCIBILITY_NOTES.md、"
        "provenance/IMPLEMENTATION_FIX_001.json 和 provenance/CACHE_COMPATIBILITY_AUDIT.json。"
        "校准覆盖率单独见 results/calibration_coverage.csv，按 split、level、context、image_id "
        "筛选 n_cells、n_calibration_invalid、calibration_coverage；不以模型有限评分代替有效输入。\n\n"
        "runtime_summary.csv 的 inference_seconds 是各完整单位 infer_one 的总耗时，"
        "包括缓存读取、模型推理、完整实例聚合及文件写入；fit_seconds 包括优化、验证、初始化及检查点保存。"
        "两者不是纯GPU算子计时，也不构成隔离机器条件下的速度基准。\n"
    )
    before = sha(report_path)
    backup = R / "provenance/ANALYSIS_REPORT.before_documentation_finalization.md"
    if backup.exists():
        assert sha(backup) == before
    else:
        backup.write_text(old_text, encoding="utf-8")
    report_path.write_text(updated, encoding="utf-8")
    readme = R / "README.md"
    readme_old = readme.read_text(encoding="utf-8")
    readme_backup = R / "provenance/README.before_completion_note.md"
    if not readme_backup.exists():
        readme_backup.write_text(readme_old, encoding="utf-8")
    completion_note = (
        f"\n本次全量交付已完成：两个模型各 {integrity['total_cells_per_model']:,} 个细胞、"
        f"{integrity['units']} 个校准单位；不可评分记录合计 {integrity['total_unscorable']}。"
        "实际性能见 [ANALYSIS_REPORT.md](ANALYSIS_REPORT.md)，"
        "验证范围和修订记录见 [REPRODUCIBILITY_NOTES.md](REPRODUCIBILITY_NOTES.md)。\n"
        "完整数值流程结束后，运行 `python audit_scripts/calibration_coverage.py` 和 "
        "`python audit_scripts/finalize_documentation.py`，生成覆盖率审计与最终交付说明。\n"
    )
    title, rest = readme_old.split("\n", 1)
    if completion_note.strip() not in readme_old:
        readme.write_text(title + "\n" + completion_note + rest, encoding="utf-8")
    complete["report_sha256"] = sha(report_path)
    complete[
        "documentation_finalization"
    ] = "provenance/documentation_finalization.json"
    save(complete_path, complete)
    save(
        marker,
        dict(
            status="complete",
            created=datetime.datetime.now().isoformat(),
            script_sha256=sha(Path(__file__)),
            report_sha256_before=before,
            report_sha256_after=sha(report_path),
            readme_sha256_after=sha(readme),
            scope="Documentation only: correct local versus server test paths, explain timing scope, add completion links. No numerical table, prediction, weight, protocol or computational script changed.",
        ),
    )
    print("DOCUMENTATION_FINALIZED")


if __name__ == "__main__":
    main()
