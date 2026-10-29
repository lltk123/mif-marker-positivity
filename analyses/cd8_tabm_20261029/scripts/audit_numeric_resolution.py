"""Read-only training-feature resolution audit for the frozen CD8 TabM pipeline.

This script never refits preprocessing or changes feature/model inputs. It writes
training-only diagnostic CSV/Markdown and an input-hash manifest to results/.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["dataset_id", "image_id", "cell_id", "marker"]
NOISE_STD = 1e-5


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def finite_summary(values, prefix):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    result = {
        prefix + "_n_finite": int(len(finite)),
        prefix + "_n_missing": int(np.isnan(values).sum()),
        prefix + "_n_infinite": int(np.isinf(values).sum()),
        prefix + "_nunique_train": int(len(np.unique(finite))),
    }
    for stat in ["min", "max", "mean", "std", "p01", "p10", "median", "p90", "p99"]:
        result[prefix + "_" + stat] = None
    if len(finite):
        # Use float64 accumulation so tiny Hu moment values are not rounded first.
        finite = finite.astype(np.float64, copy=False)
        quantiles = np.quantile(finite, [0.01, 0.10, 0.50, 0.90, 0.99])
        result.update(
            {
                prefix + "_min": float(finite.min()),
                prefix + "_max": float(finite.max()),
                prefix + "_mean": float(finite.mean()),
                prefix + "_std": float(finite.std(ddof=0)),
                **{
                    prefix + "_" + name: float(value)
                    for name, value in zip(
                        ["p01", "p10", "median", "p90", "p99"], quantiles
                    )
                },
            }
        )
    return result


def main(run_dir):
    root = Path(run_dir).resolve()
    paths = {
        "ready": root / "data/DATA_READY.json",
        "cells": root / "data/cells.parquet",
        "matrix": root / "data/X.npy",
        "indices": root / "data/train_indices.npy",
        "features": root / "FEATURE_SETS.json",
    }
    ready = read_json(paths["ready"])
    features = read_json(paths["features"])
    verified_hashes = {name: digest(path) for name, path in paths.items()}
    for name, expected in [
        ("cells", "cells_sha256"),
        ("matrix", "matrix_sha256"),
        ("indices", "train_indices_sha256"),
    ]:
        if verified_hashes[name] != ready[expected]:
            raise ValueError("Input hash mismatch: " + name)

    requested = features["C"]["requested"]
    selected = dict(zip(features["C"]["effective"], features["C"]["matrix_indices"]))
    if len(selected) != len(features["C"]["effective"]):
        raise ValueError("Duplicate effective feature name")
    cells = pd.read_parquet(
        paths["cells"], columns=KEYS + ["split", "ground_truth_binary"] + requested
    )
    if cells[KEYS].isna().any().any() or cells.duplicated(KEYS).any():
        raise ValueError("Missing or duplicated cell keys")
    if (
        not cells.dataset_id.eq("PhenoBIC-Data").all()
        or not cells.marker.eq("CD8").all()
    ):
        raise ValueError("Unexpected dataset or marker")
    train_indices = np.load(paths["indices"], allow_pickle=False)
    expected_indices = np.flatnonzero(
        cells.split.eq("training_WSIs").to_numpy()
        & cells.ground_truth_binary.isin([0, 1]).to_numpy()
    )
    if not np.array_equal(train_indices, expected_indices):
        raise ValueError(
            "Training index set/order differs from the full labelled split"
        )
    matrix = np.load(paths["matrix"], mmap_mode="r", allow_pickle=False)
    if matrix.shape != (len(cells), len(selected)) or matrix.dtype != np.float32:
        raise ValueError("Unexpected matrix shape or dtype")
    if sorted(selected.values()) != list(range(matrix.shape[1])):
        raise ValueError("Effective feature columns do not cover the matrix exactly")

    # Do not sort here: verified indices refer to the stored cells/matrix row order.
    training = cells.iloc[train_indices]
    train_key_hash = hashlib.sha256(
        training[KEYS].to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()
    rows = []
    for feature in requested:
        raw = training[feature].to_numpy(dtype=np.float64)
        raw_stats = finite_summary(raw, "raw")
        observed = raw[np.isfinite(raw)]
        # The imputed-unique denominator distinguishes true rounding from addition
        # of a median value that did not occur among the observed raw values.
        if len(observed):
            imputed = raw.copy()
            imputed[~np.isfinite(imputed)] = np.median(observed)
            imputed_unique = int(len(np.unique(imputed)))
        else:
            imputed_unique = 0
        row = {
            "feature": feature,
            "is_hu_moment": "humoment" in feature.lower().replace("_", ""),
            "requested_feature_sets": "|".join(
                name for name in "ABC" if feature in features[name]["requested"]
            ),
            "effective": feature in selected,
            "exclusion_reason": features["C"]["excluded"].get(feature, ""),
            "fit_split": "training_WSIs",
            "n_train_cells": len(training),
            "official_fit_noise_std": NOISE_STD,
            **raw_stats,
            "raw_imputed_nunique_train": imputed_unique,
            "noise_to_raw_std_ratio": (
                NOISE_STD / raw_stats["raw_std"]
                if raw_stats["raw_std"] is not None and raw_stats["raw_std"] > 0
                else None
            ),
            "unique_retention_ratio": None,
            "imputed_unique_retention_ratio": None,
            "transformed_float32_ulp_at_median": None,
            "resolution_note": "excluded_before_embedding"
            if feature not in selected
            else "",
        }
        if feature in selected:
            transformed = np.asarray(matrix[train_indices, selected[feature]])
            summary = finite_summary(transformed, "transformed_float32")
            row.update(summary)
            n_unique = summary["transformed_float32_nunique_train"]
            if row["raw_nunique_train"]:
                row["unique_retention_ratio"] = n_unique / row["raw_nunique_train"]
            if imputed_unique:
                row["imputed_unique_retention_ratio"] = n_unique / imputed_unique
            if summary["transformed_float32_median"] is not None:
                row["transformed_float32_ulp_at_median"] = float(
                    abs(np.spacing(np.float32(summary["transformed_float32_median"])))
                )
            # Descriptive audit flags only: never modify features or predictions.
            if n_unique < 2:
                row["resolution_note"] = "constant_after_transform"
            elif n_unique < 16 and imputed_unique >= 16:
                row["resolution_note"] = "fewer_than_16_transformed_values"
            elif imputed_unique and n_unique / imputed_unique < 0.1:
                row["resolution_note"] = "less_than_10_percent_unique_retained"
            else:
                row["resolution_note"] = "no_large_unique_loss_by_audit_rule"
        else:
            row.update(
                finite_summary(np.array([], dtype=np.float32), "transformed_float32")
            )
        rows.append(row)

    table = pd.DataFrame(rows)
    results = root / "results"
    results.mkdir(exist_ok=True)
    csv_path = results / "numeric_resolution_train.csv"
    md_path = results / "NUMERIC_RESOLUTION_AUDIT.md"
    manifest_path = results / "numeric_resolution_audit_manifest.json"
    for path in [csv_path, md_path, manifest_path]:
        if path.exists():
            raise FileExistsError(
                "Refusing to overwrite an existing audit: " + str(path)
            )

    hu = table.loc[table.is_hu_moment]
    lines = [
        "# 训练特征数值分辨率审计",
        "",
        "本审计只统计原训练 WSI 的有标签细胞；不读取验证/测试成绩，不重拟合变换、不调整模型或特征。",
        "",
        f"- 训练细胞数：{len(training)}。",
        f"- 请求特征数：{len(requested)}；有效特征数：{len(selected)}。",
        "- 数值表：`numeric_resolution_train.csv`，一行一个请求特征。",
        "- 列定义：`raw_nunique_train` 为原始有限值的不同值数；"
        "`transformed_float32_nunique_train` 为现有模型输入的不同值数。",
        "- `unique_retention_ratio` = 转换后不同值数 / 原始有限值不同值数；"
        "缺失中位数插补可能引入一个新值，因此此比例可略大于 1。"
        "`imputed_unique_retention_ratio` 使用训练中位数插补后的不同值数作为分母。",
        "- `raw_std` 和 `transformed_float32_std` 均使用 ddof=0；NaN 单独计数。",
        "- `noise_to_raw_std_ratio` = 1e-5 / 原始训练有限值标准差，仅描述量纲。",
        "- 少于 16 个转换值或保留不足 10% 的标记为工程审计提示，" "不是性能结论、阳性判定或删除特征规则。",
        "",
        "## Hu moments 与固定绝对噪声",
        "",
        "官方 QT 示例在拟合输入上加入标准差 1e-5 的绝对噪声，实际 transform 使用未加噪声的值。"
        "高阶 Hu moments 的量纲可能远小于该噪声；随后转换到 float32 可能压缩可区分值数。"
        "即使 compute_bins 成功，也只证明至少存在两个不同训练值，并不证明所有原始数值分辨率均被保留。",
        "",
        "下表是训练数据的描述诊断，不据此或任何验证表现修改本轮已冻结流程。" "若下一轮探索训练拟合的尺度预处理，应另立实验并与本轮明确区分。",
        "",
        "| 特征 | 原始不同值数 | float32 不同值数 | 插补后唯一值保留率 | 噪声/原始 std | 诊断 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in hu.iterrows():

        def formatted(value):
            return "NA" if pd.isna(value) else format(value, ".6g")

        lines.append(
            f"| {row.feature} | {row.raw_nunique_train} | "
            f"{row.transformed_float32_nunique_train} | "
            f"{formatted(row.imputed_unique_retention_ratio)} | "
            f"{formatted(row.noise_to_raw_std_ratio)} | {row.resolution_note} |"
        )
    lines.extend(["", "输入与脚本 SHA256 见 `numeric_resolution_audit_manifest.json`。"])
    table.to_csv(csv_path, index=False, mode="x")
    with md_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    manifest = {
        "script_sha256": digest(__file__),
        "inputs": {
            name: {"path": str(paths[name]), "sha256": value}
            for name, value in verified_hashes.items()
        },
        "training_ordered_key_sha256": train_key_hash,
        "n_training_cells": len(training),
        "fit_noise_std": NOISE_STD,
        "model_or_preprocessing_changed": False,
        "outputs": {path.name: digest(path) for path in [csv_path, md_path]},
    }
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "csv": str(csv_path),
                "markdown": str(md_path),
                "manifest": str(manifest_path),
                "n_features": len(table),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    main(parser.parse_args().run_dir)
