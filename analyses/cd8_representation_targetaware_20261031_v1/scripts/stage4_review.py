"""Export a finite, source-balanced set of REAL crops for user review."""
import os

os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import json, hashlib, html, time
import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from experiment_common import *
import stage1_prepare as prep


def cjk_font():
    names = [f.name for f in font_manager.fontManager.ttflist]
    for choice in [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Micro Hei",
        "SimHei",
        "Microsoft YaHei",
    ]:
        if choice in names:
            return choice
    raise RuntimeError(
        "Chinese font unavailable; Python figure rendering paused rather than substituting a different backend"
    )


def raw_crop(row):
    e = prep.reader.BYKEY[row.source_key]
    if row.source_key not in prep.BBOX_CACHE:
        with np.load(prep.MORPH / "bboxes" / (row.source_key + ".npz")) as z:
            ids = z["cell_id"]
            boxes = z["bbox"]
        order = np.argsort(ids)
        prep.BBOX_CACHE[row.source_key] = (ids[order], boxes[order])
    ids, boxes = prep.BBOX_CACHE[row.source_key]
    ix = np.searchsorted(ids, int(row.cell_id))
    assert ids[ix] == int(row.cell_id)
    im, lab, valid, origin, trunc = prep.reader.virtual_roi(e, boxes[ix])
    M = lab == int(row.cell_id)
    yy, xx = np.nonzero(M)
    cy = int(np.rint(yy.mean() + origin[0] - e["offset_y"]))
    cx = int(np.rint(xx.mean() + origin[1] - e["offset_x"]))
    im, lab, valid, origin, trunc = prep.reader.virtual_roi(
        e, (cy - 41, cy + 41, cx - 41, cx + 41)
    )
    assert im.shape == (96, 96)
    return im, lab, valid, origin, e


def stable(g, salt):
    keys = [
        int.from_bytes(
            hashlib.blake2b(
                (salt + "|" + str(z.image_id) + "|" + str(int(z.cell_id))).encode(),
                digest_size=8,
            ).digest(),
            "little",
        )
        for z in g.itertuples()
    ]
    return g.iloc[np.argsort(np.asarray(keys, np.uint64))]


def main():
    assert (RUN / "results/EVALUATION_COMPLETE.json").exists()
    d, man = load()
    frozen = json.loads((RUN / "models/MODEL_SELECTION_FROZEN.json").read_text())
    selected = frozen["selected_model"]
    names = [
        "H0",
        selected,
        "ENS_image",
        "ENS_image_target",
        "ENS_image_target_other",
        "ENS_tabular_only",
        "ENS_image_target_other_tabular",
    ]
    names = list(dict.fromkeys(names))
    scores = {
        name: np.load(RUN / "predictions" / (name + "_all.npy")) for name in names
    }
    threshold = {
        name: json.loads((RUN / "models" / (name + ".json")).read_text())["thresholds"][
            "dev_recall_90"
        ]
        for name in names
    }
    y = d.ground_truth_binary.to_numpy(int)
    base = scores["H0"] > threshold["H0"]
    chosen = scores[selected] > threshold[selected]
    rr = risks(d)
    categories = {
        "weak_positive_FN": rr["weak_positive"],
        "contact_bright_negative_FP": rr["contact_bright_proxy_negative"],
        "zero_target_signal_error": rr["zero_target_signal"],
        "boundary_abnormal_error": rr["boundary_abnormal"],
    }
    used = set()
    cases = []
    availability = []
    for split in [DEV, TEST]:
        for context, g in d[d.split == split].groupby("context", sort=True):
            for category, mask in categories.items():
                pool = g.loc[mask[g.index.to_numpy()]]
                pool = pool.loc[~pool.index.isin(used)]
                original_n = len(pool)
                selected_rows = []
                for desired in [
                    "baseline_wrong_candidate_correct",
                    "baseline_correct_candidate_wrong",
                ]:
                    ix = pool.index.to_numpy()
                    condition = (
                        (base[ix] != y[ix]) & (chosen[ix] == y[ix])
                        if desired == "baseline_wrong_candidate_correct"
                        else (base[ix] == y[ix]) & (chosen[ix] != y[ix])
                    )
                    candidate = pool.loc[condition]
                    if len(candidate):
                        row = stable(candidate, category + "|" + desired).iloc[0]
                        selected_rows.append((row, desired))
                        pool = pool.drop(index=row.name)
                while len(selected_rows) < 2 and len(pool):
                    ix = pool.index.to_numpy()
                    wrong = (base[ix] != y[ix]) | (chosen[ix] != y[ix])
                    candidate = pool.loc[wrong]
                    reason = "shared_or_single_model_error"
                    if not len(candidate):
                        candidate = pool
                        reason = "matched_control_no_available_error"
                    row = stable(candidate, category + "|fallback").iloc[0]
                    selected_rows.append((row, reason))
                    pool = pool.drop(index=row.name)
                availability.append(
                    {
                        "split": split,
                        "context": context,
                        "category": category,
                        "available_distinct_instances": original_n,
                        "exported": len(selected_rows),
                        "requested": 2,
                    }
                )
                for row, reason in selected_rows:
                    used.add(int(row.name))
                    cases.append((row, category, reason))
    (RUN / "review/raw").mkdir(parents=True, exist_ok=True)
    (RUN / "review/figures").mkdir(exist_ok=True)
    font = cjk_font()
    records = []
    for number, (row, category, reason) in enumerate(cases, 1):
        cid = "C%03d" % number
        im, lab, valid, origin, e = raw_crop(row)
        target = (lab == int(row.cell_id)) & valid
        other = (lab > 0) & ~target & valid
        np.savez_compressed(
            RUN / "review/raw" / (cid + ".npz"),
            raw_CD8=im,
            original_instance_labels=lab,
            valid_support=valid,
            target_cell_id=np.asarray(int(row.cell_id)),
            global_origin_yx=np.asarray(origin),
            normalization_L=np.asarray(row.L),
            normalization_U=np.asarray(row.U),
        )
        vmax = float(row.U)
        rec = {
            "case_id": cid,
            "split": row.split,
            "context": row.context,
            "image_id": row.image_id,
            "cell_id": int(row.cell_id),
            "row_id": int(row.row_id),
            "proxy_category": category,
            "selection_reason": reason,
            "original_reference_label": int(row.ground_truth_binary),
            "selected_candidate": selected,
            "threshold_rule": "internal-WSI developed recall90; strict >",
            "source_key": row.source_key,
            "source_image_relative_path": e["source"]["image"],
            "crop_native_pixels": "96x96",
            "global_origin_y": int(origin[0]),
            "global_origin_x": int(origin[1]),
            "display_vmin": 0.0,
            "display_vmax": vmax,
            "gamma": 1.0,
            "raw_npz": "raw/" + cid + ".npz",
            "raw_pixels_sha256": hashlib.sha256(
                np.ascontiguousarray(im).tobytes()
            ).hexdigest(),
            "user_decision": "",
            "segmentation_issue": "",
            "reviewer_confidence": "",
            "reviewer_notes": "",
        }
        for name in names:
            rec[name + "_score"] = float(scores[name][row.row_id])
            rec[name + "_threshold"] = threshold[name]
            rec[name + "_prediction"] = int(scores[name][row.row_id] > threshold[name])
        for lang in ["zh", "en"]:
            plt.rcParams.update(
                {
                    "font.family": font if lang == "zh" else "DejaVu Sans",
                    "font.size": 9,
                    "svg.fonttype": "none",
                }
            )
            fig, ax = plt.subplots(
                1, 2, figsize=(5.2, 2.65), facecolor="black", layout="constrained"
            )
            for a in ax:
                a.imshow(
                    np.where(valid, im, np.nan),
                    cmap="gray",
                    vmin=0,
                    vmax=vmax,
                    interpolation="nearest",
                )
                a.set_axis_off()
            ax[0].set_title(
                "原始CD8（线性显示）" if lang == "zh" else "Raw CD8 (linear display)",
                color="white",
                fontsize=9,
            )
            ax[1].set_title(
                "目标与其他既有实例" if lang == "zh" else "Target and other instances",
                color="white",
                fontsize=9,
            )
            if target.any():
                ax[1].contour(
                    target.astype(float),
                    levels=[0.5],
                    colors=["#e7b04a"],
                    linewidths=0.8,
                )
            if other.any():
                ax[1].contour(
                    other.astype(float),
                    levels=[0.5],
                    colors=["#67b8c3"],
                    linewidths=0.55,
                )
            for a in ax:
                a.plot([5, 25], [89, 89], color="white", lw=2)
                a.text(
                    15,
                    85,
                    "20像素" if lang == "zh" else "20 px",
                    ha="center",
                    color="white",
                    fontsize=8,
                )
            fig.suptitle(cid + " | " + str(row.context), color="white", fontsize=10)
            fig.savefig(
                RUN / "review/figures" / (cid + "_" + lang + ".png"),
                dpi=180,
                facecolor="black",
            )
            plt.close(fig)
        records.append(rec)
    pd.DataFrame(records).to_csv(RUN / "review/CASE_REVIEW.csv", index=False)
    pd.DataFrame(availability).to_csv(RUN / "review/CASE_AVAILABILITY.csv", index=False)
    for lang in ["zh", "en"]:
        cards = []
        for rec in records:
            score_rows = "".join(
                "<tr><td>"
                + html.escape(name)
                + "</td><td>"
                + format(rec[name + "_score"], ".4f")
                + "</td><td>"
                + str(rec[name + "_prediction"])
                + "</td></tr>"
                for name in names
            )
            title = (
                "参考标签与冻结模型分数（可先完成独立判读再展开）"
                if lang == "zh"
                else "Reference annotation and frozen model scores (review image first if desired)"
            )
            cards.append(
                "<article><h2>"
                + rec["case_id"]
                + " · "
                + html.escape(rec["context"])
                + '</h2><img src="figures/'
                + rec["case_id"]
                + "_"
                + lang
                + '.png"><p>'
                + html.escape(rec["proxy_category"] + " / " + rec["selection_reason"])
                + "</p><details><summary>"
                + title
                + "</summary><p>Reference: "
                + str(rec["original_reference_label"])
                + "</p><table><tr><th>Model</th><th>Score</th><th>Prediction</th></tr>"
                + score_rows
                + "</table></details></article>"
            )
        intro = (
            "真实像素病例审阅；目标为扩张核近似实例，金色为目标、青色为其他既有实例。比例尺仅为像素，未知微米尺度。每例以原始单位共同上界U作线性显示上限，未做局部增强；原像素与mask保存在NPZ。此为说明性抽样，不能据此估计总体错误率。请在CASE_REVIEW.csv填写 user_decision（positive/negative/uncertain）、segmentation_issue、reviewer_confidence 和 reviewer_notes；参考标签与自动代理不等同新增专家复核。"
            if lang == "zh"
            else "Real pixel review. Gold: target expanded-nucleus proxy; cyan: other existing instances. Scale is in native pixels, not micrometres. Linear display uses the original unit shared upper bound U; raw arrays and masks are retained. This illustrative sample cannot estimate the population error rate. Fill user_decision, segmentation_issue, reviewer_confidence and reviewer_notes in CASE_REVIEW.csv. Reference labels and automated proxies are not new expert review."
        )
        page = (
            '<!doctype html><meta charset="utf-8"><title>CD8 review</title><style>body{font:16px sans-serif;max-width:1150px;margin:30px auto;padding:0 20px;background:#f4f5f6;color:#182936}article{background:white;margin:24px 0;padding:20px;border-radius:12px}img{max-width:850px;width:100%;display:block}table{border-collapse:collapse}td,th{padding:7px 16px;border-bottom:1px solid #ddd}summary{cursor:pointer}</style><h1>CD8 '
            + ("真实病例审阅" if lang == "zh" else "real-case review")
            + "</h1><p>"
            + intro
            + "</p>"
            + "".join(cards)
        )
        (RUN / "review" / ("index_" + lang + ".html")).write_text(
            page, encoding="utf-8"
        )
    savej(
        RUN / "review/REVIEW_COMPLETE.json",
        {
            "n_cases": len(records),
            "requested_maximum": 48,
            "all_real_crops": True,
            "user_review_status": "pending",
            "reference_labels_unchanged": True,
            "mask_type": "expanded_nucleus; not verified true membrane",
            "contrast": "linear [0,unit U]; no gamma/local selective manipulation",
            "source_data": "raw/*.npz",
            "code_sha256": sha(__file__),
        },
    )
    print("REVIEW_EXPORT_COMPLETE", len(records), flush=True)


if __name__ == "__main__":
    main()
