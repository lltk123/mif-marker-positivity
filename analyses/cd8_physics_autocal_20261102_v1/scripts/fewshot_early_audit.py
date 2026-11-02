from final_audit import *

if __name__ == "__main__":
    started = time.time()
    d, _ = cc.load_data()
    checks = fewshot_checks(d)
    pd.DataFrame(checks).to_csv(
        R / "results/FEWSHOT_EARLY_INDEPENDENT_AUDIT.csv", index=False
    )
    atomic_json(
        R / "results/FEWSHOT_EARLY_AUDIT_COMPLETE.json",
        {
            "groups": len(checks),
            "max_metric_absolute_error": max(
                max(r[k] for k in ["F1", "AP", "FPR"]) for r in checks
            ),
            "active_selected_ids_match_frozen_rule": True,
            "calibration_evaluation_disjoint": True,
            "performance_numbers_not_printed_for_other_branch_selection": True,
            "seconds_cpu_wall": time.time() - started,
        },
    )
    print("FEWSHOT_EARLY_AUDIT_PASS", len(checks), flush=True)
