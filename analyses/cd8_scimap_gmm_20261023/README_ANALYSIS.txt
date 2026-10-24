PhenoBIC CD8 SCIMAP GMM analysis
Server analysis root: /mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_scimap_gmm_20261023
Primary report: CD8_GMM_ANALYSIS_REPORT.html
Human-readable annotation links: annotations/<original_split>/<context>/<unit>_CD8.csv.gz
Complete cell table index: ANNOTATION_INDEX.csv (server); results/phenobic/cell_table_index.csv
Original per-cell gzip tables are retained on the server and are not included in the compact local report bundle.
Key columns: cell_id (original segmentation ID), mean_CD8, rescale_score, prediction_CD8, ground_truth_original, eligible_binary_evaluation, record_status.
All segmented cells have predictions. Unannotated cells remain in the tables. No human labels entered GMM fitting.
Continuous rescale_score is a ranking score, not a calibrated probability. Binary CD8+ is strictly score > 0.5.
Reproduce using the analysis-root .venv/bin/python with scripts/extract_cd8_v2.py, scripts/analyze_cd8_v2.py, and scripts/verify_results.py. The verify script creates an exclusive ZIP; use a fresh output name on rerun.
Plots: python scripts/make_report.py --root <analysis-root>. Requires matplotlib and a Chinese font (Microsoft YaHei on Windows or Noto Sans CJK SC on Linux).
Pan-Multiplex Gold-Standard is sealed and excluded from this analysis. No encryption or filesystem access changes are implied.
