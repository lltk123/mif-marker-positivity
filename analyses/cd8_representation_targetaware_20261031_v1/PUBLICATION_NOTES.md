# Publication changes and provenance

Source: verified `CD8_report_code_20261031_final.zip`, SHA256 `917de7c86f117d5e2a3a8e8b765698de49ac2e1e355b277d1bf44d61c7bb5ab8`, completed 2026-10-31. The 2026-11-01 publication uses final correctness_001 code, not the unfinished top-level working scripts.

1. Result CSVs, model JSON metadata, frozen protocol and scientific PNG/SVG/PDF files retain original bytes. Prediction completion JSON files are moved under provenance; prediction arrays are excluded.
2. Fixed executable paths become explicit environment configuration in `runtime_paths.py`. Compatibility copies of legacy feature definitions/readers change imports/path resolution only, avoiding machine-specific `common.py` import side effects. Existing experiments are unchanged.
3. Python source is formatted with Black; its AST is checked before/after formatting. Source/publication hashes and reasons are recorded in [SOURCE_MANIFEST.csv](provenance/SOURCE_MANIFEST.csv). Formatting/path changes do not describe the code hash used during the original training.
4. Chinese report changes are repository links and publication notes. English report and table/conclusion/version indexes are added from the saved evidence. The historical protocol's no-automatic-push flag describes the original experiment; this publication is separately authorized.
5. Raw data, models/weights, per-cell predictions/reserved IDs/composition rows, case images, review IDs, credentials, private conversations and Library receipts are omitted. Only review-generation code and non-image instructions are public.

No numerical results, selected model or threshold values were changed. No training, GPU jobs, SSH experiment access or Pan-M access occurred during publication.

Git attributes disable automatic newline rewriting inside this new experiment directory so frozen evidence and source-manifest SHA256 values survive checkout unchanged. This does not change attributes for older experiments.
