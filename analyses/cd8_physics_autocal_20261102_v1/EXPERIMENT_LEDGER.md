# EXPERIMENT_LEDGER


## Completed / reuse / missing

| Requirement | Existing evidence | This task |
|---|---|---|
| Source/zero/MAD/cache audit | condition_targetaware results/INPUT_AUDIT, CONDITION_*; 55 raw/cache checks | Reuse; not repeat full source scan |
| HGB/table/normalization/feature/mask ablations | old representation and condition runs | Reuse; do not retrain |
| A three seeds, old target+neighbor, CV folds and checkpoints | 24 model hashes; cached full predictions; CV_A_f0–2 | Freeze/reuse weights and predictions |
| 14 model/input/evaluation tests | 10+4 passed | Run inherited tests as regression, not repeat old fits |
| Random calibration20/50/100 | ENS_A/H0/H1,3repeats;433280 common population | Reuse results, same reserved100/core |
| Stage0 technical perturbations | Not run previously | New required computation |
| Zero-label spatial candidates/mixture/proxy regression | Not run | Three bounded candidate branches, source-grouped dev gates |
| Physical consistency/SDF/ranking | Not run | Conditional development only; one core change per branch, matched added-training control; stop if gates fail |
| Active5/10/20 + random5/10 | Not run | New calibration; same reserved pool; deterministic topK has no inverse-propensity weighting |
| Six metrics incl core F1 median/IQR | Most historical metrics exist; median/IQR not consolidated | Compute from existing per-cell predictions and frozen populations |
|55-case new/old predictions | Fixed55 IDs/images exist; expert decisions blank | ReuseIDs; add predictions for accepted branches, keep expert status pending |

## Evidence and mathematics corrections

- 94.85% zeros precede project clipping; upstream origin remains unknown. No proven background-subtraction/clipping causal claim.
- All-zero cells are not biological negatives. Spatial masks define candidate low-signal anchors only; contamination is possible.
- Helpful supervised intercept adjustment does not establish intercept drift as the sole or main error mechanism.
- Local FDR is not conditional FPR; mixture component identities require assumptions and development checks.
- Non-contact edge strips may contain PSF signal,2D overlap and segmentation error; never call them uncontaminated membrane ground truth.
- Clipping/noise are not generally invertible. Test determinism, valid monotonicity, identities and failure boundaries only.
- 12 prior dev fits =3held-out-trainingWSIs ×4groups, not12independentfolds. Internal-derived perturbations remain grouped with their sourceWSI.
- B/C/D were negative in this bounded experiment, not universally disproven model families. They will not be repeated here.
- 0.8148→0.8780 is not a controlled architectural gain: different runs/training plus operating-point change. Old same-rule mask ensemble remains0.8803.
- Prior new-run validation involved19model score sets,2280unit metric rows and3268stratum rows. MD's28sets/327rows must not be copied as this run's count.
- Three training units make SD estimates unstable; ±5SD alone is not evidence of non-generalization.
- Gates1/2 are targets, not guaranteed outcomes. External is previously exposed exploratory benchmark, never a new blind test.
- Serious information-loss transforms are stress tests; no unconditional logit/observable-label invariance. Scaling/noise are synthetic proxies, not reconstructed acquisition physics.

## Frozen minimum execution sequence (before new predictions)

1. Verify original weights/probabilities/keys, existing tests and no duplicate new server task. New total neural active wall cap28800s; inference/training/failed activities separately recorded.
2. Fixed1536-cell pilot (512 per internalWSI; label-blind hash order) checks identity against stored A scores and runtime. Re-estimate full Stage0 cost from actual pilot; do not silently subsample.
3. Stage0:16 changed scenarios ×3 existing A seeds, full539132 internal labelled cells if timing remains within budget. Reuse clean stored predictions and old mask clean anchors. No full baseline re-inference.
4. Develop three zero-label calibrators with internalWSI-held-out grouping. Freeze before any new external application. Reject cleanF1 degradation or AP loss>.003; do not use external failure to iterate.
5. Conditional model development only after cheap no-retrain diagnostics; at most physical-consistency,SDF,ranking separately, one shared added-training-duration control. Use existing CV_A fold checkpoints with correct training exclusions. New final3seed fits only for a development winner; no old full-model rerun.
6. Active calibration uses only reserved100/core/repeat. New random5/10 and boundary-topK5/10/20; reuse random20/50/100. Fixed regularized intercept, no invalid inverse-propensity weighting or unsupported isotonic fit.
7. Full six metrics,source/core summaries,paired comparisons,seedSD vs probabilityensemble clearly separated;55-case comparisons retained locally; negative and unexecuted branches retained.

## Resource estimate (provisional, pilot will refine)

Old recorded inference ~64s per1.08million rows permodel on RTX4000. Stage0 contains~25.88million changed cell-model evaluations; nominal inference~26minutes, reserve up to3h for transforms/I/O. First round target well below8h; total hardcap8h includes new smoke,inference,training and failures. No paid compute,PanM,network changes or gitpush.

## Execution update 2026-11-02 09:23 UTC

- PROTOCOL SHA256 45780b17daf7e19cdc4039b6c19253f0bf886a11e2402cc698e4c93d9fea5f25.
-14 inherited +17 new tests passed. Pilot1536cells x3seeds exactly reproduces priorprobabilities and decisions.12k-per-scenario timing probes: gain1.5=3.0494s,noisehigh=4.6405s,zero95=2.9994s. Pilot neural charge16.0587s; total pilot/preparation wall168.77s.
- Conservative fullStage0 estimate12509.31s (3.475h), replacing the provisional earlier estimate. Budget gatepassed.
- Conditional plan and training/calibration code frozen before new external evaluation. No newtraining or calibration has executed at this timestamp.

## Final execution outcome (2026-11-02)

- Stage0 completed all48changed-seed predictions:16scenarios ×3seeds ×539132internal cells. Clean baseline reused; no external synthetic sweep. Stage0 audit136metric checks, maxAPerror7.77e-16.
- Zero-label source-grouped development: spatial and mixture passed all3seeds+ENS; ridge failed and was not evaluated externally. Independent replay204metric groups and12source-fold checks reproduced every decision (maxerror1.33e-15), without refitting.
- Pretraining correctness amendments: eligibility now also requires integrated-signal retention and known synthetic-noise RMS ratio; clean-ensemble and subsequent synthetic AP stopping gates were made explicit. Both were frozen before any new training and external branch evaluation; originals and exact revisions retained.
- New continuation:12CVfits completed (3trainingWSIs ×4variants), physics selected. SDF and ranking stopped at CV;6finalfits completed (physics/control ×3seeds). Physics failed full internal gate: ensemble ΔAP+0.003187283 but ΔF1−0.003022875 and meanseedF1declined. No new-model synthetic follow-up or external prediction was run; no replacement branch or tuning after gate failure.
- Two development-passed zero-label methods applied to all437480known-exploratory external cells,3seeds andENS, frozen0.5/internalFPR thresholds. Spatial ENS F1=.892868125,AP=.965875528,weakR=.507091129,FPR=.002579797: Gate1FAIL becauseAP<.966. Mixture also failedGate1. No rounded pass or post-external retuning.
- New15calibration groups completed: random5/10 and active5/10/20 ×3reserved-pool repeats. Historicalrandom0/20/50/100 reused. Eachrepeat excludes4200reservedcells and evaluates433280commoncells. Active20meanF1=.903875168,weakR=.557923478,FPR=0.002178425212: Gate2meanPASS; repeats101/102pass,103fails (.900122F1,.538368weakR). Onecore inrepeat103 hasFPR1.2825%. Deterministicboundaryselection is not IPW-unbiased.
- SpatialcoreF1 improved27/42,worsened14,tied1; mixture22improved20worsened. Both methods leave eachcoreAPunchanged: pooledAPdifferences concern cross-core ordering. SpatialmeanprevalenceMAE=.006077781 vs baseline.007649016.
- Runtime18newweights and actual parent/initialization/epoch-index hashes verified.12paired-epochchecks passed. Final59metricgroups maxerror6.66e-16, plus15earlyfewshotchecks (max5.55e-16).14inherited+17newtests+1synthetic-gate integration test passed;no scientific test failed.
- Totalneuralactivewall5397.5814s/28800s: training409.7261s,inference4978.2117s,smoke/accounting9.6437s;0failedneuralactivities. This includes loading/setup,notGPUkernel-onlytime. CPU phases separately recorded.
-55fixedcases/165priorimage+rawfiles reused byte-for-byte; new scoretables attached. Expertbiologicaladjudication remainsPENDING. DisplayQA is separately recorded. No newdata-source pixels,physicalcontrols,patientmapping or realstainbatchinformation acquired.

## Final local QA and delivery closure

14bilingualscientificfigures were directly visually checked;crowdedF4ticks corrected. All110caseimages and55rawarchives are readable andbyte-identical topriorfiles;6images(R001/R028/R055,bothlanguages) visually sampled. The55cases comprise27internal and28external, with640predictioncomparisonrows. No expertadjudication was performed. HTMLstructure and159localtargets validated;browserUIinspection unavailable because nativeappinventorytimedout andno localbrowserwas listed. This is an explicit verificationlimit,not a biologicalQApass.
