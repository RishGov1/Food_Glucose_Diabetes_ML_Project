# Local training run — 12 October 2026

Training and inference completed on this laptop. The deployment is a research prototype, not a clinically validated digital twin.

- CPU: Intel Core Ultra 5 225H; statistical-model training and personal updates.
- GPU: Intel Arc 130T; actual dense-network forward propagation, backpropagation and Adam updates through OpenCL. CPU reference comparison verifies the first optimizer step.
- NPU: Intel AI Boost; OpenVINO inference verified on six different curves. Maximum difference from NumPy: 0.086 mg/dL. NPU inference is available through the Streamlit comparison button.
- Training corpus: 60 fully observed meal curves from 7 participants; 387 extracted events overall. Sparse events are not given invented future labels.
- Selected forecast: Extra Trees residual regression, based on four-fold participant-separated CV within 5 training participants. CV MAE: 23.55 mg/dL, vs 29.51 for persistence.
- Untouched holdout: 2 participants / 23 meals; 60-minute MAE 14.16, 120-minute MAE 52.65, overall MAE 33.41 mg/dL. Baseline overall MAE 31.19. This does not demonstrate generalization improvement.
- Additional forecast candidates: ridge, random forest, and 4→32→16→2 dense neural network. All retained under `models/`.
- Legacy classifiers: logistic regression for six-point cohort classification; random forest for glucose-only cross-dataset stress testing. PIMA AUC 0.7271 on 763 valid-glucose records. Protocol differences make these unsuitable for clinical diagnosis.
- Personal model: saved measured outcomes update a shrunk offset; after eight meals, a food-aware ridge candidate must pass a later-meal comparison. Synthetic examples use separate demo profiles. Source datasets and historical meal records are preserved.
- Verification: 9 automated tests passed, including GPU gradient comparison, persistent learning, duplicate/profile isolation, stale form rejection, food search and NPU inference through Streamlit. Dependency consistency and Git whitespace checks passed. The live localhost health endpoint returned HTTP 200.

See `outputs/twin_metrics.json` and `outputs/hardware.json` for machine-readable evidence, and the README for instructions and sample values. The earlier `PROJECT_REPORT.md` is historical and does not describe the current deployment.
