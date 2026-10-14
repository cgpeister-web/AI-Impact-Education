# Preliminary reproduced analysis results

These aggregate results come from a local Python run on the original 28850645.zip dataset downloaded from the published DOI. The figures were recomputed independently from the ZIP parsing logic on October 14, 2026 and matched the earlier modeling run (rounded to six decimals).

- Source: https://doi.org/10.17045/STHLMUNI.28850645
- Original records: 1,266
- Included records: 1,230 (high: 575; low: 655)
- Training set: 984; test set: 246; stratified random seed 42
- Binary target: Use_school values 1 or 2 (daily/several times weekly) = high; 3, 4, 5 = low; exclude 0 and missing
- 5-fold stratified cross-validation on training data
- Metrics reflect the held-out test portion. Cross-validation F1 is reported separately.

See `src/reproduce_study.py` to repeat the analysis and regenerate the result figures. The selected threshold and features require group justification; the observational data do not establish adverse effects, academic dishonesty, or causation. Model results are not guaranteed identical under different package versions.

**Authorship notice:** An AI assistant supplied the reproducibility scaffold and documentation; the students must review, understand, adapt, disclose permitted support, and independently validate all submitted work under the course policy.
