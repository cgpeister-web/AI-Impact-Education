# Running the AIM 350 model comparison

This repository currently provides a **starter implementation**, not independently verified project results.

## 1. Obtain the data
Follow `data/README.md` and the published dataset license. Place the survey CSV at `data/survey.csv` locally. Do **not** commit respondent-level data without license and privacy review.

## 2. Inspect the codebook
Decide how the survey measures frequency of AI use. Make a **documented binary target** (higher versus lower use) and explain the threshold. The script requires a column that already contains **exactly two distinct values**; it deliberately does not invent a threshold.

Remove all columns that directly determine the target (including original frequency questions used to compute it), identifiers, and any post-outcome items that would create leakage. Pass them to `--exclude`.

## 3. Install dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 4. Run
The following is a **template**, not a confirmed dataset schema; replace column names and values after inspecting your own codebook.

```bash
python -m src.train_models --data data/survey.csv --target AI_USE_BINARY --positive high --exclude ORIGINAL_AI_FREQUENCY STUDENT_ID
```

Outputs are written locally under `results/`: actual model comparison CSV, run metadata and confusion-matrix images. Enter actual results into the report only after executing this script against the intended dataset.

## Important limitations
- One stratified 80/20 holdout; five-fold CV occurs only on the training portion.
- Logistic Regression, Decision Tree, and Random Forest are compared using the same preprocessing and split.
- Interpret these as predictive **associations**, not evidence of cheating, learning harm, or causal impact.
- Check questionnaire wording, class balance, missingness, target definition, leakage, and privacy before submission.
- All group members should test and understand the model and document real individual contributions.
