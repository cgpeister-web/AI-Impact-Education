# AI-Impact-Education
Survey-based prediction of generative AI use among Swedish upper-secondary students.
Adverse Effects of Artificial Intelligence on Modern Education
AIM 350 – Term Group Project

Group 4: Almonte, Murtaza, Peister, Valiunas
Instructor: Dr. Bruno G. Kamdem
Semester: Fall 2026

Project Overview

This project examines how secondary school students use generative artificial intelligence (AI) tools for schoolwork and whether patterns in AI use are associated with students' attitudes, awareness, and perceptions of AI-related academic behavior.

The project uses supervised machine learning classification to identify patterns associated with different levels of student AI use for schoolwork. We will compare three classification models and evaluate their performance using the same dataset and evaluation criteria.

The project is not intended to assume that AI use is inherently positive or negative. Instead, the goal is to use survey data to identify patterns that can help us better understand how students are using AI in an educational setting.


## Reproducing the machine-learning comparison

A generic modeling template is in `src/train_models.py`; the **tested source-specific runner** is `src/reproduce_study.py`. It compares Logistic Regression, Decision Tree, and Random Forest using a stratified 80/20 split and cross-validation on the training portion.

**Read [RUNNING_MODELS.md](RUNNING_MODELS.md) before execution.** The team must first obtain the survey data, inspect its codebook, construct and justify the binary target, and exclude target-defining/identifying columns.

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

For the actual published survey use the command in **Verified-dataset reproduction workflow** below. The generic template in RUNNING_MODELS.md is optional and requires different input preparation. The published source ZIP was run successfully on Windows using the source-specific runner, with aggregate results saved in `analysis_results/model_comparison.csv`.

## Code, data and ethics

- [Tested survey model comparison script](src/reproduce_study.py)\n- [Generic modeling template](src/train_models.py)\n- [Aggregate model results](analysis_results/model_comparison.csv)
- [Data provenance](data/README.md)
- [Setup and reproducibility instructions](RUNNING_MODELS.md)

This observational survey may identify associations, not causal harms or proof of cheating. Only commit data permitted by its license, and report results only after running the models.


## Verified-dataset reproduction workflow (October 2026)

The group proposal selects Wickberg Hugerth's 1,266-response Swedish secondary-school survey ([dataset DOI](https://doi.org/10.17045/STHLMUNI.28850645)). The script `src/reproduce_study.py` reads the **original 28850645.zip** directly, including the CSV's double-quoted wrapper. No manually normalized CSV is required.

```bash
python -m pip install -r requirements.txt
python -m src.reproduce_study --zip 28850645.zip --out results
```

The source ZIP is **not committed**; download it from the DOI and keep it in the project directory. The program writes `results/model_comparison.csv`, `results/methodology.json`, and comparison/confusion-matrix figures. The local working analysis (not a CI run) previously returned 1,230 eligible responses, using `Use_school` 1-2 as high and 3-5 as low frequency; 0/unknown and blanks were excluded. The target binning is an analytical choice and should be explained in the report.

**Research caution:** This is a Swedish school survey, despite the earlier one-line repository description mentioning the U.S. It predicts survey-reported frequency, not actual academic dishonesty or whether AI *caused* adverse effects. Document genuine group contributions and disclose any AI support as required by the course.
