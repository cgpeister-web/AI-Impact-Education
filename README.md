# AI-Impact-Education
AI use and its potential adverse effects in U.S. education.
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

The classification runner is `src/train_models.py`. It compares Logistic Regression, Decision Tree, and Random Forest using a stratified 80/20 split and cross-validation on the training portion.

**Read [RUNNING_MODELS.md](RUNNING_MODELS.md) before execution.** The team must first obtain the survey data, inspect its codebook, construct and justify the binary target, and exclude target-defining/identifying columns.

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

The command template in RUNNING_MODELS.md must be customized to the actual dataset schema. Output files in `results/` must be generated from real runs; performance metrics have not yet been verified for this repository.

## Code, data and ethics

- [Model comparison script](src/train_models.py)
- [Data provenance](data/README.md)
- [Setup and reproducibility instructions](RUNNING_MODELS.md)

This observational survey may identify associations, not causal harms or proof of cheating. Only commit data permitted by its license, and report results only after running the models.
