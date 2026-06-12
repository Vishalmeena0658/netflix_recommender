# 🎬 Netflix Prize Recommender System (Cult Open Projects 2026)

A production-ready, highly optimized personalized recommendation pipeline built entirely from scratch using low-level NumPy and SciPy primitives on the Netflix Prize dataset. This architecture bypasses high-level black-box wrappers (such as `scikit-surprise`) to implement Collaborative Filtering, Content-Based Similarity Matching, and a Hybrid Ensemble Blending engine optimized for continuous rating estimation and ranked discovery performance.

---

## 📌 Project Submission Quick Links
* 📄 **[Technical_Report.pdf](./Technical_Report.pdf)** — Full 10-page analytical report mapping EDA, algorithmic selection, mathematical definitions, and evaluation metrics.
* 📊 **[Project_Presentation.pdf](./Project_Presentation.pdf)** — High-level 8-slide summary deck designed for the judging panel pitch.
* 📓 **[netflix_recommender.ipynb](./netflix_recommender.ipynb)** — Live interactive execution log displaying step-by-step pipeline verifications and visualization charts.

---

## 📂 Core Codebase & Pipeline Directory

Reviewers can find the core operational scripts mapped across the following structural architecture:

```text
netflix_recommender/
│
├── configs/
│   └── config.py               # Hyperparameter scopes, random seeds, and environment paths
│
├── src/                        # Core Modular Implementation Code
│   ├── data/                   # DATA PROCESSING & EDA PIPELINES
│   │   ├── loader.py           # Ingestion pipeline: Text parser converting raw files to CSR Matrices
│   │   └── eda.py              # EDA pipeline: Computes matrix sparsity, trends, and distributions
│   │
│   ├── models/                 # MODEL TRAINING & EVALUATION ARCHITECTURES
│   │   └── recommenders.py     # Contains PureSVD, Content-Based, and Hybrid generation modules
│   │
│   └── evaluation/             # METRICS COMPUTING LAYER
│       └── metrics.py          # Custom formulas for RMSE, MAE, MAP@10, and NDCG@10
│
├── Technical_Report.pdf        # Main 10-page analytical report (PDF of Report)
├── Project_Presentation.pdf    # Main 8-slide pitch deck (PDF of PPT / Slides)
├── netflix_recommender.ipynb   # Interactive analysis notebook (Result showing)
├── app.py                      # Streamlit interactive web dashboard server code
├── requirements.txt            # Hard python environmental library dependencies
├── train_and_evaluate.py       # Terminal pipeline master orchestrator execution script
└── README.md                   # Complete system documentation and path guideation documentation
