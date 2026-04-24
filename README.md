# CICIDS2017 Classification

Comparing classification algorithms for network intrusion detection using CICIDS2017 dataset.

## Project Overview

This project evaluates and compares 7 machine learning classification algorithms (kNN, SVM, Logistic Regression, Decision Tree, Random Forest, Naive Bayes, XGBoost) on the CICIDS2017 network attack dataset.

For each algorithm, we test variations in:
- Preprocessing: StandardScaler vs MinMaxScaler
- Oversampling: with vs without SMOTE
- Dimensionality reduction: PCA, LDA, Kernel PCA
- Hyperparameter tuning: GridSearchCV

## Team

- Kelechi Uko
- Topi Korhonen
- John Ramstedt
- Nadja Ahonen

## Dataset

Download the CICIDS2017 dataset from Kaggle:
https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed

Place the file at `~/Downloads/cicids2017_cleaned.csv` (or adjust the path in your notebook).

Do NOT commit the CSV to git. The `.gitignore` excludes all `.csv` files.

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Kelz09/cicids2017-classification.git
cd cicids2017-classification
```

### 2. Create a virtual environment (optional but recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify data split works

The most critical file is `data_split.py`. Test it with:

```bash
python data_split.py
```

Edit the `csv_path` variable in `data_split.py` to match your local download path if needed.

## Project Structure

```
cicids2017-classification/
├── README.md                         # This file
├── .gitignore                        # Git ignore rules
├── requirements.txt                  # Python dependencies
├── data_split.py                     # Shared data splitting module (critical)
├── data/                             # Dataset folder (gitignored, local only)
├── training/                         # Personal working notebooks
│   ├── kelechi.ipynb
│   ├── topi.ipynb
│   ├── john.ipynb
│   └── nadja.ipynb
└── assignment1_submission.ipynb      # Final group submission notebook
```

## How to Work

1. **Everyone clones this repo and downloads the dataset locally.**

2. **Each person works in their personal notebook** in the `training/` folder.
   - Import `data_split.py` to load data: `from data_split import load_and_split_data`
   - This guarantees identical train/val/test splits across all team members
   - Train your assigned algorithms with all preprocessing variations

3. **When done, extract your best results** (accuracy, confusion matrix, best hyperparameters) and share with the group.

4. **Assemble the final submission notebook** (`assignment1_submission.ipynb`) with all results combined.

5. **Push everything to git**, then submit the notebooks to your course platform.

## Key Principle

The `data_split.py` file uses a fixed random seed (`RANDOM_STATE = 42`). This means every team member gets the exact same train/validation/test split. This is essential for comparing results across different algorithms.

Do not modify the random seed or the splitting logic without telling the whole group first.

## References

- CICIDS2017 dataset: https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed
- Original paper: "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization"
- scikit-learn docs: https://scikit-learn.org/
- imbalanced-learn (SMOTE): https://imbalanced-learn.org/

## Questions?

Check the assignment instructions in Moodle for detailed requirements on preprocessing, algorithms, and evaluation criteria.
