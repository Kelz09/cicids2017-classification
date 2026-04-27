# cicids2017-classification

Comparing classification algorithms for network intrusion detection using the CICIDS2017 dataset.

## Overview

This project evaluates and compares multiple machine learning classification algorithms on the CICIDS2017 network traffic dataset. The goal is to accurately classify different types of network attacks using algorithms including kNN, SVM, Logistic Regression, Decision Tree, Random Forest, Naive Bayes, and XGBoost.

Each algorithm is tested with different preprocessing pipelines including feature scaling, SMOTE oversampling, PCA, and LDA, with hyperparameter tuning via GridSearchCV.

## Team

- Kelechi Uko - Decision Tree, Random Forest
- Topi Korhonen - Logistic Regression, Naive Bayes
- John Ramstedt - kNN, SVM
- Nadja Ahonen - XGBoost

## Dataset

CICIDS2017 - available on Kaggle:
https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed

Download the CSV and place it in the `data/` folder. See `data/README.md` for instructions.

## Setup

```bash
git clone https://github.com/Kelz09/cicids2017-classification.git
cd cicids2017-classification
pip install -r requirements.txt
```

## Structure

```
cicids2017-classification/
├── data_split.py                  # Shared data loading and splitting module
├── requirements.txt
├── README.md
├── data/
│   └── README.md                  # Instructions to download and place the CSV here
├── training/                      # Personal working notebooks
│   ├── DT_RF_Kelechi.ipynb
│   ├── kNN_SVM_John.ipynb
│   ├── LR_NB_Topi.ipynb
│   └── XGBoost_Nadja.ipynb
└── assignment1_submission.ipynb   # Final group submission notebook
```

## Data split

All team members load data using `data_split.py` to guarantee identical 60/20/20 train/validation/test splits across all machines. Do not modify the random seed or split logic without informing the group.
