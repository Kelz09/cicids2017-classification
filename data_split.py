"""
Shared data splitting module for CICIDS2017 classification assignment.

We imports this file in our personal notebook to guarantee
identical train/validation/test splits across all machines and algorithms.

Usage in your notebook (from inside the training/ folder):
    import sys
    sys.path.append('..')
    from data_split import load_and_split_data, RANDOM_STATE

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
        "../data/cicids2017_cleaned.csv",
        max_rows=100000
    )
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TRAIN_SIZE = 0.6


def load_and_split_data(csv_path, max_rows=100000, shuffle=True):
    """
    Load CICIDS2017 CSV and return identical 60/20/20 train/val/test splits.

    Args:
        csv_path (str): Path to cicids2017_cleaned.csv
        max_rows (int): Maximum rows to sample (default 100000)
        shuffle (bool): Shuffle before sampling so the sample is representative

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    if shuffle:
        df = df.sample(
            n=min(max_rows, len(df)),
            random_state=RANDOM_STATE
        ).reset_index(drop=True)

    print(f"Loaded {len(df)} rows")

    # Strip whitespace from column names to avoid subtle mismatches
    df.columns = df.columns.str.strip()

    # Detect target column - handles different CSV versions
    candidates = ["Attack Type", "attack_type", "Label", "label", "Class", "class", "Target", "target"]
    target_col = next((c for c in candidates if c in df.columns), df.columns[-1])
    print(f"Target column: {target_col}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Handle categorical columns and bad numeric values
    X = pd.get_dummies(X, drop_first=False)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"Features: {X.shape[1]} columns")
    print(f"Target distribution:\n{y.value_counts()}\n")

    # Split 1: 60% train, 40% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        train_size=TRAIN_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Split 2: temp into 50/50 giving 20% val and 20% test of original
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        train_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    print(f"Train:      {len(X_train)} rows ({100*len(X_train)/len(df):.0f}%)")
    print(f"Validation: {len(X_val)} rows ({100*len(X_val)/len(df):.0f}%)")
    print(f"Test:       {len(X_test)} rows ({100*len(X_test)/len(df):.0f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test
