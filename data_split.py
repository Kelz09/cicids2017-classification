"""
Shared data splitting module for CICIDS2017 classification assignment.

This module ensures all team members use identical train/validation/test splits.
Use this in every personal notebook to guarantee result comparability.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Fixed random seed ensures identical splits across all machines
RANDOM_STATE = 42
TRAIN_SIZE = 0.6
VAL_SIZE = 0.2
TEST_SIZE = 0.2


def load_and_split_data(csv_path, max_rows=100000, shuffle=True):
    """
    Load CICIDS2017 CSV and split into train/validation/test sets.
    
    Args:
        csv_path (str): Path to cicids2017_cleaned.csv
        max_rows (int): Maximum rows to load (default 100000 for memory efficiency)
        shuffle (bool): Whether to shuffle rows before sampling (default True)
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    
    # Load data
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, nrows=max_rows if shuffle else None)
    
    if shuffle:
        df = df.sample(n=min(max_rows, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
    
    print(f"Loaded {len(df)} rows")
    
    # Separate features and target
    # Assuming 'Label' is the target column (adjust if different)
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # First split: 60% train, 40% temp (for val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        train_size=TRAIN_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    # Second split: split temp 50/50 into validation (20% of original) and test (20% of original)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        train_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(X_train)} ({100*len(X_train)/len(df):.1f}%)")
    print(f"  Val:   {len(X_val)} ({100*len(X_val)/len(df):.1f}%)")
    print(f"  Test:  {len(X_test)} ({100*len(X_test)/len(df):.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Test: run this file directly to verify the split works
    # Adjust csv_path to your local path
    csv_path = "/Users/mymac/Downloads/cicids2017_cleaned.csv"
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(csv_path)
