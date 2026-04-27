# Data

Place the CICIDS2017 dataset CSV file in this folder.

## Download

Download `cicids2017_cleaned.csv` from Kaggle:
https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed

## Setup

1. Download the CSV from the link above
2. Place it here: `cicids2017-classification/data/cicids2017_cleaned.csv`
3. In your notebook, load it with:

```python
import sys
sys.path.append('..')
from data_split import load_and_split_data

X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
    "../data/cicids2017_cleaned.csv",
    max_rows=100000
)
```

The CSV is excluded from git via `.gitignore`. Do not commit it.
