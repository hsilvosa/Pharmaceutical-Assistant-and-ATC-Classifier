"""Group-aware train/validation/test split for the ATC dataset.

The dataset has one row per medicine presentation, so many rows share the same
text and the same active ingredients. A random row split therefore leaks almost
every validation text into training. Splitting by active-ingredient set keeps
all presentations of one ingredient combination in the same partition, so the
validation and test sets measure generalisation to unseen ingredients.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def group_key(df: pd.DataFrame) -> pd.Series:
    """Normalised active-ingredient set, falling back to the medicine name."""
    ing = df["active_ingredients_str"].fillna("").str.strip().str.upper()
    name = df["name"].fillna("").str.strip().str.upper()
    return ing.where(ing != "", name)


def grouped_split(df: pd.DataFrame, seed: int, val_size: float = 0.10, test_size: float = 0.10):
    """Return (train, val, test) with disjoint active-ingredient groups."""
    groups = group_key(df).values
    outer = GroupShuffleSplit(n_splits=1, test_size=val_size + test_size, random_state=seed)
    train_idx, rest_idx = next(outer.split(df, groups=groups))
    rest = df.iloc[rest_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=test_size / (val_size + test_size), random_state=seed)
    val_rel, test_rel = next(inner.split(rest, groups=groups[rest_idx]))
    train, val, test = df.iloc[train_idx], rest.iloc[val_rel], rest.iloc[test_rel]
    assert not set(group_key(train)) & set(group_key(val))
    assert not set(group_key(train)) & set(group_key(test))
    assert not set(group_key(val)) & set(group_key(test))
    return train, val, test
