"""
preprocessing.py
----------------
Loading, cleaning, encoding, splitting and scaling the Penguins data.
(This is the same code as in the Colab notebook, first part.)
"""

import numpy as np
import pandas as pd

origin = "OriginLocation"

CLASSES = ["Adelie", "Chinstrap", "Gentoo"]
FEATURES = ["CulmenLength", "CulmenDepth", "FlipperLength", origin, "BodyMass"]

label ="Species"


def load_data(path="penguins.csv"):

    data = pd.read_csv(path, na_values=["NA"])

    data.columns = data.columns.str.strip()


    requir= [label, *FEATURES]
    for column in requir:
        if column not in data.columns:
            raise ValueError(f"column not here: {column}")

    data[origin] = data[origin].str.strip()
    data[label] = data[label].str.strip()


    return data


# 2- clean
numeric_f = [
    "CulmenLength",
    "CulmenDepth",
    "FlipperLength",
    "BodyMass"
]

def fill_missing_values(df):
    data = df.copy()

    for column in numeric_f:
        mean_v = data.groupby(label)[column].transform("mean")
        data[column] = data[column].fillna(mean_v)
        data[column] = data[column].fillna(data[column].mean())

    # OriginLocation is text so it has no mean, we use the most common
    # location of the same species instead (no row is dropped)
    for species in data[label].unique():
        same_species = data[label] == species
        most_common = data.loc[same_species, origin].mode()
        missing = same_species & data[origin].isna()
        if len(most_common) > 0:
            data.loc[missing, origin] = most_common[0]

    return data


# text -> number
orgin_map = {
    "Torgersen": 0,
    "Biscoe": 1,
    "Dream": 2
}


def encode_origin_value(value):
    if not isinstance(value, str):
        return value

    loc= value.strip()

    if loc not in orgin_map:
        raise ValueError(f"unknown originLocation: {loc}")

    return orgin_map[loc]


def transform_sample(num1, num2, scaler):
    values = []

    for val in (num1, num2):
        if isinstance(val, str) and val.strip() in orgin_map:
            val = orgin_map[val.strip()]

        values.append(float(val))

    raw = np.array([values])
    return scaler.transform(raw)[0]


def encode_origin(df):
    data = df.copy()

    data[origin] = data[origin].map(encode_origin_value).astype(float)

    return data


# 4- train test
def split_train_test(df, class_names, rng, train_per_class=30):
    train_indices = []
    test_indices = []

    for class_na in class_names:
        class_num = np.where(df[label] == class_na)[0]

        if len(class_num) < 50:
            raise ValueError(f"Class {class_na} has fewer than 50 samples")

        shuffled_indices = rng.permutation(class_num)

        train_indices.extend(shuffled_indices[:train_per_class])
        test_indices.extend(shuffled_indices[train_per_class:50])

    test_indices = rng.permutation(test_indices)
    train_indices = rng.permutation(train_indices)


    return np.array(train_indices), np.array(test_indices)



# 5- (x - mean) / std)
class StandardScaler:
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1
        return self

    def transform(self, X):
        return (X - self.mean) / self.std


def check_choices(feature_names, class_names):
    if len(set(feature_names)) != 2 or any(f not in FEATURES for f in feature_names):
        raise ValueError(f"Choose 2 different features from {FEATURES}")

    if len(set(class_names)) != 2 or any(c not in CLASSES for c in class_names):
        raise ValueError(f"Choose 2 different classes from {CLASSES}")


pos = 1     # first
negative = -1 # last

def prepare_data(featuren, class_n, csv_path="penguins.csv", seed=None):

    check_choices(featuren, class_n)

    rng = np.random.default_rng(seed)
    df = load_data(csv_path)
    df = fill_missing_values(df)
    df = encode_origin(df)

    train_idx, test_idx = split_train_test(df, class_n, rng)
    X = df[list(featuren)].to_numpy(dtype=float)
    y = np.where(df[label] == class_n[0], pos, negative)

    scaler = StandardScaler().fit(X[train_idx])

    X_test = scaler.transform(X[test_idx])
    X_train = scaler.transform(X[train_idx])

    y_test = y[test_idx]
    y_train = y[train_idx]

    return X_train, y_train, X_test, y_test, scaler