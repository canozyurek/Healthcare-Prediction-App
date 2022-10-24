import autofeat
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from .modeltools import make_preprocessor, return_col_names
import joblib


class Autofeaturetool:
    def __init__(self, data_x, data_y, task="classification"):
        if task == "classification":
            self.auto_feat = autofeat.AutoFeatClassifier(
                n_jobs=-1,
                transformations=("1/", "exp", "log", "abs", "sqrt", "^2", "^3"),
                verbose=True,
            )
        else:
            self.auto_feat = autofeat.AutoFeatRegressor(
                n_jobs=-1,
                transformations=("1/", "exp", "log", "abs", "sqrt", "^2", "^3"),
                verbose=True,
            )
        self.data = data_x.copy()
        self.label = data_y.copy()
        self.preprocessor = make_preprocessor(self.data, scaler=False)
        self.preprocessor.fit(self.data)
        self.data_x_new = self.preprocessor.transform(self.data)

        self.data_x_new = return_col_names(self.data_x_new, self.preprocessor)
        self.col_names = self.data_x_new.columns
        self.auto_feat.fit(self.data_x_new, self.label)

    def transform(self, data_x):
        data_x_transformed = self.preprocessor.transform(data_x)
        data_x_transformed = pd.DataFrame(data_x_transformed, columns=self.col_names)
        transformed_data = self.auto_feat.transform(data_x_transformed)
        return transformed_data


def cluster_data(data):
    with open("models/kmeans.sav", "rb") as cluster_loc:
        clustering = joblib.load(cluster_loc)
    new_data = data.copy()
    preprocessor = make_preprocessor(new_data, scaler=True)
    data_scaled = preprocessor.fit_transform(new_data)
    new_data["clusters"] = clustering.predict(data_scaled)
    new_data["clusters"] = new_data["clusters"].astype("category")
    return new_data


def add_features(data, exclude=None):
    transformed_data = data.copy()
    bins = pd.IntervalIndex.from_tuples(
        [
            (-1, 3),
            (3, 13),
            (13, 20),
            (20, 30),
            (30, 40),
            (40, 50),
            (50, 60),
            (60, 70),
            (70, 80),
            (80, 90),
            (90, 100),
        ]
    )
    labels = [
        "0-3",
        "4-13",
        "14-20",
        "21-30",
        "31-40",
        "41-50",
        "51-60",
        "61-70",
        "71-80",
        "81-90",
        "91-100",
    ]
    transformed_data["age_bins"] = pd.cut(data["age"], bins).map(
        dict(zip(bins, labels))
    )
    transformed_data["age_bins"] = transformed_data["age_bins"].astype("object")
    if exclude == "glucose":
        pass
    else:
        hyperglycemia = data["avg_glucose_level"] >= 200
        hypoglycemia = data["avg_glucose_level"] < 70
        transformed_data.loc[hyperglycemia, "glucose_condition"] = "hyperglycemia"
        transformed_data.loc[hypoglycemia, "glucose_condition"] = "hypoglycemia"
        transformed_data.loc[
            (~hyperglycemia) & (~hypoglycemia), "glucose_condition"
        ] = "normal"
    if exclude == "bmi":
        pass
    else:
        bmi_bins = pd.IntervalIndex.from_tuples(
            [(0, 18.5), (18.5, 25), (25, 30), (30, 40), (40, 60), (60, np.inf)]
        )
        weight_labels = [
            "Underweight",
            "Healthy",
            "Overweight",
            "Obese",
            "Severely Obese",
            "Extreme Values",
        ]
        transformed_data["weight_cats"] = pd.cut(data["bmi"], bmi_bins).map(
            dict(zip(bmi_bins, weight_labels))
        )

    return transformed_data
