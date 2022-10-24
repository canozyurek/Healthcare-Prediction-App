import pickle
from helper.modeltools import return_col_names
from helper.feattools import add_features
import joblib


def stroke_data_transform(data):
    with open("models/stroke_preprocessor_1", "rb") as preprocessor_1_loc:
        preprocessor_1 = pickle.load(preprocessor_1_loc)
    with open("models/stroke_preprocessor_2", "rb") as preprocessor_2_loc:
        preprocessor_2 = pickle.load(preprocessor_2_loc)
    with open("models/stroke_autofeat", "rb") as autofeat_loc:
        auto_features = pickle.load(autofeat_loc)
    with open("models/stroke_final_cols", "rb") as final_cols:
        lim_cols = pickle.load(final_cols)
    X_test_transformed = preprocessor_1.transform(data)
    X_test_transformed = return_col_names(X_test_transformed, preprocessor_1)
    X_test_transformed = auto_features.transform(X_test_transformed)
    X_test_transformed = add_features(X_test_transformed)
    X_test_transformed = preprocessor_2.transform(X_test_transformed)
    X_test_transformed = return_col_names(X_test_transformed, preprocessor_2)
    X_test_transformed = X_test_transformed[lim_cols]
    return X_test_transformed


def hypertension_data_transform(data):
    with open("models/hypertension_preprocessor_2", "rb") as preprocessor_2_loc:
        preprocessor_2 = joblib.load(preprocessor_2_loc)
    X_test_transformed = preprocessor_2.transform(data)
    X_test_transformed = return_col_names(X_test_transformed, preprocessor_2)
    return X_test_transformed


def glucose_data_transform(data):
    with open("models/glucose_preprocessor_2", "rb") as preprocessor_2_loc:
        preprocessor_2 = pickle.load(preprocessor_2_loc)
    with open("models/glucose_autofeat", "rb") as autofeat_loc:
        auto_features = pickle.load(autofeat_loc)
    with open("models/glucose_final_cols", "rb") as final_cols:
        lim_cols = pickle.load(final_cols)
    X_test_transformed = auto_features.transform(data)
    X_test_transformed = add_features(X_test_transformed, exclude="glucose")
    X_test_transformed = preprocessor_2.transform(X_test_transformed)
    X_test_transformed = return_col_names(X_test_transformed, preprocessor_2)
    X_test_transformed = X_test_transformed[lim_cols]
    return X_test_transformed


def bmi_data_transform(data):
    with open("models/bmi_preprocessor_2", "rb") as preprocessor_2_loc:
        preprocessor_2 = pickle.load(preprocessor_2_loc)
    with open("models/bmi_autofeat", "rb") as autofeat_loc:
        auto_features = pickle.load(autofeat_loc)
    with open("models/bmi_final_cols", "rb") as final_cols:
        lim_cols = pickle.load(final_cols)
    X_test_transformed = auto_features.transform(data)
    X_test_transformed = preprocessor_2.transform(X_test_transformed)
    X_test_transformed = return_col_names(X_test_transformed, preprocessor_2)
    X_test_transformed = X_test_transformed[lim_cols]
    return X_test_transformed


def hyp_glu_transformation(data):
    with open("models/hyp_glu_dual/cols_1", "rb") as col_1:
        cols_1 = pickle.load(col_1)
    with open("models/hyp_glu_dual/cols_2", "rb") as col_2:
        cols_2 = pickle.load(col_2)
    data_1 = hypertension_data_transform(data)
    data_2 = glucose_data_transform(data)
    data_1 = data_1[cols_1]
    data_2 = data_2[cols_2]
    return data_1, data_2


def hyp_bmi_transformation(data):
    with open("models/hyp_bmi_dual/cols_1", "rb") as col_1:
        cols_1 = pickle.load(col_1)
    with open("models/hyp_bmi_dual/cols_2", "rb") as col_2:
        cols_2 = pickle.load(col_2)
    data_1 = hypertension_data_transform(data)
    data_2 = bmi_data_transform(data)
    data_1 = data_1[cols_1]
    data_2 = data_2[cols_2]
    return data_1, data_2


def glu_bmi_transformation(data):
    with open("models/glu_bmi_dual/cols_1", "rb") as col_1:
        cols_1 = pickle.load(col_1)
    with open("models/glu_bmi_dual/cols_2", "rb") as col_2:
        cols_2 = pickle.load(col_2)
    data_1 = glucose_data_transform(data)
    data_2 = bmi_data_transform(data)
    data_1 = data_1[cols_1]
    data_2 = data_2[cols_2]
    return data_1, data_2


def hyp_glu_bmi_transformation(data):
    with open("models/glu_bmi_hyp/cols_1", "rb") as col_1:
        cols_1 = pickle.load(col_1)
    with open("models/glu_bmi_hyp/cols_2", "rb") as col_2:
        cols_2 = pickle.load(col_2)
    with open("models/glu_bmi_hyp/cols_3", "rb") as col_2:
        cols_3 = pickle.load(col_2)
    data_1 = glucose_data_transform(data)
    data_2 = hypertension_data_transform(data)
    data_3 = bmi_data_transform(data)
    data_1 = data_1[cols_1]
    data_2 = data_2[cols_2]
    data_3 = data_3[cols_3]
    return data_1, data_2, data_3
