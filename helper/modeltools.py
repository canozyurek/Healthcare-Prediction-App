import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import sklearn.compose
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    log_loss,
    mean_squared_error,
    r2_score,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import (
    cross_validate,
    RepeatedStratifiedKFold,
    RandomizedSearchCV,
)
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.exceptions import NotFittedError


def make_preprocessor(data, scaler=True):
    cat_cols = data.select_dtypes(include=["object", "category"]).columns
    num_cols = data.select_dtypes(include=["int", "float"]).columns

    if scaler:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first")),
        ]
    )
    preprocessor = sklearn.compose.ColumnTransformer(
        transformers=[
            ("numeric", num_pipe, num_cols),
            ("categorical", cat_pipe, cat_cols),
        ]
    )
    return preprocessor


def return_col_names(data, preprocessor):
    try:
        cat_col_names = list(
            preprocessor.transformers_[1][1]
            .named_steps["encoder"]
            .get_feature_names_out()
        )
    except NotFittedError:
        cat_col_names = []
    num_cols = list(preprocessor.transformers_[0][2])
    col_names = num_cols + cat_col_names
    final_data = pd.DataFrame(data, columns=col_names)
    return final_data


def cross_validate_model(
    classifiers, X_train, y_train, pipeline=True, task="classification", scaler=True
):
    preprocessor = make_preprocessor(X_train, scaler=scaler)
    if task == "classification":
        score_table = pd.DataFrame()

        if pipeline:
            for i in range(len(classifiers)):
                pipeline = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor),
                        ("classifier", classifiers[i]),
                    ]
                )

                model = pipeline.fit(X_train, y_train)
                predictions = model.predict(X_train)
                pred_proba = model.predict_proba(X_train)

                score_table.loc[i, "model"] = str(classifiers[i])
                score_table.loc[i, "train_accuracy"] = accuracy_score(
                    y_train, predictions
                )
                score_table.loc[i, "train_recall"] = recall_score(
                    y_train, predictions, zero_division=False
                )
                score_table.loc[i, "train_precision"] = precision_score(
                    y_train, predictions, zero_division=False
                )
                score_table.loc[i, "train_log_loss"] = log_loss(
                    y_train, pred_proba, normalize=True
                )
                kfold = RepeatedStratifiedKFold(
                    n_splits=10, n_repeats=3, random_state=1
                )
                cv = cross_validate(
                    pipeline,
                    X_train,
                    y=y_train,
                    cv=kfold,
                    scoring=[
                        "accuracy",
                        "recall",
                        "precision",
                        "f1",
                        "neg_log_loss",
                        "roc_auc",
                    ],
                )
                for j in cv.keys():
                    score_table.loc[i, j] = np.nanmean(cv[j])
                    score_table.loc[i, f"{j}_std"] = np.std(cv[j])

        else:
            X_train = preprocessor.fit_transform(X_train)
            for i in range(len(classifiers)):
                model = classifiers[i]
                model.fit(X_train, y_train)
                predictions = model.predict(X_train)
                pred_proba = model.predict_proba(X_train)

                score_table.loc[i, "model"] = str(classifiers[i])
                score_table.loc[i, "train_accuracy"] = accuracy_score(
                    y_train, predictions
                )
                score_table.loc[i, "train_recall"] = recall_score(
                    y_train, predictions, zero_division=False
                )
                score_table.loc[i, "train_precision"] = precision_score(
                    y_train, predictions, zero_division=False
                )
                score_table.loc[i, "train_log_loss"] = log_loss(
                    y_train, pred_proba, normalize=True
                )
                kfold = RepeatedStratifiedKFold(
                    n_splits=10, n_repeats=3, random_state=1
                )
                cv = cross_validate(
                    model,
                    X_train,
                    y=y_train,
                    cv=kfold,
                    scoring=[
                        "accuracy",
                        "recall",
                        "precision",
                        "f1",
                        "neg_log_loss",
                        "roc_auc",
                    ],
                )
                for j in cv.keys():
                    score_table.loc[i, j] = np.nanmean(cv[j])
                    score_table.loc[i, f"{j}_std"] = np.std(cv[j])

        return score_table

    elif task == "regression":
        score_table = pd.DataFrame()

        if pipeline:
            for i in range(len(classifiers)):
                pipeline = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor),
                        ("classifier", classifiers[i]),
                    ]
                )

                model = pipeline.fit(X_train, y_train)
                predictions = model.predict(X_train)

                score_table.loc[i, "model"] = str(classifiers[i])
                score_table.loc[i, "train_mse"] = mean_squared_error(
                    y_train, predictions
                )
                score_table.loc[i, "train_rmse"] = mean_squared_error(
                    y_train, predictions, squared=False
                )
                score_table.loc[i, "train_r2"] = r2_score(y_train, predictions)

                cv = cross_validate(
                    pipeline,
                    X_train,
                    y=y_train,
                    cv=10,
                    scoring=[
                        "neg_mean_squared_error",
                        "neg_root_mean_squared_error",
                        "r2",
                    ],
                )
                for j in cv.keys():
                    score_table.loc[i, j] = np.nanmean(cv[j])
                    score_table.loc[i, f"{j}_std"] = np.std(cv[j])

        else:
            X_train = preprocessor.fit_transform(X_train)
            for i in range(len(classifiers)):
                model = classifiers[i]
                model.fit(X_train, y_train)
                predictions = model.predict(X_train)

                score_table.loc[i, "model"] = str(classifiers[i])
                score_table.loc[i, "train_mse"] = mean_squared_error(
                    y_train, predictions
                )
                score_table.loc[i, "train_rmse"] = mean_squared_error(
                    y_train, predictions, squared=False
                )
                score_table.loc[i, "train_r2"] = r2_score(y_train, predictions)

                cv = cross_validate(
                    model,
                    X_train,
                    y=y_train,
                    cv=10,
                    scoring=[
                        "neg_mean_squared_error",
                        "neg_root_mean_squared_error",
                        "r2",
                    ],
                )
                for j in cv.keys():
                    score_table.loc[i, j] = np.nanmean(cv[j])
                    score_table.loc[i, f"{j}_std"] = np.std(cv[j])

        return score_table


def shap_feat_reduce_regression(model, X_train, y_train, feats_no):
    train_set = X_train.copy()
    train_label = y_train.copy()
    for i in range(feats_no + 1):
        model.fit(train_set, train_label)
        ex = shap.Explainer(model, train_set)
        shap_values = ex.shap_values(train_set)
        vals = np.abs(shap_values).mean(0)

        feature_importance = pd.DataFrame(
            list(zip(train_set.columns, vals)),
            columns=["col_name", "feature_importance_vals"],
        )
        feature_importance.sort_values(by=["feature_importance_vals"], inplace=True)
        drop_feat = feature_importance.head(1)["col_name"]
        train_set.drop(drop_feat, axis=1, inplace=True)

    cross_val_feature_test = cross_validate_model(
        [model], train_set, y_train=train_label, pipeline=False, task="regression"
    )
    return cross_val_feature_test, train_set


def shap_feat_reduce_classification(model, X_train, y_train, feats_no):
    train_set = X_train.copy()
    train_label = y_train.copy()
    for i in range(feats_no + 1):
        model.fit(train_set, train_label)
        shap_values = shap.TreeExplainer(
            model, feature_names=train_set.columns
        ).shap_values(train_set)
        vals = np.abs(shap_values).mean(0)

        feature_importance = pd.DataFrame(
            list(zip(train_set.columns, vals)),
            columns=["col_name", "feature_importance_vals"],
        )
        feature_importance.sort_values(by=["feature_importance_vals"], inplace=True)
        drop_feat = feature_importance.head(1)["col_name"]
        train_set.drop(drop_feat, axis=1, inplace=True)

    cross_val_feature_test = cross_validate_model(
        [model], train_set, y_train=train_label, pipeline=False
    )
    return cross_val_feature_test, train_set


def shap_feat_viz_regression(model, X_train, y_train, shap_explainer):
    train_mse = []
    test_mse = []
    train_rmse = []
    test_rmse = []
    train_r2 = []
    test_r2 = []

    train_set = X_train.copy()
    train_label = y_train.copy()

    for i in range(len(train_set.columns) - 2):
        if shap_explainer == "tree":
            shap_values = shap.TreeExplainer(
                model, feature_names=train_set.columns
            ).shap_values(train_set)
        elif shap_explainer == "linear":
            model.fit(train_set, train_label)
            ex = shap.Explainer(model, train_set)
            shap_values = ex.shap_values(train_set)
        elif shap_explainer == "kernel":
            model.fit(train_set, train_label)
            ex = shap.KernelExplainer(model, train_set)
            shap_values = ex.shap_values(train_set)
        vals = np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame(
            list(zip(train_set.columns, vals)),
            columns=["col_name", "feature_importance_vals"],
        )
        feature_importance.sort_values(by=["feature_importance_vals"], inplace=True)
        drop_feat = feature_importance.head(1)["col_name"]
        train_set.drop(drop_feat, axis=1, inplace=True)

        cross_val_feature_test = cross_validate_model(
            [model], train_set, y_train=train_label, pipeline=False, task="regression"
        )
        train_mse.append(cross_val_feature_test["train_mse"].values[0])
        test_mse.append(
            -cross_val_feature_test["test_neg_mean_squared_error"].values[0]
        )
        train_rmse.append(cross_val_feature_test["train_rmse"].values[0])
        test_rmse.append(
            -cross_val_feature_test["test_neg_root_mean_squared_error"].values[0]
        )
        train_r2.append(cross_val_feature_test["train_r2"].values[0])
        test_r2.append(cross_val_feature_test["test_r2"].values[0])

    feature_stats = {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "test_r2": test_r2,
    }

    feature_drop_stats = pd.DataFrame(feature_stats)

    fig, ax = plt.subplots(1, 3, figsize=(15, 8))
    sns.lineplot(
        x=feature_drop_stats.index,
        y="train_mse",
        data=feature_drop_stats,
        ax=ax[0],
        label="train",
    )
    sns.lineplot(
        x=feature_drop_stats.index,
        y="test_mse",
        data=feature_drop_stats,
        ax=ax[0],
        label="test",
    )
    sns.lineplot(
        x=feature_drop_stats.index,
        y="train_rmse",
        data=feature_drop_stats,
        ax=ax[1],
        label="train",
    )
    sns.lineplot(
        x=feature_drop_stats.index,
        y="test_rmse",
        data=feature_drop_stats,
        ax=ax[1],
        label="test",
    )
    sns.lineplot(
        x=feature_drop_stats.index,
        y="train_r2",
        data=feature_drop_stats,
        ax=ax[2],
        label="train",
    )
    sns.lineplot(
        x=feature_drop_stats.index,
        y="test_r2",
        data=feature_drop_stats,
        ax=ax[2],
        label="test",
    )
    ax[0].legend()

    print(f"Minimum test mse: {feature_drop_stats['test_mse'].argmin()}")
    print(f"Minimum test rmse: {feature_drop_stats['test_rmse'].argmin()}")
    print(f"Maximum test r2: {feature_drop_stats['test_r2'].argmax()}")


def shap_feat_viz_classification(model, X_train, y_train, shap_explainer):
    train_log_loss = []
    test_log_loss = []
    train_precision = []
    test_precision = []
    train_recall = []
    test_recall = []

    train_set = X_train.copy()
    train_label = y_train.copy()

    for i in range(len(train_set.columns) - 2):
        if shap_explainer == "tree":
            shap_values = shap.TreeExplainer(
                model, feature_names=train_set.columns
            ).shap_values(train_set)
        elif shap_explainer == "linear":
            model.fit(train_set, train_label)
            ex = shap.Explainer(model, train_set)
            shap_values = ex.shap_values(train_set)
        elif shap_explainer == "kernel":
            model.fit(train_set, train_label)
            ex = shap.KernelExplainer(model, train_set)
            shap_values = ex.shap_values(train_set)
        vals = np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame(
            list(zip(train_set.columns, vals)),
            columns=["col_name", "feature_importance_vals"],
        )
        feature_importance.sort_values(by=["feature_importance_vals"], inplace=True)
        drop_feat = feature_importance.head(1)["col_name"]
        train_set.drop(drop_feat, axis=1, inplace=True)

        cross_val_feature_test = cross_validate_model(
            [model], train_set, y_train=train_label, pipeline=False
        )
        train_log_loss.append(cross_val_feature_test["train_log_loss"].values[0])
        test_log_loss.append(-cross_val_feature_test["test_neg_log_loss"].values[0])
        train_precision.append(cross_val_feature_test["train_precision"].values[0])
        test_precision.append(cross_val_feature_test["test_precision"].values[0])
        train_recall.append(cross_val_feature_test["train_recall"].values[0])
        test_recall.append(cross_val_feature_test["test_recall"].values[0])

    feature_stats = {
        "train_log_loss": train_log_loss,
        "test_log_loss": test_log_loss,
        "train_precision": train_precision,
        "test_precision": test_precision,
        "train_recall": train_recall,
        "test_recall": test_recall,
    }

    feature_drop_stats = pd.DataFrame(feature_stats)

    fig, ax = plt.subplots(1, 3, figsize=(15, 8))
    sns.lineplot(
        x=feature_drop_stats.index,
        y="train_log_loss",
        data=feature_drop_stats,
        ax=ax[0],
        label="train",
    )
    sns.lineplot(
        x=feature_drop_stats.index,
        y="test_log_loss",
        data=feature_drop_stats,
        ax=ax[0],
        label="test",
    )
    sns.lineplot(
        x=feature_drop_stats.index,
        y="train_recall",
        data=feature_drop_stats,
        ax=ax[1],
        label="train",
    )
    sns.lineplot(
        x=feature_drop_stats.index,
        y="test_recall",
        data=feature_drop_stats,
        ax=ax[1],
        label="test",
    )
    sns.lineplot(
        x=feature_drop_stats.index,
        y="train_precision",
        data=feature_drop_stats,
        ax=ax[2],
        label="train",
    )
    sns.lineplot(
        x=feature_drop_stats.index,
        y="test_precision",
        data=feature_drop_stats,
        ax=ax[2],
        label="test",
    )
    ax[0].legend()

    print(f"Minimum test log loss: {feature_drop_stats['test_log_loss'].argmin()}")
    print(f"Maximum test recall: {feature_drop_stats['test_recall'].argmax()}")
    print(f"Maximum test precision: {feature_drop_stats['test_precision'].argmax()}")


def shap_viz(model, X_train, y_train, kind, scaled=True):
    print(model)
    preprocessor = make_preprocessor(X_train, scaler=scaled)
    X_train = preprocessor.fit_transform(X_train)
    X_train = return_col_names(X_train, preprocessor)
    model.fit(X_train, y_train)
    if kind == "tree":
        shap_values = shap.TreeExplainer(
            model, feature_names=X_train.columns
        ).shap_values(X_train)
        shap.summary_plot(
            shap_values, X_train, feature_names=X_train.columns, max_display=100
        )
    elif kind == "linear":
        model.fit(X_train, y_train)
        ex = shap.Explainer(model, X_train)
        shap_values = ex.shap_values(X_train)
        shap.summary_plot(
            shap_values, X_train, feature_names=X_train.columns, max_display=100
        )
    elif kind == "kernel":
        model.fit(X_train, y_train)
        ex = shap.KernelExplainer(model.predict, X_train)
        shap_values = ex.shap_values(X_train)
        shap.summary_plot(
            shap_values, X_train, feature_names=X_train.columns, max_display=100
        )
    return shap_values


def randomized_tuning(
    classifier,
    params,
    X_train,
    y_train,
    pipeline=True,
    n_iters=50,
    task="classification",
    scaled=True,
):
    preprocessor = make_preprocessor(X_train, scaler=scaled)
    if task == "classification" and pipeline:

        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", classifier)]
        )

        clf = RandomizedSearchCV(
            pipeline,
            param_distributions=params,
            random_state=42,
            n_iter=n_iters,
            cv=10,
            scoring=["accuracy", "recall", "precision", "f1", "roc_auc"],
            n_jobs=-1,
            refit=False,
        )
        cv = clf.fit(X_train, y_train)
        return pd.DataFrame(cv.cv_results_)

    elif task == "classification" and pipeline is False:

        clf = RandomizedSearchCV(
            classifier,
            param_distributions=params,
            random_state=42,
            n_iter=n_iters,
            cv=10,
            scoring=["accuracy", "recall", "precision", "f1", "roc_auc"],
            n_jobs=-1,
            refit=False,
        )
        cv = clf.fit(X_train, y_train)
        return pd.DataFrame(cv.cv_results_)
    elif task == "regression" and pipeline:

        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", classifier)]
        )

        clf = RandomizedSearchCV(
            pipeline,
            param_distributions=params,
            random_state=42,
            n_iter=n_iters,
            cv=10,
            scoring=["neg_mean_squared_error", "neg_root_mean_squared_error", "r2"],
            n_jobs=-1,
            refit=False,
        )
        cv = clf.fit(X_train, y_train)
        return pd.DataFrame(cv.cv_results_)

    elif task == "regression" and pipeline is False:

        clf = RandomizedSearchCV(
            classifier,
            param_distributions=params,
            random_state=42,
            n_iter=n_iters,
            cv=10,
            scoring=["neg_mean_squared_error", "neg_root_mean_squared_error", "r2"],
            n_jobs=-1,
            refit=False,
        )
        cv = clf.fit(X_train, y_train)
        return pd.DataFrame(cv.cv_results_)
