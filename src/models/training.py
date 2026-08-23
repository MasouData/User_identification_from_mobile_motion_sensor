# src/models/training.py

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier

from xgboost import XGBClassifier


def compare_models_repeated(
    X,
    y,
    seeds,
    test_size=0.2,
):
    """
    Compare several classifiers using exactly the same
    stratified train/validation split for each random seed.

    Models
    ------
    - Dummy most-frequent baseline
    - Dummy stratified baseline
    - Logistic Regression
    - Linear SVM
    - Random Forest
    - XGBoost

    Metric
    ------
    Macro-F1

    Parameters
    ----------
    X : pandas.DataFrame
        Session-level feature matrix.

    y : pandas.Series
        User labels.

    seeds : iterable
        Random seeds used to create repeated train/validation splits.

    test_size : float
        Fraction of sessions used for validation.

    Returns
    -------
    dict
        Model name -> list of Macro-F1 scores.
    """

    results = {
        "Dummy_most_frequent": [],
        "Dummy_stratified": [],
        "LR": [],
        "SVM": [],
        "RF": [],
        "XGB": [],
    }

    # XGBoost needs integer-encoded class labels.
    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(y)

    # We split indices once per seed so every model
    # receives exactly the same train/validation data.
    indices = np.arange(len(y))

    for seed in seeds:

        train_idx, valid_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )

        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]

        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        y_train_encoded = y_encoded[train_idx]
        y_valid_encoded = y_encoded[valid_idx]

        # -----------------------------------------------------
        # 1. Dummy: most frequent
        # -----------------------------------------------------

        dummy_most_frequent = DummyClassifier(
            strategy="most_frequent",
            random_state=seed,
        )

        dummy_most_frequent.fit(
            X_train,
            y_train,
        )

        predictions = dummy_most_frequent.predict(
            X_valid
        )

        results["Dummy_most_frequent"].append(
            f1_score(
                y_valid,
                predictions,
                average="macro",
            )
        )

        # -----------------------------------------------------
        # 2. Dummy: stratified
        # -----------------------------------------------------

        dummy_stratified = DummyClassifier(
            strategy="stratified",
            random_state=seed,
        )

        dummy_stratified.fit(
            X_train,
            y_train,
        )

        predictions = dummy_stratified.predict(
            X_valid
        )

        results["Dummy_stratified"].append(
            f1_score(
                y_valid,
                predictions,
                average="macro",
            )
        )

        # -----------------------------------------------------
        # 3. Logistic Regression
        # -----------------------------------------------------

        logistic_regression = Pipeline([
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    C=1.0,
                ),
            ),
        ])

        logistic_regression.fit(
            X_train,
            y_train,
        )

        predictions = logistic_regression.predict(
            X_valid
        )

        results["LR"].append(
            f1_score(
                y_valid,
                predictions,
                average="macro",
            )
        )

        # -----------------------------------------------------
        # 4. Linear SVM
        # -----------------------------------------------------

        svm = Pipeline([
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LinearSVC(
                    C=1.0,
                    random_state=seed,
                ),
            ),
        ])

        svm.fit(
            X_train,
            y_train,
        )

        predictions = svm.predict(
            X_valid
        )

        results["SVM"].append(
            f1_score(
                y_valid,
                predictions,
                average="macro",
            )
        )

        # -----------------------------------------------------
        # 5. Random Forest
        # -----------------------------------------------------

        random_forest = RandomForestClassifier(
            n_estimators=500,
            random_state=seed,
            n_jobs=-1,
        )

        random_forest.fit(
            X_train,
            y_train,
        )

        predictions = random_forest.predict(
            X_valid
        )

        results["RF"].append(
            f1_score(
                y_valid,
                predictions,
                average="macro",
            )
        )

        # -----------------------------------------------------
        # 6. XGBoost
        # -----------------------------------------------------

        xgboost = XGBClassifier(
            objective="multi:softmax",
            num_class=len(
                label_encoder.classes_
            ),
            eval_metric="mlogloss",
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
        )

        xgboost.fit(
            X_train,
            y_train_encoded,
        )

        predictions = xgboost.predict(
            X_valid
        )

        results["XGB"].append(
            f1_score(
                y_valid_encoded,
                predictions,
                average="macro",
            )
        )

    return results


def tune_random_forest(
    X,
    y,
    seeds,
):
    """
    Evaluate a small predefined Random Forest parameter grid
    using repeated stratified train/validation splits.

    Returns
    -------
    tuple
        (
            best_parameters,
            sorted_results
        )
    """

    parameter_grid = [
        {
            "n_estimators": 300,
            "max_depth": None,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
        },
        {
            "n_estimators": 500,
            "max_depth": None,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
        },
        {
            "n_estimators": 500,
            "max_depth": 20,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
        },
        {
            "n_estimators": 500,
            "max_depth": 10,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
        },
        {
            "n_estimators": 500,
            "max_depth": None,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
        },
        {
            "n_estimators": 500,
            "max_depth": None,
            "max_features": "sqrt",
            "min_samples_leaf": 5,
        },
    ]

    results = []

    for parameters in parameter_grid:

        scores = []

        for seed in seeds:

            X_train, X_valid, y_train, y_valid = (
                train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=seed,
                    stratify=y,
                )
            )

            model = RandomForestClassifier(
                random_state=seed,
                class_weight="balanced_subsample",
                n_jobs=-1,
                **parameters,
            )

            model.fit(
                X_train,
                y_train,
            )

            predictions = model.predict(
                X_valid
            )

            score = f1_score(
                y_valid,
                predictions,
                average="macro",
            )

            scores.append(score)

        results.append({
            "params": parameters,
            "mean": float(
                np.mean(scores)
            ),
            "std": float(
                np.std(scores)
            ),
            "scores": scores,
        })

    sorted_results = sorted(
        results,
        key=lambda result: result["mean"],
        reverse=True,
    )

    best_parameters = (
        sorted_results[0]["params"]
    )

    return (
        best_parameters,
        sorted_results,
    )


def train_final_random_forest(
    X,
    y,
    best_parameters,
    random_state=42,
):
    """
    Train the final Random Forest on all available
    training sessions.
    """

    model = RandomForestClassifier(
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
        **best_parameters,
    )

    model.fit(
        X,
        y,
    )

    return model