import argparse
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


RANDOM_STATE = 42


def resolve_default_data_path() -> str:
    # Prefer creditcard.csv in repo root if present; otherwise fall back to extracted folder
    candidates = [
        os.path.join(os.getcwd(), "creditcard.csv"),
        os.path.join(os.getcwd(), "creditcardfraud", "creditcard.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate credit card fraud detection model")
    parser.add_argument("--data", default=resolve_default_data_path(), help="Path to creditcard.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if df.isnull().any().any():
        raise ValueError("Dataset contains null values; please investigate before training.")

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    scale_cols = ["Time", "Amount"]
    other_cols = [c for c in X.columns if c not in scale_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), scale_cols),
            ("passthrough", "passthrough", other_cols),
        ]
    )

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)

    print("5-Fold CV F1 scores:", cv_f1)
    print("Mean CV F1:", float(np.mean(cv_f1)))

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))
    print("AUPRC (Average Precision):", float(average_precision_score(y_test, y_proba)))


if __name__ == "__main__":
    main()
