import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

from db.models import SessionLocal, ProcessedCustomer, ClusterAssignment, ModelRun, init_db

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "..", "models")


class ChurnResponseClassifier:
    """
    Trains a classifier to predict campaign Response (0/1).
    Can optionally include the cluster label as an extra feature.
    """

    BASE_FEATURES = [
        "age", "income", "total_spend", "total_purchases",
        "total_kids", "recency", "num_web_visits_month"
    ]

    def __init__(self):
        self.session = SessionLocal()

    def load_data(self, include_cluster=False):
        processed = self.session.query(ProcessedCustomer).all()
        data = []
        for p in processed:
            row = {
                "id": p.id,
                "age": p.age,
                "income": p.income,
                "total_spend": p.total_spend,
                "total_purchases": p.total_purchases,
                "total_kids": p.total_kids,
                "recency": p.recency,
                "num_web_visits_month": p.num_web_visits_month,
                "response": p.response,
            }
            data.append(row)
        df = pd.DataFrame(data)

        if include_cluster:
            clusters = self.session.query(ClusterAssignment).all()
            cluster_map = {c.processed_customer_id: c.cluster_label for c in clusters}
            df["cluster_label"] = df["id"].map(cluster_map)

        df = df.dropna()
        return df

    def train_and_evaluate(self, df, include_cluster=False, model_type="random_forest"):
        features = self.BASE_FEATURES.copy()
        if include_cluster:
            features.append("cluster_label")

        X = df[features]
        y = df["response"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if model_type == "logistic_regression":
            model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=200, random_state=42)

        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        run = ModelRun(
            model_name=model_type,
            used_cluster_feature=include_cluster,
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1_score=metrics["f1_score"],
            roc_auc=metrics["roc_auc"],
        )
        self.session.add(run)
        self.session.commit()

        os.makedirs(MODELS_DIR, exist_ok=True)
        suffix = "with_cluster" if include_cluster else "without_cluster"
        joblib.dump(model, os.path.join(MODELS_DIR, f"{model_type}_{suffix}.joblib"))
        joblib.dump(scaler, os.path.join(MODELS_DIR, f"{model_type}_{suffix}_scaler.joblib"))

        return metrics


if __name__ == "__main__":
    init_db()
    clf = ChurnResponseClassifier()

    print("=" * 60)
    print("EXPERIMENT 1: WITHOUT cluster feature")
    print("=" * 60)
    df_no_cluster = clf.load_data(include_cluster=False)
    metrics_no_cluster = clf.train_and_evaluate(df_no_cluster, include_cluster=False)
    for k, v in metrics_no_cluster.items():
        print(f"{k}: {v:.4f}")

    print("\n" + "=" * 60)
    print("EXPERIMENT 2: WITH cluster feature")
    print("=" * 60)
    df_with_cluster = clf.load_data(include_cluster=True)
    metrics_with_cluster = clf.train_and_evaluate(df_with_cluster, include_cluster=True)
    for k, v in metrics_with_cluster.items():
        print(f"{k}: {v:.4f}")

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    for metric in metrics_no_cluster:
        diff = metrics_with_cluster[metric] - metrics_no_cluster[metric]
        print(f"{metric}: without={metrics_no_cluster[metric]:.4f}, "
              f"with={metrics_with_cluster[metric]:.4f}, diff={diff:+.4f}")

    clf.session.close()