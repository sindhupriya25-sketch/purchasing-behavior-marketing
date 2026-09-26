import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib

from db.models import SessionLocal, ProcessedCustomer, ClusterAssignment, init_db

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "..", "models")


class CustomerClusterer:
    """
    Performs unsupervised customer segmentation using K-Means.
    Trained WITHOUT the target variable (response) to avoid data leakage.
    """

    FEATURE_COLUMNS = [
        "age", "income", "total_spend", "total_purchases",
        "total_kids", "recency", "num_web_visits_month"
    ]

    def __init__(self):
        self.session = SessionLocal()
        self.scaler = StandardScaler()

    def load_processed_data(self):
        rows = self.session.query(ProcessedCustomer).all()
        data = []
        for r in rows:
            data.append({
                "id": r.id,
                "age": r.age,
                "income": r.income,
                "total_spend": r.total_spend,
                "total_purchases": r.total_purchases,
                "total_kids": r.total_kids,
                "recency": r.recency,
                "num_web_visits_month": r.num_web_visits_month,
            })
        df = pd.DataFrame(data)
        df = df.dropna()
        return df

    def find_optimal_k(self, df, k_range=range(2, 11)):
        X = self.scaler.fit_transform(df[self.FEATURE_COLUMNS])

        results = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            inertia = km.inertia_
            sil_score = silhouette_score(X, labels)
            results.append({"k": k, "inertia": inertia, "silhouette_score": sil_score})
            print(f"K={k}: inertia={inertia:.2f}, silhouette_score={sil_score:.4f}")

        return pd.DataFrame(results)

    def train(self, df, k):
        X = self.scaler.fit_transform(df[self.FEATURE_COLUMNS])

        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)

        df["cluster_label"] = labels

        self.session.query(ClusterAssignment).delete()
        self.session.commit()

        for _, row in df.iterrows():
            assignment = ClusterAssignment(
                processed_customer_id=row["id"],
                cluster_label=int(row["cluster_label"]),
                k_value=k,
            )
            self.session.add(assignment)

        self.session.commit()

        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(model, os.path.join(MODELS_DIR, "kmeans_model.joblib"))
        joblib.dump(self.scaler, os.path.join(MODELS_DIR, "kmeans_scaler.joblib"))

        print(f"\nTrained KMeans with K={k}. Saved {len(df)} cluster assignments.")
        print("\nCluster profiles (mean values per cluster):")
        print(df.groupby("cluster_label")[self.FEATURE_COLUMNS].mean())

        self.session.close()
        return df


if __name__ == "__main__":
    init_db()
    clusterer = CustomerClusterer()
    df = clusterer.load_processed_data()
    print(f"Loaded {len(df)} processed rows for clustering.\n")

    print("Finding optimal K (testing K=2 to 10)...")
    k_results = clusterer.find_optimal_k(df)

    chosen_k = 4
    print(f"\nTraining final model with K={chosen_k}...")
    clusterer.train(df, chosen_k)