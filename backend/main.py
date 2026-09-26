import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from db.models import init_db, SessionLocal, RawCustomer, ProcessedCustomer, ClusterAssignment, ModelRun
from ml.preprocessing import DataPreprocessor
from ml.clustering import CustomerClusterer
from ml.classification import ChurnResponseClassifier
from load_data import load_csv_to_db

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

app = FastAPI(title="Purchasing Behavior for Targeted Marketing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()


@app.get("/")
def root():
    return {"message": "Purchasing Behavior for Targeted Marketing API is running"}


@app.post("/load-data")
def load_data_endpoint():
    """Loads the CSV file into the raw_customers table (safe to call multiple times)."""
    try:
        load_csv_to_db()
        session = SessionLocal()
        count = session.query(RawCustomer).count()
        session.close()
        return {"message": "Data loading completed", "total_raw_customers": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/summary")
def get_data_summary():
    session = SessionLocal()
    total_raw = session.query(RawCustomer).count()
    total_processed = session.query(ProcessedCustomer).count()
    total_clusters = session.query(ClusterAssignment).count()
    session.close()
    return {
        "raw_customers": total_raw,
        "processed_customers": total_processed,
        "cluster_assignments": total_clusters,
    }


@app.post("/preprocess")
def run_preprocessing():
    preprocessor = DataPreprocessor()
    df = preprocessor.run_pipeline()
    return {
        "message": "Preprocessing completed",
        "rows_processed": len(df),
    }


@app.get("/clustering/elbow")
def get_elbow_results(k_min: int = 2, k_max: int = 10):
    clusterer = CustomerClusterer()
    df = clusterer.load_processed_data()
    results_df = clusterer.find_optimal_k(df, k_range=range(k_min, k_max + 1))
    return results_df.to_dict(orient="records")


@app.post("/clustering/train")
def train_clustering(k: int = 4):
    clusterer = CustomerClusterer()
    df = clusterer.load_processed_data()
    result_df = clusterer.train(df, k)

    profiles = result_df.groupby("cluster_label")[clusterer.FEATURE_COLUMNS].mean().reset_index()
    return {
        "message": f"Clustering trained with K={k}",
        "rows_clustered": len(result_df),
        "cluster_profiles": profiles.to_dict(orient="records"),
    }


@app.post("/classification/train")
def train_classification(include_cluster: bool = False, model_type: str = "random_forest"):
    clf = ChurnResponseClassifier()
    df = clf.load_data(include_cluster=include_cluster)
    metrics = clf.train_and_evaluate(df, include_cluster=include_cluster, model_type=model_type)
    return {
        "model_type": model_type,
        "used_cluster_feature": include_cluster,
        "metrics": metrics,
    }


@app.get("/experiments")
def get_experiments():
    session = SessionLocal()
    runs = session.query(ModelRun).all()
    result = [
        {
            "id": r.id,
            "model_name": r.model_name,
            "used_cluster_feature": r.used_cluster_feature,
            "accuracy": r.accuracy,
            "precision": r.precision,
            "recall": r.recall,
            "f1_score": r.f1_score,
            "roc_auc": r.roc_auc,
            "run_timestamp": r.run_timestamp.isoformat() if r.run_timestamp else None,
        }
        for r in runs
    ]
    session.close()
    return result


class CustomerInput(BaseModel):
    age: int
    income: float
    total_spend: float
    total_purchases: int
    total_kids: int
    recency: int
    num_web_visits_month: int


@app.post("/predict")
def predict_customer(customer: CustomerInput):
    try:
        kmeans_model = joblib.load(os.path.join(MODELS_DIR, "kmeans_model.joblib"))
        kmeans_scaler = joblib.load(os.path.join(MODELS_DIR, "kmeans_scaler.joblib"))
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Clustering model not found. Train clustering first.")

    try:
        clf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest_with_cluster.joblib"))
        clf_scaler = joblib.load(os.path.join(MODELS_DIR, "random_forest_with_cluster_scaler.joblib"))
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Classifier not found. Train classification with cluster feature first.")

    cluster_features = np.array([[
        customer.age, customer.income, customer.total_spend,
        customer.total_purchases, customer.total_kids,
        customer.recency, customer.num_web_visits_month
    ]])
    cluster_features_scaled = kmeans_scaler.transform(cluster_features)
    predicted_cluster = int(kmeans_model.predict(cluster_features_scaled)[0])

    clf_features = np.array([[
        customer.age, customer.income, customer.total_spend,
        customer.total_purchases, customer.total_kids,
        customer.recency, customer.num_web_visits_month,
        predicted_cluster
    ]])
    clf_features_scaled = clf_scaler.transform(clf_features)
    response_probability = float(clf_model.predict_proba(clf_features_scaled)[0][1])
    predicted_label = int(response_probability >= 0.5)

    if response_probability >= 0.6:
        recommendation = "High likelihood to respond — prioritize for the campaign with a premium offer."
    elif response_probability >= 0.3:
        recommendation = "Moderate likelihood — consider a targeted discount to nudge conversion."
    else:
        recommendation = "Low likelihood — deprioritize for this campaign, or use a low-cost reactivation email."

    return {
        "predicted_cluster": predicted_cluster,
        "response_probability": round(response_probability, 4),
        "predicted_label": predicted_label,
        "recommendation": recommendation,
    }