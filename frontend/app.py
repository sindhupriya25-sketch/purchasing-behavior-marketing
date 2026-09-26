import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Purchasing Behavior for Targeted Marketing", layout="wide")

st.title("🛍️ Purchasing Behavior for Targeted Marketing")
st.caption("Customer segmentation + campaign response prediction, powered by a live FastAPI backend")

# ---- Data Summary Section ----
st.header("1. Data Summary")

if st.button("Refresh Data Summary"):
    try:
        response = requests.get(f"{API_URL}/data/summary")
        if response.status_code == 200:
            data = response.json()
            col1, col2, col3 = st.columns(3)
            col1.metric("Raw Customers", data["raw_customers"])
            col2.metric("Processed Customers", data["processed_customers"])
            col3.metric("Cluster Assignments", data["cluster_assignments"])
        else:
            st.error(f"Error: {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to backend. Make sure the FastAPI server is running.")

st.divider()

# ---- Preprocessing Section ----
st.header("2. Preprocessing")

if st.button("Run Preprocessing Pipeline"):
    with st.spinner("Cleaning and engineering features..."):
        response = requests.post(f"{API_URL}/preprocess")
        if response.status_code == 200:
            result = response.json()
            st.success(f"Preprocessing complete! {result['rows_processed']} rows processed.")
        else:
            st.error(f"Error: {response.status_code}")

st.divider()

# ---- Clustering Section ----
st.header("3. Customer Segmentation (Clustering)")

if st.button("Find Optimal K (Elbow Method)"):
    with st.spinner("Testing K=2 to 10..."):
        response = requests.get(f"{API_URL}/clustering/elbow")
        if response.status_code == 200:
            elbow_data = pd.DataFrame(response.json())
            st.session_state["elbow_data"] = elbow_data

if "elbow_data" in st.session_state:
    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(st.session_state["elbow_data"].set_index("k")["inertia"])
        st.caption("Inertia vs K (lower isn't always better — look for the 'elbow')")
    with col2:
        st.line_chart(st.session_state["elbow_data"].set_index("k")["silhouette_score"])
        st.caption("Silhouette Score vs K (higher = better separated clusters)")

chosen_k = st.slider("Choose number of clusters (K)", min_value=2, max_value=10, value=4)

if st.button("Train Clustering Model"):
    with st.spinner(f"Training KMeans with K={chosen_k}..."):
        response = requests.post(f"{API_URL}/clustering/train", params={"k": chosen_k})
        if response.status_code == 200:
            result = response.json()
            st.success(result["message"])
            st.subheader("Cluster Profiles (average values per cluster)")
            profiles_df = pd.DataFrame(result["cluster_profiles"])
            st.dataframe(profiles_df, use_container_width=True)
        else:
            st.error(f"Error: {response.status_code}")

st.divider()

# ---- Classification Section ----
st.header("4. Campaign Response Prediction (Classification)")

col1, col2 = st.columns(2)
with col1:
    model_type = st.selectbox("Choose Algorithm", ["random_forest", "logistic_regression"])
with col2:
    include_cluster = st.checkbox("Include Cluster Label as Feature")

if st.button("Train Classifier"):
    with st.spinner("Training classifier..."):
        response = requests.post(
            f"{API_URL}/classification/train",
            params={"include_cluster": include_cluster, "model_type": model_type},
        )
        if response.status_code == 200:
            result = response.json()
            st.success(f"Model trained: {result['model_type']} (cluster feature: {result['used_cluster_feature']})")
            metrics = result["metrics"]
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            m2.metric("Precision", f"{metrics['precision']:.4f}")
            m3.metric("Recall", f"{metrics['recall']:.4f}")
            m4.metric("F1 Score", f"{metrics['f1_score']:.4f}")
            m5.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
        else:
            st.error(f"Error: {response.status_code}")

st.divider()

# ---- Experiment Comparison Section ----
st.header("5. Experiment Comparison (All Training Runs)")

if st.button("Load All Experiments"):
    response = requests.get(f"{API_URL}/experiments")
    if response.status_code == 200:
        experiments_df = pd.DataFrame(response.json())
        st.dataframe(experiments_df, use_container_width=True)

        if len(experiments_df) > 0:
            st.subheader("F1 Score Comparison")
            chart_df = experiments_df.copy()
            chart_df["label"] = chart_df["model_name"] + " (" + chart_df["used_cluster_feature"].astype(str) + ")"
            st.bar_chart(chart_df.set_index("label")["f1_score"])

            st.divider()

# ---- Live Prediction Section ----
st.header("6. Live Prediction — Will This Customer Respond?")

st.write("Enter a hypothetical customer's details to predict their campaign response and get a marketing recommendation.")

col1, col2, col3 = st.columns(3)
with col1:
    input_age = st.number_input("Age", min_value=18, max_value=100, value=45)
    input_income = st.number_input("Income", min_value=0.0, value=50000.0, step=1000.0)
with col2:
    input_total_spend = st.number_input("Total Spend", min_value=0.0, value=500.0, step=50.0)
    input_total_purchases = st.number_input("Total Purchases", min_value=0, value=10)
with col3:
    input_total_kids = st.number_input("Total Kids", min_value=0, max_value=5, value=0)
    input_recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
    input_web_visits = st.number_input("Web Visits per Month", min_value=0, value=5)

if st.button("Predict Response"):
    payload = {
        "age": input_age,
        "income": input_income,
        "total_spend": input_total_spend,
        "total_purchases": input_total_purchases,
        "total_kids": input_total_kids,
        "recency": input_recency,
        "num_web_visits_month": input_web_visits,
    }
    response = requests.post(f"{API_URL}/predict", json=payload)

    if response.status_code == 200:
        result = response.json()
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Cluster", result["predicted_cluster"])
        col2.metric("Response Probability", f"{result['response_probability']:.2%}")
        col3.metric("Predicted Response", "Yes" if result["predicted_label"] == 1 else "No")
        st.info(f"**Recommendation:** {result['recommendation']}")
    else:
        st.error(f"Error: {response.json().get('detail', response.status_code)}")