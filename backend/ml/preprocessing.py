import pandas as pd
from datetime import datetime
from db.models import SessionLocal, RawCustomer, ProcessedCustomer, init_db


class DataPreprocessor:
    """
    Handles cleaning and feature engineering for raw customer data.
    Reads from raw_customers table, writes cleaned+engineered data
    into processed_customers table.
    """

    def __init__(self):
        self.session = SessionLocal()

    def load_raw_data(self):
        """Pull all raw customer rows from the database into a pandas DataFrame."""
        raw_customers = self.session.query(RawCustomer).all()
        data = []
        for c in raw_customers:
            data.append({
                "id": c.id,
                "year_birth": c.year_birth,
                "education": c.education,
                "marital_status": c.marital_status,
                "income": c.income,
                "kidhome": c.kidhome,
                "teenhome": c.teenhome,
                "dt_customer": c.dt_customer,
                "recency": c.recency,
                "mnt_wines": c.mnt_wines,
                "mnt_fruits": c.mnt_fruits,
                "mnt_meat_products": c.mnt_meat_products,
                "mnt_fish_products": c.mnt_fish_products,
                "mnt_sweet_products": c.mnt_sweet_products,
                "mnt_gold_prods": c.mnt_gold_prods,
                "num_web_purchases": c.num_web_purchases,
                "num_catalog_purchases": c.num_catalog_purchases,
                "num_store_purchases": c.num_store_purchases,
                "num_web_visits_month": c.num_web_visits_month,
                "response": c.response,
            })
        return pd.DataFrame(data)

    def clean_data(self, df):
        """Handle missing values and remove unrealistic outliers."""
        # Fill missing income with the median income
        median_income = df["income"].median()
        df["income"] = df["income"].fillna(median_income)

        # Remove unrealistic outliers
        df = df[df["income"] < 200000]          # extreme income outliers
        df = df[df["year_birth"] > 1930]         # unrealistic birth years

        return df

    def engineer_features(self, df):
        """Create new, more meaningful features from raw columns."""
        current_year = datetime.now().year

        df["age"] = current_year - df["year_birth"]

        df["total_spend"] = (
            df["mnt_wines"] + df["mnt_fruits"] + df["mnt_meat_products"] +
            df["mnt_fish_products"] + df["mnt_sweet_products"] + df["mnt_gold_prods"]
        )

        df["total_purchases"] = (
            df["num_web_purchases"] + df["num_catalog_purchases"] + df["num_store_purchases"]
        )

        df["total_kids"] = df["kidhome"] + df["teenhome"]

        # Customer tenure: days since they enrolled
        df["dt_customer_parsed"] = pd.to_datetime(df["dt_customer"], format="%d-%m-%Y", errors="coerce")
        df["customer_tenure_days"] = (datetime.now() - df["dt_customer_parsed"]).dt.days

        return df

    def save_processed_data(self, df):
        """Write the cleaned + engineered data into the processed_customers table."""
        # Clear old processed data first (so re-running doesn't duplicate)
        self.session.query(ProcessedCustomer).delete()
        self.session.commit()

        inserted = 0
        for _, row in df.iterrows():
            processed = ProcessedCustomer(
                raw_customer_id=row["id"],
                age=row["age"],
                income=row["income"],
                total_spend=row["total_spend"],
                total_purchases=row["total_purchases"],
                total_kids=row["total_kids"],
                customer_tenure_days=row["customer_tenure_days"] if pd.notna(row["customer_tenure_days"]) else None,
                recency=row["recency"],
                num_web_visits_month=row["num_web_visits_month"],
                education=row["education"],
                marital_status=row["marital_status"],
                response=row["response"],
            )
            self.session.add(processed)
            inserted += 1

        self.session.commit()
        return inserted

    def run_pipeline(self):
        """Runs the full preprocessing pipeline end to end."""
        print("Loading raw data...")
        df = self.load_raw_data()
        print(f"Loaded {len(df)} raw rows.")

        print("Cleaning data...")
        df = self.clean_data(df)
        print(f"{len(df)} rows remain after cleaning.")

        print("Engineering features...")
        df = self.engineer_features(df)

        print("Saving processed data...")
        count = self.save_processed_data(df)
        print(f"Saved {count} processed rows.")

        self.session.close()
        return df


if __name__ == "__main__":
    init_db()  # ensures the new table gets created
    preprocessor = DataPreprocessor()
    result_df = preprocessor.run_pipeline()
    print("\nSample of processed data:")
    print(result_df[["age", "income", "total_spend", "total_purchases", "total_kids", "customer_tenure_days"]].head())