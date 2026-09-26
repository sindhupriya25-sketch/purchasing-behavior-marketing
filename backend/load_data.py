import os
import pandas as pd
from db.models import init_db, SessionLocal, RawCustomer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_csv_to_db(csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(BASE_DIR, "..", "data", "marketing_campaign.csv")

    init_db()

    df = pd.read_csv(csv_path, sep="\t")

    print(f"CSV loaded successfully. Shape: {df.shape}")
    print(f"Columns found: {list(df.columns)}")

    session = SessionLocal()

    existing_count = session.query(RawCustomer).count()
    if existing_count > 0:
        print(f"Data already loaded ({existing_count} rows). Skipping insert.")
        session.close()
        return

    inserted = 0
    for _, row in df.iterrows():
        customer = RawCustomer(
            year_birth=row.get("Year_Birth"),
            education=row.get("Education"),
            marital_status=row.get("Marital_Status"),
            income=row.get("Income") if pd.notna(row.get("Income")) else None,
            kidhome=row.get("Kidhome"),
            teenhome=row.get("Teenhome"),
            dt_customer=row.get("Dt_Customer"),
            recency=row.get("Recency"),
            mnt_wines=row.get("MntWines"),
            mnt_fruits=row.get("MntFruits"),
            mnt_meat_products=row.get("MntMeatProducts"),
            mnt_fish_products=row.get("MntFishProducts"),
            mnt_sweet_products=row.get("MntSweetProducts"),
            mnt_gold_prods=row.get("MntGoldProds"),
            num_web_purchases=row.get("NumWebPurchases"),
            num_catalog_purchases=row.get("NumCatalogPurchases"),
            num_store_purchases=row.get("NumStorePurchases"),
            num_web_visits_month=row.get("NumWebVisitsMonth"),
            accepted_cmp1=row.get("AcceptedCmp1"),
            accepted_cmp2=row.get("AcceptedCmp2"),
            accepted_cmp3=row.get("AcceptedCmp3"),
            accepted_cmp4=row.get("AcceptedCmp4"),
            accepted_cmp5=row.get("AcceptedCmp5"),
            response=row.get("Response"),
            complain=row.get("Complain"),
        )
        session.add(customer)
        inserted += 1

    session.commit()
    session.close()
    print(f"Successfully inserted {inserted} rows into the database.")


if __name__ == "__main__":
    load_csv_to_db()