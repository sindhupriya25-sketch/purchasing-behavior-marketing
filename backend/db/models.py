import os
from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()


class RawCustomer(Base):
    __tablename__ = "raw_customers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    year_birth = Column(Integer)
    education = Column(String)
    marital_status = Column(String)
    income = Column(Float)
    kidhome = Column(Integer)
    teenhome = Column(Integer)
    dt_customer = Column(String)
    recency = Column(Integer)
    mnt_wines = Column(Float)
    mnt_fruits = Column(Float)
    mnt_meat_products = Column(Float)
    mnt_fish_products = Column(Float)
    mnt_sweet_products = Column(Float)
    mnt_gold_prods = Column(Float)
    num_web_purchases = Column(Integer)
    num_catalog_purchases = Column(Integer)
    num_store_purchases = Column(Integer)
    num_web_visits_month = Column(Integer)
    accepted_cmp1 = Column(Integer)
    accepted_cmp2 = Column(Integer)
    accepted_cmp3 = Column(Integer)
    accepted_cmp4 = Column(Integer)
    accepted_cmp5 = Column(Integer)
    response = Column(Integer)
    complain = Column(Integer)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)


class ProcessedCustomer(Base):
    __tablename__ = "processed_customers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    raw_customer_id = Column(Integer, ForeignKey("raw_customers.id"))

    age = Column(Integer)
    income = Column(Float)
    total_spend = Column(Float)
    total_purchases = Column(Integer)
    total_kids = Column(Integer)
    customer_tenure_days = Column(Integer)
    recency = Column(Integer)
    num_web_visits_month = Column(Integer)

    education = Column(String)
    marital_status = Column(String)

    response = Column(Integer)
    processed_timestamp = Column(DateTime, default=datetime.utcnow)


class ClusterAssignment(Base):
    __tablename__ = "cluster_assignments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    processed_customer_id = Column(Integer, ForeignKey("processed_customers.id"))
    cluster_label = Column(Integer)
    k_value = Column(Integer)
    run_timestamp = Column(DateTime, default=datetime.utcnow)


class ModelRun(Base):
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String)
    used_cluster_feature = Column(Boolean)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    roc_auc = Column(Float)
    run_timestamp = Column(DateTime, default=datetime.utcnow)


# ---- Database connection setup (path-safe for deployment) ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use PostgreSQL if DATABASE_URL env variable is set (production/Render),
# otherwise fall back to local SQLite (for local development)
DATABASE_URL = os.environ.get("DATABASE_URL")

if DATABASE_URL:
    # Render's Postgres URLs sometimes start with "postgres://" but SQLAlchemy needs "postgresql://"
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    engine = create_engine(DATABASE_URL)
else:
    DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, '..', 'purchasing_behavior.db')}"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Creates all tables in the database if they don't already exist."""
    Base.metadata.create_all(bind=engine)