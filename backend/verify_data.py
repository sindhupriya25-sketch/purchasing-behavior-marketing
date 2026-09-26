from db.models import SessionLocal, RawCustomer

session = SessionLocal()

# Count total rows
total = session.query(RawCustomer).count()
print(f"Total rows in database: {total}")

# Look at the first 3 customers
first_three = session.query(RawCustomer).limit(3).all()
for customer in first_three:
    print(f"ID: {customer.id}, Income: {customer.income}, Education: {customer.education}, "
          f"Marital Status: {customer.marital_status}, Response: {customer.response}")

session.close()