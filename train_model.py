import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("Personal_Finance_Dataset.csv")

# Clean columns
df.columns = df.columns.str.strip().str.lower()

print("Columns:", df.columns.tolist())
print("First 5 rows:\n", df.head())

# Convert date
df["date"] = pd.to_datetime(df["date"], errors='coerce')

# Check nulls
print("Null dates:", df["date"].isna().sum())

# Drop bad rows
df = df.dropna(subset=["date", "amount"])

print("Rows after cleaning:", len(df))

# Extract feature
df["day"] = df["date"].dt.day

X = df[["day"]]
y = df["amount"]

print("X shape:", X.shape)
print("y shape:", y.shape)

# STOP if empty
if len(df) == 0:
    print("❌ Dataset became empty after cleaning")
    exit()

model = LinearRegression()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained successfully!")