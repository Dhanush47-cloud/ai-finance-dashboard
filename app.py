import streamlit as st
import pandas as pd

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Finance Dashboard", layout="wide")

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("Personal_Finance_Dataset.csv")

# Clean columns
df.columns = df.columns.str.strip().str.lower()

# Clean data
df["type"] = df["type"].str.strip().str.lower()
df["date"] = pd.to_datetime(df["date"])

# ----------------------------
# SIDEBAR FILTERS
# ----------------------------
st.sidebar.title("🔍 Filters")

category_filter = st.sidebar.multiselect(
    "Select Category",
    options=df["category"].unique(),
    default=df["category"].unique()
)

type_filter = st.sidebar.selectbox(
    "Select Type",
    ["all", "expense", "income"]
)

# Apply filters
filtered_df = df[df["category"].isin(category_filter)]

if type_filter != "all":
    filtered_df = filtered_df[filtered_df["type"] == type_filter]

# ----------------------------
# TITLE
# ----------------------------
st.title("💰 AI Personal Finance Dashboard")

# ----------------------------
# METRICS
# ----------------------------
col1, col2 = st.columns(2)

total_expense = filtered_df[filtered_df["type"] == "expense"]["amount"].sum()
total_income = filtered_df[filtered_df["type"] == "income"]["amount"].sum()

col1.metric("💸 Total Expense", f"₹ {round(total_expense,2)}")
col2.metric("💰 Total Income", f"₹ {round(total_income,2)}")

# ----------------------------
# CATEGORY CHART
# ----------------------------
st.subheader("📊 Category-wise Spending")
cat_data = filtered_df[filtered_df["type"] == "expense"].groupby("category")["amount"].sum()
st.bar_chart(cat_data)

# ----------------------------
# MONTHLY TREND
# ----------------------------
st.subheader("📅 Monthly Spending Trend")
monthly = filtered_df[filtered_df["type"] == "expense"].groupby(filtered_df["date"].dt.month)["amount"].sum()
st.line_chart(monthly)

# ----------------------------
# PREDICTION SECTION
# ----------------------------
st.subheader("🔮 Predict Expense")

df["day"] = df["date"].dt.day

day = st.slider("Select Day", 1, 31)
category = st.selectbox("Select Category", df["category"].unique())

# Smart prediction
filtered_pred = df[
    (df["day"] == day) &
    (df["category"] == category) &
    (df["type"] == "expense")
]

if st.button("Predict Expense"):
    if len(filtered_pred) > 0:
        prediction = filtered_pred["amount"].mean()
    else:
        prediction = df[
            (df["category"] == category) &
            (df["type"] == "expense")
        ]["amount"].mean()

    st.success(f"Estimated Expense: ₹ {round(prediction,2)}")

# ----------------------------
# RAW DATA
# ----------------------------
if st.checkbox("📋 Show Raw Data"):
    st.dataframe(filtered_df)