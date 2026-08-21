import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Connect to database
conn = sqlite3.connect("online_retail.db")

# Corrected RFM Extraction SQL Query (InvoiceNo replaced with Invoice)
rfm_sql = """
WITH DatasetBounds AS (
    SELECT MAX(InvoiceDate) AS MaxDate FROM fact_orders
),
CutoffConfig AS (
    SELECT 
        MaxDate,
        DATE(MaxDate, '-90 days') AS CutoffDate
    FROM DatasetBounds
),
ObservationPeriod AS (
    SELECT 
        f.CustomerID,
        f.Invoice,
        f.InvoiceDate,
        f.TotalPrice
    FROM fact_orders f
    CROSS JOIN CutoffConfig c
    WHERE f.InvoiceDate < c.CutoffDate
      AND f.IsCancelled = 0 
      AND f.Quantity > 0
)
SELECT 
    o.CustomerID,
    CAST(JULIANDAY(c.CutoffDate) - JULIANDAY(MAX(o.InvoiceDate)) AS INTEGER) AS Recency,
    COUNT(DISTINCT o.Invoice) AS Frequency,
    ROUND(SUM(o.TotalPrice), 2) AS Monetary
FROM ObservationPeriod o
CROSS JOIN CutoffConfig c
GROUP BY o.CustomerID, c.CutoffDate;
"""

# 1. Fetch RFM Features into DataFrame
rfm_df = pd.read_sql_query(rfm_sql, conn)

print(f"[INFO] RFM Features generated for {len(rfm_df)} unique active customers.")

# 2. Save Features as SQLite Table & CSV for Downstream Models
rfm_df.to_sql("dim_customer_rfm", conn, if_exists="replace", index=False)
os.makedirs("outputs", exist_ok=True)
rfm_df.to_csv("outputs/rfm_features.csv", index=False)

conn.close()

# 3. Visualizations: Distribution of R, F, M to check Skewness
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Recency Plot
sns.histplot(rfm_df['Recency'], kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Recency Distribution (Days)')

# Frequency Plot
sns.histplot(rfm_df['Frequency'], kde=True, ax=axes[1], color='salmon')
axes[1].set_title('Frequency Distribution (Order Count)')
axes[1].set_xlim(0, 30)  # Capping x-axis for better visibility due to outliers

# Monetary Plot
sns.histplot(rfm_df['Monetary'], kde=True, ax=axes[2], color='green')
axes[2].set_title('Monetary Distribution ($ Total Spend)')
axes[2].set_xlim(0, 10000) # Capping x-axis for better visibility

plt.suptitle('RFM Feature Distributions & Skewness Check', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/rfm_distribution.png", dpi=300)
plt.show()

print("[SUCCESS] Step 3 complete! Table 'dim_customer_rfm' created in SQLite & plots saved to outputs/rfm_distribution.png")