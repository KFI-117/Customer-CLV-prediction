import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Connecting to Database
conn = sqlite3.connect("online_retail.db")

# SQL QUERY: HOLDOUT TARGET EXTRACTION & MASTER DATASET MERGE
target_sql = """
WITH CutoffConfig AS (
    SELECT DATE(MAX(InvoiceDate), '-90 days') AS CutoffDate 
    FROM fact_orders
),
HoldoutActivity AS (
    -- Aggregate actual customer activity in the last 90 days (Holdout Window)
    SELECT 
        f.CustomerID,
        COUNT(DISTINCT f.Invoice) AS HoldoutOrders,
        ROUND(SUM(f.TotalPrice), 2) AS HoldoutMonetary
    FROM fact_orders f
    CROSS JOIN CutoffConfig c
    WHERE f.InvoiceDate >= c.CutoffDate
      AND f.IsCancelled = 0 
      AND f.Quantity > 0
    GROUP BY f.CustomerID
)
SELECT 
    -- Features (X) from Step 4
    s.CustomerID,
    s.Recency,
    s.Frequency,
    s.Monetary,
    s.R_Score,
    s.F_Score,
    s.M_Score,
    s.RFM_Segment,
    s.RFM_Score_Sum,
    s.Customer_Persona,
    
    -- Targets (y) from Holdout Period
    COALESCE(h.HoldoutOrders, 0) AS HoldoutOrders,
    COALESCE(h.HoldoutMonetary, 0.0) AS HoldoutMonetary,
    
    -- Churn Definition: 1 if NO orders in holdout period, else 0
    CASE 
        WHEN COALESCE(h.HoldoutOrders, 0) = 0 THEN 1 
        ELSE 0 
    END AS IsChurned
    
FROM dim_customer_segmented s
LEFT JOIN HoldoutActivity h ON s.CustomerID = h.CustomerID;
"""

# Fetching full master dataset
master_df = pd.read_sql_query(target_sql, conn)

print(f"[INFO] Master ML Dataset generated with {len(master_df)} rows and {len(master_df.columns)} columns.")
print(f"[INFO] Overall Churn Rate in Holdout Window: {master_df['IsChurned'].mean() * 100:.2f}%")

# SAVING MASTER DATASET TO SQLITE & CSV
master_df.to_sql("dataset_churn_clv_gold", conn, if_exists="replace", index=False)

os.makedirs("outputs", exist_ok=True)
master_df.to_csv("outputs/master_ml_dataset.csv", index=False)

conn.close()

# VALIDATION VISUALIZATION: CHURN RATE BY PERSONA
# Business Sanity Check: Champions ka churn sabse kam aur At Risk ka zyada hona chahiye!

persona_churn = master_df.groupby('Customer_Persona')['IsChurned'].agg(
    Total_Customers='count',
    Churned_Count='sum',
    Churn_Rate='mean'
).reset_index().sort_values(by='Churn_Rate', ascending=False)

plt.figure(figsize=(12, 6))
ax = sns.barplot(
    data=persona_churn, 
    x='Churn_Rate', 
    y='Customer_Persona', 
    palette='Reds_r'
)

# Label rendering
for container in ax.containers:
    ax.bar_label(container, fmt=lambda x: f'{x*100:.1f}%', padding=5, fontsize=10)

plt.title('Validation Check: Churn Rate (%) Across RFM Personas', fontsize=14, fontweight='bold')
plt.xlabel('Holdout Churn Rate (0.0 to 1.0)', fontsize=12)
plt.ylabel('Customer Persona', fontsize=12)
plt.xlim(0, 1.1)  # Spacing for text labels
plt.tight_layout()
plt.savefig("outputs/churn_rate_by_persona.png", dpi=300)
plt.show()

print("[SUCCESS] Step 5 complete! Table 'dataset_churn_clv_gold' saved to SQLite & plot saved to 'outputs/churn_rate_by_persona.png'")