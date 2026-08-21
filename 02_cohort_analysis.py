import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Connecting Database
conn = sqlite3.connect("online_retail.db")

# Completing Cohort SQL Query
cohort_sql = """
WITH CustomerFirstPurchase AS (
    SELECT 
        CustomerID,
        MIN(InvoiceDate) AS FirstPurchaseDate
    FROM fact_orders
    WHERE IsCancelled = 0 AND Quantity > 0
    GROUP BY CustomerID
),
CohortBase AS (
    SELECT 
        f.CustomerID,
        DATE(STRFTIME('%Y-%m-01', f.InvoiceDate)) AS TransactionMonth,
        DATE(STRFTIME('%Y-%m-01', c.FirstPurchaseDate)) AS CohortMonth
    FROM fact_orders f
    JOIN CustomerFirstPurchase c ON f.CustomerID = c.CustomerID
    WHERE f.IsCancelled = 0 AND f.Quantity > 0
),
CohortIndexCalculated AS (
    SELECT 
        CustomerID,
        CohortMonth,
        (CAST(STRFTIME('%Y', TransactionMonth) AS INTEGER) - CAST(STRFTIME('%Y', CohortMonth) AS INTEGER)) * 12 +
        (CAST(STRFTIME('%m', TransactionMonth) AS INTEGER) - CAST(STRFTIME('%m', CohortMonth) AS INTEGER)) AS CohortIndex
    FROM CohortBase
)
SELECT 
    CohortMonth,
    CohortIndex,
    COUNT(DISTINCT CustomerID) AS ActiveCustomers
FROM CohortIndexCalculated
GROUP BY CohortMonth, CohortIndex
ORDER BY CohortMonth, CohortIndex;
"""

# Fetching data into Pandas DataFrame
cohort_df = pd.read_sql_query(cohort_sql, conn)
conn.close()

# 1. Pivot Table Creation (Rows = CohortMonth, Columns = CohortIndex, Values = ActiveCustomers)
cohort_pivot = cohort_df.pivot(index='CohortMonth', columns='CohortIndex', values='ActiveCustomers')

# 2. Converting Absolute Customer Counts to Percentages (Retention Rate %)
# Column 0 (CohortIndex = 0) represents the initial total cohort size (100%)
cohort_size = cohort_pivot.iloc[:, 0]
retention_matrix = cohort_pivot.divide(cohort_size, axis=0)

# 3. Plotting Seaborn Heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(
    data=retention_matrix, 
    annot=True, 
    fmt='.0%', 
    cmap='Blues', 
    vmin=0.0, 
    vmax=0.5
)
plt.title('Monthly Customer Retention Cohort Matrix (%)', fontsize=14, fontweight='bold')
plt.xlabel('Cohort Index (Months Since First Purchase)', fontsize=12)
plt.ylabel('Cohort Month (Acquisition Group)', fontsize=12)
plt.tight_layout()
plt.savefig("cohort_retention_heatmap.png", dpi=300)
plt.show()

print("[SUCCESS] Step 2 complete. Heatmap saved as 'cohort_retention_heatmap.png'")