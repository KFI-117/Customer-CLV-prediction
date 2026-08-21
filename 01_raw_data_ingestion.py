import sqlite3
import pandas as pd


# 1. RAW DATA INGESTION
excel_file = "online_retail_II.xlsx"

# Reading both sheets (2009-2010 and 2010-2011)
df_09_10 = pd.read_excel(excel_file, sheet_name="Year 2009-2010")
df_10_11 = pd.read_excel(excel_file, sheet_name="Year 2010-2011")

# Combining sheets into a single raw DataFrame
df_raw = pd.concat([df_09_10, df_10_11], ignore_index=True)

# Connecting to SQLite database (creates 'online_retail.db' automatically)
conn = sqlite3.connect("online_retail.db")

# Ingesting raw uncleaned data into SQL
df_raw.to_sql("raw_transactions", conn, if_exists="replace", index=False)

# 2. STAR SCHEMA CREATION VIA SQL
cursor = conn.cursor()

cursor.executescript("""
-- Fact Table: Clean Order Items
DROP TABLE IF EXISTS fact_orders;
CREATE TABLE fact_orders AS
SELECT 
    Invoice,
    StockCode,
    Description,
    `Customer ID` AS CustomerID,
    InvoiceDate,
    Quantity,
    Price,
    (Quantity * Price) AS TotalPrice,
    CASE WHEN Invoice LIKE 'C%' THEN 1 ELSE 0 END AS IsCancelled
FROM raw_transactions
WHERE `Customer ID` IS NOT NULL
  AND Price > 0;

-- Dimension Table: Unique Customers & Acquisition Context
DROP TABLE IF EXISTS dim_customer;
CREATE TABLE dim_customer AS
SELECT 
    `Customer ID` AS CustomerID,
    Country,
    MIN(InvoiceDate) AS FirstAcquisitionDate
FROM raw_transactions
WHERE `Customer ID` IS NOT NULL
GROUP BY `Customer ID`, Country;

-- Dimension Table: Unique Products Metadata
DROP TABLE IF EXISTS dim_product;
CREATE TABLE dim_product AS
SELECT 
    StockCode,
    MAX(Description) AS Description,
    ROUND(AVG(Price), 2) AS AvgUnitPrice
FROM raw_transactions
WHERE StockCode IS NOT NULL
GROUP BY StockCode;
""")

# 3. DATA QUALITY AUDIT REPORT
audit_query = """
SELECT 
    (SELECT COUNT(*) FROM raw_transactions) AS Total_Raw_Rows,
    (SELECT COUNT(*) FROM raw_transactions WHERE `Customer ID` IS NULL) AS Missing_CustomerID_Rows,
    (SELECT COUNT(*) FROM raw_transactions WHERE Invoice LIKE 'C%') AS Cancelled_Invoices_Rows,
    (SELECT COUNT(*) FROM fact_orders WHERE IsCancelled = 0 AND Quantity > 0) AS Valid_Sales_Rows;
"""

audit_df = pd.read_sql_query(audit_query, conn)
print("=== DATA QUALITY AUDIT REPORT ===")
print(audit_df.T)

conn.commit()
conn.close()
print("\n[SUCCESS] Step 1 complete. Database initialized and Star Schema created!")