import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Database Connection & Data Loading
conn = sqlite3.connect("online_retail.db")

# Table creation
rfm_df = pd.read_sql_query("SELECT * FROM dim_customer_rfm", conn)

print(f"[INFO] Loaded {len(rfm_df)} customer RFM records from SQLite.")

# QUANTILES SCORING (1 to 5)
# Recency Score: Lower Days = Higher Score (Reverse scoring)
rfm_df['R_Score'] = pd.qcut(
    rfm_df['Recency'], 
    q=5, 
    labels=[5, 4, 3, 2, 1]
).astype(int)

# Frequency Score: Higher Orders = Higher Score
# Note: Frequency has many duplicate 1s, so we use rank(method='first') to split cleanly
rfm_df['F_Score'] = pd.qcut(
    rfm_df['Frequency'].rank(method='first'), 
    q=5, 
    labels=[1, 2, 3, 4, 5]
).astype(int)

# Monetary Score: Higher Spend = Higher Score
rfm_df['M_Score'] = pd.qcut(
    rfm_df['Monetary'], 
    q=5, 
    labels=[1, 2, 3, 4, 5]
).astype(int)

# Composite RFM String (e.g., '555', '111') and Combined Numerical Score
rfm_df['RFM_Segment'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)
rfm_df['RFM_Score_Sum'] = rfm_df['R_Score'] + rfm_df['F_Score'] + rfm_df['M_Score']


# BUSINESS PERSONA SEGMENTATION MAPPING
def assign_persona(df):
    r = df['R_Score']
    f = df['F_Score']
    
    if r >= 4 and f >= 4:
        return 'Champions'              # Recent + Frequent Buyers
    elif r >= 3 and f >= 3:
        return 'Loyal Customers'         # Consistent Buyers
    elif r >= 4 and f == 1:
        return 'New Customers'           # Joined Recently, low frequency
    elif r >= 3 and f == 2:
        return 'Promising / Potential'   # Recent, moderate frequency
    elif r == 2 and f >= 3:
        return 'At Risk'                 # Bought often in past, but not recently
    elif r == 1 and f >= 4:
        return "Can't Lose Them"         # Used to be high value, hasn't returned
    elif r == 2 and f <= 2:
        return 'About to Sleep'          # Below average recency & frequency
    elif r == 1 and f <= 2:
        return 'Hibernating / Lost'      # Long time no purchase, low frequency
    else:
        return 'Others / Need Attention'

# Applying Persona Function
rfm_df['Customer_Persona'] = rfm_df.apply(assign_persona, axis=1)


# SAVING RESULTS TO DATABASE & CSV
rfm_df.to_sql("dim_customer_segmented", conn, if_exists="replace", index=False)

os.makedirs("outputs", exist_ok=True)
rfm_df.to_csv("outputs/rfm_segmented_customers.csv", index=False)

conn.close()

print("[INFO] Saved 'dim_customer_segmented' table to SQLite.")


# VISUALIZATION: PERSONA DISTRIBUTION
plt.figure(figsize=(12, 6))
palette = sns.color_palette("viridis", len(rfm_df['Customer_Persona'].value_counts()))

ax = sns.countplot(
    data=rfm_df, 
    y='Customer_Persona', 
    order=rfm_df['Customer_Persona'].value_counts().index,
    palette=palette
)

# Add count and percentage labels on bars
total = len(rfm_df)
# Purana loop hatakar bas ye 2 lines likh do:
for container in ax.containers:
    ax.bar_label(container, fmt=lambda x: f'{int(x)} ({x/total*100:.1f}%)', padding=5)

plt.title('Customer Base Breakdown by RFM Persona', fontsize=14, fontweight='bold')
plt.xlabel('Customer Count', fontsize=12)
plt.ylabel('Business Persona Segment', fontsize=12)
plt.xlim(0, max(rfm_df['Customer_Persona'].value_counts()) * 1.15) # Spacing for text
plt.tight_layout()
plt.savefig("outputs/persona_distribution.png", dpi=300)
plt.show()

print("[SUCCESS] Step 4 complete! Plot saved as 'outputs/persona_distribution.png'")