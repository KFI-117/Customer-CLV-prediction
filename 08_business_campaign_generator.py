import sqlite3, os
import pandas as pd

# 1. Load Final Predictions from SQLite
conn = sqlite3.connect("online_retail.db")
df = pd.read_sql_query("SELECT * FROM dim_customer_predictions", conn)
conn.close()

# 2. Vectorized Campaign Mapping Logic
def assign_action(row):
    persona, clv = row['Customer_Persona'], row['Predicted_90Day_CLV']
    
    if persona == 'Champions':
        return ('VIP Loyalty Program', 'Early access to new launches, zero discount needed.')
    elif persona in ['Loyal Customers', 'Promising / Potential']:
        return ('Cross-Sell Drive', 'Recommend bundle products based on purchase history.')
    elif persona == 'New Customers':
        return ('Onboarding Sequence', '10% discount on 2nd purchase within 30 days.')
    elif persona == 'At Risk':
        return ('High-Value Win-Back', '20% off limited promo code + direct outreach.') if clv > 500 else ('Standard Retention Push', 'Automated email re-engagement.')
    elif persona == "Can't Lose Them":
        return ('Executive Outreach', 'Feedback survey + customized high-value offer.')
    else:
        return ('Suppress / Low-Cost Ads', 'Exclude from paid campaigns to save marketing budget.')

# Apply mapping and unpack tuples
actions = df.apply(assign_action, axis=1)
df['Recommended_Campaign'] = [a[0] for a in actions]
df['Campaign_Strategy_Note'] = [a[1] for a in actions]

# 3. Print Executive Summary
summary = df.groupby('Customer_Persona').agg(
    Customer_Count=('CustomerID', 'count'),
    Avg_Recency_Days=('Recency', 'mean'),
    Total_Predicted_CLV=('Predicted_90Day_CLV', 'sum'),
    Avg_Predicted_CLV=('Predicted_90Day_CLV', 'mean')
).reset_index().sort_values(by='Total_Predicted_CLV', ascending=False)

summary['CLV_Share_%'] = (summary['Total_Predicted_CLV'] / summary['Total_Predicted_CLV'].sum()) * 100

print("\n=================== EXECUTIVE CLV & CAMPAIGN SUMMARY ===================")
print(summary.to_string(index=False, formatters={
    'Avg_Recency_Days': '{:.1f}'.format,
    'Total_Predicted_CLV': '${:,.2f}'.format,
    'Avg_Predicted_CLV': '${:,.2f}'.format,
    'CLV_Share_%': '{:.1f}%'.format
}))

# 4. Export Priority CRM List & Update Database
os.makedirs("outputs/crm_campaign_lists", exist_ok=True)

# Urgent Action List: High CLV At-Risk & Can't Lose Them Users
urgent_list = df[(df['Customer_Persona'].isin(['At Risk', "Can't Lose Them"])) & (df['Predicted_90Day_CLV'] > 300)]
urgent_list.sort_values(by='Predicted_90Day_CLV', ascending=False).to_csv("outputs/crm_campaign_lists/URGENT_high_value_winback_list.csv", index=False)

# Save Master Strategic Database Table
conn = sqlite3.connect("online_retail.db")
df.to_sql("dim_customer_marketing_master", conn, if_exists="replace", index=False)
df.to_csv("outputs/master_customer_marketing_dataset.csv", index=False)
conn.close()

print(f"\n[SUCCESS] Project Complete! Exported {len(urgent_list)} high-priority targets to CRM list.")