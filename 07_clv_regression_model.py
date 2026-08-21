import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

# 1. Connect to Database & Load Master Dataset
conn = sqlite3.connect("online_retail.db")
df = pd.read_sql_query("SELECT * FROM dataset_churn_clv_gold", conn)
conn.close()

print(f"[INFO] Master ML Dataset loaded: {df.shape[0]} rows.")

# Load Churn Classification Model from Step 6 for $P(\text{Active})$
churn_pipeline = joblib.load("models/churn_pipeline_best.joblib")

# Calculate P(Active) for all customers
df['Prob_Active'] = churn_pipeline.predict_proba(
    df[['Recency', 'Frequency', 'Monetary', 'R_Score', 'F_Score', 'M_Score', 'RFM_Score_Sum', 'Customer_Persona']]
)[:, 0]  # Index 0 is probability of class 0 (Active/Not Churned)


# 2. FILTER & SPLIT FOR REGRESSION (Active Buyers Only)
# We train regression on customers who had non-zero spend in holdout
active_spenders = df[df['HoldoutMonetary'] > 0].copy()

feature_cols = [
    'Recency', 'Frequency', 'Monetary', 
    'R_Score', 'F_Score', 'M_Score', 'RFM_Score_Sum', 
    'Customer_Persona'
]

X_reg = active_spenders[feature_cols]
# Applying log1p to remove extreme right-skewness of spend
y_reg = np.log1p(active_spenders['HoldoutMonetary']) 

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.20, random_state=42
)

print(f"[INFO] Regression dataset: {len(active_spenders)} non-zero holdout spenders.")


# 3. PREPROCESSING & REGRESSION PIPELINES
num_features = ['Recency', 'Frequency', 'Monetary', 'R_Score', 'F_Score', 'M_Score', 'RFM_Score_Sum']
cat_features = ['Customer_Persona']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_features)
    ]
)

reg_models = {
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42),
    'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
}

results = {}

for name, model in reg_models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Predict in log scale, then transform back using expm1
    y_pred_log = pipeline.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_orig = np.expm1(y_test)
    
    mae = mean_absolute_error(y_test_orig, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    r2 = r2_score(y_test_orig, y_pred)
    
    results[name] = {
        'pipeline': pipeline,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
    
    print(f"\n================ {name} ================")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"R-squared Score (R2): {r2:.4f}")


# 4. BEST MODEL SELECTION & EXPECTED CLV PREDICTION
best_reg_name = max(results, key=lambda k: results[k]['r2'])
best_reg_pipeline = results[best_reg_name]['pipeline']

print(f"\n[WINNER] Best Regressor: {best_reg_name} (R2 = {results[best_reg_name]['r2']:.4f})")

# Save CLV model artifact
joblib.dump(best_reg_pipeline, "models/clv_pipeline_best.joblib")

# Predict 90-day conditional spend for ALL customers
X_all = df[feature_cols]
predicted_conditional_spend = np.expm1(best_reg_pipeline.predict(X_all))

# Calculate Expected 90-Day CLV = P(Active) * Predicted Spend
df['Predicted_90Day_CLV'] = df['Prob_Active'] * predicted_conditional_spend


# 5. SAVE FINAL PREDICTIONS TO SQLITE & CSV
conn = sqlite3.connect("online_retail.db")
df.to_sql("dim_customer_predictions", conn, if_exists="replace", index=False)
conn.close()

df.to_csv("outputs/customer_clv_predictions.csv", index=False)
print("[INFO] Predictions saved to table 'dim_customer_predictions' and 'outputs/customer_clv_predictions.csv'")


# 6. VISUALIZATION: ACTUAL VS PREDICTED SPEND & CLV BY PERSONA
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Average Expected CLV by Customer Persona
persona_clv = df.groupby('Customer_Persona')['Predicted_90Day_CLV'].mean().reset_index()
persona_clv = persona_clv.sort_values(by='Predicted_90Day_CLV', ascending=False)

ax1 = sns.barplot(
    data=persona_clv, 
    x='Predicted_90Day_CLV', 
    y='Customer_Persona', 
    palette='Blues_r', 
    ax=axes[0]
)

for container in ax1.containers:
    ax1.bar_label(container, fmt='$%.2f', padding=5, fontsize=10)

axes[0].set_title('Average Predicted 90-Day CLV ($) by Persona', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Expected Spend ($)', fontsize=12)

# Plot 2: Top High-Value At-Risk Customers (Target for Marketing Campaigns)
at_risk_high_clv = df[df['Customer_Persona'].isin(['At Risk', "Can't Lose Them"])]
ax2 = sns.scatterplot(
    data=at_risk_high_clv, 
    x='Recency', 
    y='Predicted_90Day_CLV', 
    hue='Customer_Persona', 
    size='Monetary', 
    sizes=(20, 200), 
    palette='Set1', 
    ax=axes[1]
)

axes[1].set_title('High-Value At-Risk Customers (Priority Retention List)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Recency (Days Inactive)', fontsize=12)
axes[1].set_ylabel('Predicted 90-Day CLV ($)', fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("outputs/clv_evaluation_plots.png", dpi=300)
plt.show()

print("[SUCCESS] Step 7 complete! Visualizations saved to 'outputs/clv_evaluation_plots.png'")