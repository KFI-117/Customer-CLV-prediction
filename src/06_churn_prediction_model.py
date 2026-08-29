import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    RocCurveDisplay, 
    precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# 1. Connect to Database & Load Master ML Dataset
conn = sqlite3.connect("online_retail.db")
df = pd.read_sql_query("SELECT * FROM dataset_churn_clv_gold", conn)
conn.close()

print(f"[INFO] Master ML Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

# 2. FEATURE & TARGET SELECTION
# Dropping identifiers and holdout targets to prevent Data Leakage
feature_cols = [
    'Recency', 'Frequency', 'Monetary', 
    'R_Score', 'F_Score', 'M_Score', 'RFM_Score_Sum', 
    'Customer_Persona'
]

X = df[feature_cols]
y = df['IsChurned']

# Train-Test Split (80% Train, 20% Test) with Stratification (preserves Churn ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"[INFO] Churn Ratio in Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")


# 3. PREPROCESSING PIPELINE
num_features = ['Recency', 'Frequency', 'Monetary', 'R_Score', 'F_Score', 'M_Score', 'RFM_Score_Sum']
cat_features = ['Customer_Persona']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_features)
    ]
)


# 4. MODEL TRAINING & COMPARISON
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42)
}

results = {}

for name, model in models.items():
    # Build complete sklearn pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Evaluate
    auc = roc_auc_score(y_test, y_proba)
    results[name] = {
        'pipeline': pipeline,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'auc': auc
    }
    
    print(f"\n================ {name} ================")
    print(f"ROC-AUC Score: {auc:.4f}")
    print(classification_report(y_test, y_pred))


# 5. BEST MODEL SELECTION & ARTIFACT SAVING
best_model_name = max(results, key=lambda k: results[k]['auc'])
best_pipeline = results[best_model_name]['pipeline']

print(f"\n[WINNER] Best Performing Model: {best_model_name} (AUC = {results[best_model_name]['auc']:.4f})")

# Save winning model artifact
os.makedirs("models", exist_ok=True)
joblib.dump(best_pipeline, "models/churn_pipeline_best.joblib")
print("[SUCCESS] Model artifact saved to 'models/churn_pipeline_best.joblib'")


# 6. VISUALIZATIONS: ROC CURVE & FEATURE IMPORTANCE
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: ROC Curves Comparison
for name, res in results.items():
    RocCurveDisplay.from_predictions(
        y_test, res['y_proba'], 
        name=f"{name} (AUC = {res['auc']:.2f})", 
        ax=axes[0]
    )
axes[0].plot([0, 1], [0, 1], 'k--', label='Random Chance')
axes[0].set_title('ROC Curve Comparison for Churn Prediction', fontsize=14, fontweight='bold')
axes[0].grid(True, linestyle='--', alpha=0.5)

# Plot 2: Feature Importance (From Random Forest or Gradient Boosting)
rf_model = results['Random Forest']['pipeline'].named_steps['classifier']
ohe_cols = list(
    results['Random Forest']['pipeline']
    .named_steps['preprocessor']
    .named_transformers_['cat']
    .get_feature_names_out(cat_features)
)
all_feature_names = num_features + ohe_cols

importances = rf_model.feature_importances_
feat_imp_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

sns.barplot(
    data=feat_imp_df.head(10), 
    x='Importance', 
    y='Feature', 
    palette='viridis', 
    ax=axes[1]
)

# Clean label rendering using ax.bar_label
for container in axes[1].containers:
    axes[1].bar_label(container, fmt='%.3f', padding=5, fontsize=10)

axes[1].set_title('Top 10 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
axes[1].set_xlim(0, max(importances) * 1.15)

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/model_evaluation_metrics.png", dpi=300)
plt.show()

print("[SUCCESS] Step 6 complete! Plots saved to 'outputs/model_evaluation_metrics.png'")