# 🛒 End-to-End E-Commerce Customer Analytics & Lifetime Value Prediction System

An end-to-end Machine Learning and Business Intelligence framework built using Python and SQL. This system ingests raw transaction logs, models customer retention behavior, and predicts 90-Day Expected Customer Lifetime Value (CLV) to generate automated CRM retention strategies.

---

## 📌 Project Architecture & Workflow

```text
Raw Transaction Logs
       │
       ▼
[ Step 1: SQL Data Warehouse Creation (Star Schema) ]
       │
       ▼
[ Step 2: Quality Audit & Cohort Retention Analysis ]
       │
       ▼
[ Step 3 & 4: Cutoff Windowing & RFM Segmentation ]
       │
       ├───────────────────────────────┐
       ▼                               ▼
[ Step 6: Churn Classifier ]    [ Step 7: CLV Hurdle Regressor ]
  (Gradient Boosting/RF)          (Gradient Boosting)
       │                               │
       └───────────────┬───────────────┘
                       ▼
    [ Expected CLV = P(Active) × Spend ]
                       │
                       ▼
[ Step 8: Automated CRM Retention Engine ]
```

---

## 🚀 Key Features & Highlights

* **Data Engineering & Warehousing**: Structured raw retail records into a clean SQL Star Schema (`fact_orders`, `dim_customer`, `dim_product`) with robust SQL-level quality checks.
* **Non-Overlapping Windowing**: Designed observation (X) and holdout (y) windows to eliminate data leakage in predictive features.
* **RFM Customer Segmentation**: Classified customer personas (Champions, At Risk, Hibernating, etc.) using quantile-based scoring (`qcut`).
* **Two-Stage Hurdle Model for CLV**:
  * **Stage 1**: Binary Classification for Churn / Repeat Purchase Probability P(Active).
  * **Stage 2**: Log-Transformed Machine Learning Regression to estimate conditional holdout spend.
* **Automated Strategy Allocation Engine**: Mapped predicted risk and CLV directly to tailored marketing campaigns (e.g., VIP Loyalty vs. High-Value Win-Back) to minimize ad waste.

---

## 📁 Repository Structure

```text
├── data/
│   └── online_retail_II.xlsx                         # Raw online retail data
|
├── docs/
│   ├── Customer Lifetime Value Report.docx           # Technical report source document (DOCX)
│   └── Customer Lifetime Value Report.pdf            # Comprehensive technical analytical report (PDF)
|
├── models/                                           # Saved ML Pipelines (.joblib)
│   ├── churn_pipeline_best.joblib
│   └── clv_pipeline_best.joblib
|
├── outputs/                                          # Visual plots & exported CRM targeting lists
│   ├── crm_campaign_lists/
│   │   └── URGENT_high_value_winback_list.csv
│   ├── churn_rate_by_persona.png
│   ├── clv_evaluation_plots.png
│   ├── cohort_retention_heatmap.png
│   ├── customer_clv_predictions.csv
│   ├── master_customer_marketing_dataset.csv
│   ├── master_ml_dataset.csv
│   ├── model_evaluation_metrics.png
│   ├── persona_distribution.png
│   ├── rfm_distribution.png
│   ├── rfm_features.csv
│   └── rfm_segmented_customers.csv
|
├── src
│   ├── 01_data_warehouse_setup.py                    # SQL Data Warehouse & Star Schema creation
│   ├── 02_cohort_retention_analysis.py               # SQL CTE-based Cohort Retention Analysis
│   ├── 03_rfm_feature_engineering.py                 # Cutoff feature engineering & leak prevention
│   ├── 04_customer_segmentation.py                   # Quantile-based RFM Persona modeling
│   ├── 06_churn_prediction_model.py                  # Supervised Classification for Churn
│   ├── 07_clv_regression_model.py                    # Log-Transformed Regression & Hurdle CLV
│   ├── 08_business_campaign_generator.py             # CRM Strategy Mapping & CSV Exporter
│   ├── crm_automation_sync
│   └── crm_automation_sync
│
│
│
└── .gitignore                                        # Excludes heavy SQLite files (>100MB)
```

---

## 📊 Business Impact & Campaign Mapping Logic

| Customer Persona | Predicted Risk / Value | Automated Recommended Strategy |
| :--- | :--- | :--- |
| **Champions** | High CLV, Low Risk | VIP Loyalty Program (No discounts, early product access) |
| **Promising / Potential** | Medium CLV, Low Risk | Cross-Sell / Upsell Drive based on historical basket data |
| **At Risk (CLV > $500)** | High CLV, High Risk | Urgent High-Value Win-Back (Personalized high-value offers) |
| **About to Sleep / Lost** | Low CLV, High Risk | Campaign Suppression (Preserves marketing/ad budget) |

---

## 🛠️ Tech Stack & Dependencies

* **Language**: Python 3.10+
* **Database & Querying**: SQLite3, SQL CTEs, Window Functions
* **Data Processing**: `pandas`, `numpy`
* **Machine Learning & Modeling**: `scikit-learn`, `joblib`
* **Visualization**: `matplotlib`, `seaborn`

---

## ⚡ How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/KFI-117/Customer-CLV-prediction.git
   cd Customer-CLV-prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn joblib
   ```

3. **Execute scripts sequentially**:
   ```bash
   python 01_data_warehouse_setup.py
   python 02_cohort_retention_analysis.py
   python 03_rfm_feature_engineering.py
   python 04_customer_segmentation.py
   python 06_churn_prediction_model.py
   python 07_clv_regression_model.py
   python 08_business_campaign_generator.py
   ```
