# ROI Simulation Model comparing Broad vs. Value-Targeted Campaigns
total_customers = 5281
at_risk_high_val = 80
at_risk_low_val = 426
avg_discount_cost = 15.00  # $15 promo code per customer
winback_conversion_rate = 0.18  # 18% expected conversion
avg_order_value = 350.00

# Strategy A: Broad Blast (Discount sent to ALL At-Risk customers: 506 accounts)
cost_broad = (at_risk_high_val + at_risk_low_val) * avg_discount_cost
revenue_broad = (at_risk_high_val + at_risk_low_val) * winback_conversion_rate * avg_order_value
net_profit_broad = revenue_broad - cost_broad

# Strategy B: Machine Learning Targeted (Discount sent ONLY to high-CLV At-Risk accounts: 80 accounts)
cost_targeted = at_risk_high_val * avg_discount_cost
revenue_targeted = at_risk_high_val * winback_conversion_rate * avg_order_value
net_profit_targeted = revenue_targeted - cost_targeted

print(f"Broad Strategy Spend: ${cost_broad:,.2f} | Net Revenue: ${net_profit_broad:,.2f}")
print(f"Targeted Strategy Spend: ${cost_targeted:,.2f} | Net Revenue: ${net_profit_targeted:,.2f}")
print(f"Marketing Budget Saved: ${cost_broad - cost_targeted:,.2f}")