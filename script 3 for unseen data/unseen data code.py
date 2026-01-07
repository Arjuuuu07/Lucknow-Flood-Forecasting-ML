import pandas as pd
import numpy as np


unseen_df = pd.read_csv('/content/2008 CHANGED.csv') 


unseen_dates = unseen_df['date']

# Drop non-feature columns (adjust this list to match exactly what you dropped in training)
drop_cols = [c for c in unseen_df.columns if 'lat' in c or 'lon' in c or 'date' in c or 'flood' in c]
X_unseen = unseen_df.drop(columns=drop_cols, errors='ignore')


X_unseen_scaled = scaler.transform(X_unseen)


probs = model.predict_proba(X_unseen_scaled)[:, 1]

# 2. Apply your 75% threshold
threshold = 0.75
# A date is marked as 'Flood' only if probability is >= 0.75
custom_preds = (probs >= threshold).astype(int)


report = pd.DataFrame({
    'Date': unseen_dates,
    'Flood_Risk_%': (probs * 100).round(2),
    'Alert_Level': [" HIGH RISK" if p >= threshold else " Normal" for p in probs]
})

alerts = report[report['Alert_Level'] == " HIGH RISK"].sort_values(by='Date')

print("="*60)
print(f"   LUCKNOW FLOOD DISCOVERY REPORT (Threshold: {threshold*100}%)   ")
print("="*60)

if not alerts.empty:
    print(f"Total High-Risk Dates Identified: {len(alerts)}")
    print("-" * 60)
    print(alerts[['Date', 'Flood_Risk_%', 'Alert_Level']].to_string(index=False))
else:
    print(f"No dates found with risk above {threshold*100}%.")
    
   
    peak = report.loc[report['Flood_Risk_%'].idxmax()]
    print(f"\nPeak risk detected: {peak['Flood_Risk_%']}% on {peak['Date']}")

print("="*60)