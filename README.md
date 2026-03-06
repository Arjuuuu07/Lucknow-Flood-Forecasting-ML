# Lucknow Flood Forecasting — Geospatial AI/ML

> A high-precision geospatial machine learning model for flood risk forecasting in Lucknow District, achieving **100% Recall** — zero missed flood events — validated on unseen 2008 flood data.

---

## Overview

This project builds a predictive system to forecast flood risk in Lucknow District using historical rainfall data. The model is trained on extreme weather events from **1971, 2021, and 2023**, and validated against the **2008 flood** — data the model had never seen during training.

The system goes beyond simple daily rainfall by modeling how water physically accumulates and flows *into* the Lucknow basin, using 11 geospatial monitoring points and 7-day cumulative rainfall logic.

---

## The Core Problem: Rare Event Modeling

Floods are rare events. In any historical rainfall record, flood days are a tiny fraction of total days — creating extreme class imbalance.

A naive model would predict "No Flood" every day and still achieve ~99% accuracy. That is dangerous.

This project addresses that directly:

- **No SMOTE** — synthetic oversampling was avoided to preserve data integrity
- **`class_weight='balanced'`** — the model is penalized heavily for missing a real flood
- **Custom inference threshold (0.75)** — only flags a flood when confidence is ≥ 75%, reducing false alarms while keeping recall at 1.00

---

## Why Only Rainfall Features?

A deliberate design choice was made to use **only rainfall and cumulative rainfall** as features — no elevation, slope, soil type, or land use data.

The reasoning:

Within Lucknow district, physical parameters like terrain height, soil type, and land composition are **relatively constant**. They do not meaningfully vary across the district in a way that changes flood prediction.

If modeling flood risk across all of Uttar Pradesh, these factors would matter — because Varanasi and Lucknow flood on **different dates**, from different rivers, in different terrain. Elevation and soil would be necessary to capture that regional specificity.

But this model answers one question: **"Will Lucknow flood — yes or no?"** For that binary district-level question, the physical constants cancel out. What drives the answer is **how much water is arriving and where it is coming from** — which is exactly what the 11-point geospatial rainfall network captures.

---

## Hydrological Feature Engineering

### 11-Point Geospatial Monitoring Network

The district is monitored through two sets of coordinates:

#### 7 Internal Lucknow Points
Capture localized precipitation intensity *within* the district.

| Point | Coordinates |
|---|---|
| 1 | (26.50°N, 81.00°E) |
| 2 | (26.75°N, 81.25°E) |
| 3 | (26.75°N, 81.00°E) |
| 4 | (26.75°N, 80.75°E) |
| 5 | (27.00°N, 81.00°E) |
| 6 | (27.00°N, 80.75°E) |
| 7 | (27.00°N, 80.50°E) |

#### 4 Upstream River Inflow Points
Monitor rainfall at coordinates where rivers enter Lucknow from the north. This enables the model to predict upstream-driven flooding even when local rainfall is low.

| Point | Coordinates |
|---|---|
| 1 | (27.75°N, 80.25°E) |
| 2 | (27.50°N, 80.25°E) |
| 3 | (27.25°N, 80.75°E) |
| 4 | (27.50°N, 80.75°E) |

### CUMI — 7-Day Cumulative Rainfall Index

For each of the 11 monitoring points, a **Cumulative Moisture Index** is engineered:

```
CUMI = rolling 7-day sum of rainfall at each coordinate
```

**Why 7 days?** If the ground is already saturated from a week of rain, even a modest storm can trigger a catastrophic flood. A single-day rainfall feature misses this saturation effect entirely. CUMI directly models soil moisture state and basin storage.

This produces the full feature set of **22 features** per row:

- `rainfall_orig_1` ... `rainfall_orig_7` — daily rainfall at 7 internal points
- `cumi_orig_1` ... `cumi_orig_7` — 7-day cumulative rainfall at internal points
- `rainfall_adj_1` ... `rainfall_adj_4` — daily rainfall at 4 upstream inflow points
- `cumi_adj_1` ... `cumi_adj_4` — 7-day cumulative rainfall at inflow points

Latitude/longitude columns are dropped before training — they are coordinate identifiers, not predictive signals.

---

## Model

**Algorithm:** Logistic Regression with `class_weight='balanced'`

**Preprocessing:**
- Median imputation for missing values
- Standard scaling

**Train/Test Split:** 75/25 stratified split

**Inference Threshold:** 0.75 (conservative — reduces false alarms)

Logistic Regression was chosen deliberately: it is interpretable, its coefficients directly reveal which features drive flood prediction, and it avoids overfitting on a small rare-event dataset.

---

## Model Performance

### Key Metrics

| Metric | Value |
|---|---|
| Recall | **1.00** — zero missed floods |
| F1-Score | **0.89** |
| Inference Threshold | 0.75 |
| SMOTE Used | No — class_weight='balanced' only |

### Error Analysis

| Outcome | Count |
|---|---|
| True Positives (Correct Flood Alerts) | 4 |
| True Negatives (Correct Quiet Days) | 269 |
| False Positives (False Alarms) | 1 |
| False Negatives (Missed Floods) | **0** |

Every historical flood was detected. One false alarm on a non-flood day is an acceptable trade-off for zero missed disasters.

### Top 5 Flood Drivers

| Feature | Importance |
|---|---|
| `cumi_orig_4` | 1.1077 |
| `cumi_orig_1` | 1.0451 |
| `cumi_orig_5` | 0.9342 |
| `cumi_adj_3` | 0.7330 |
| `cumi_adj_2` | 0.6176 |

Cumulative rainfall features dominate — confirming that **soil saturation over days**, not single-day rainfall spikes, is the primary driver of Lucknow floods. The presence of upstream inflow points (`cumi_adj_*`) in the top 5 validates the river flow hypothesis.

---

## Blind Test — 2008 Flood Validation

The model was evaluated on **unlabeled 2008 data** — a flood year it had never encountered during training.

Without being told that 2008 was a flood year, the model correctly flagged the monsoon flood dates. This demonstrates genuine generalization to unseen hydrological events, not overfitting to the training years.

---


## Limitations

- Binary district-level prediction — does not localize *where* in Lucknow flooding will occur
- Trained on three flood years; more events would improve robustness
- Does not incorporate river gauge levels or groundwater data
- Currently offline — not a real-time operational system

---

## Planned Extensions

- [ ] Integration of real-time IMD rainfall API for live forecasting
- [ ] Expansion to other Uttar Pradesh districts with region-specific inflow networks
- [ ] Multi-day ahead forecasting (predicting flood risk 3–5 days in advance)
- [ ] River gauge level data integration for improved upstream flood detection

---

## Author

**Arjun**
MSc Artificial Intelligence & Machine Learning
Indian Institute of Information Technology, Lucknow (IIIT-L)
