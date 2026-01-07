# Lucknow-Flood-Forecasting-ML
Geospatial  model for Lucknow flood forecasting with 100% recall. Features 11-point coordinate monitoring and 7-day CUMULATIVE  RAINFALL logic. Validated on unseen 2008 flood data.

Lucknow Flood Forecasting: A Geospatial AI/ML Approach
Project Overview
This project implements a high-precision predictive model to forecast flood risks in the Lucknow District. By training on extreme weather events from 1971, 2021, and 2023, and validating against the 2008 unseen flood data, this system demonstrates robust generalization in identifying hydrological triggers.

The Problem: Rare Event Modeling
Floods are "Rare Events," leading to a massive class imbalance. A standard model would simply predict "No Flood" to achieve 99% accuracy. This project bypasses that trap using **Cost-Sensitive Learning and a **Custom Inference Threshold to ensure maximum public safety.

 Hydrological Feature Engineering
I engineered features based on the physical movement of water into the Lucknow basin:
11-Point Geospatial Monitoring:7 Internal Points: Capturing localized precipitation intensity within Lucknow.
4 Upstream Inflow Points:** Monitoring rainfall at coordinates where rivers enter the district. This allows the model to predict "Upstream Flooding" even when local rain is low
CUMI (7-Day Cumulative Rainfall):
Instead of looking at a single day, I engineered the Cumulative Moisture Index.
Logic: If the ground is already saturated from 7 days of rain, even a small storm can trigger a catastrophic flood.

Model Performance & "Blind Test" Validation
1. The 2008 Discovery (Unseen Data)
The model was tested on unlabeled data from 2008. Even without being "told" 2008 was a flood year, the model flagged the dates of the 2008 monsoon floods.

2. Strategic Metrics
| Recall | 1.00 | Zero Missed Floods. Every historical disaster was caught. |

| Threshold | 0.75 | Conservative 75% confidence used to reduce "False Alarm" noise. |

| Logic| No SMOTE | Maintained data integrity; used class_weight='balanced'. |

Repository Structure
data-unseen data, master data(data of 2023,2021,1971 combined is given), separate data also present
*code of unseen data, ml code(machine learning code), code for cleaning is uploaded

result-screenshot of result and flood news uploaded.
