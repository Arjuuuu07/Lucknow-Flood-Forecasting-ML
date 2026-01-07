import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load raw UP data
raw_data = "rainfall_UP_1971.csv"
df = pd.read_csv(raw_data)

# Fix dates and sort for rolling calculation
df["TIME"] = pd.to_datetime(df["TIME"])
df = df.sort_values(["lat", "lon", "TIME"])

# Lucknow specific zones (1-7)
lucknow_coords = [
    (26.5, 81), (26.75, 81.25), (26.75, 81), (26.75, 80.75),
    (27, 81), (27, 80.75), (27, 80.50)
]

# Upstream river inflow points (1-4)
river_inflow_coords = [
    (27.75, 80.25), (27.5, 80.25), (27.25, 80.75), (27.5, 80.75)
]

def process_station(df, lat, lon):
    mask = (df.lat == lat) & (df.lon == lon)
    subset = df[mask].copy()
    # 7-day total rainfall for soil saturation (CUMI)
    subset["cumi"] = subset["rainfall"].rolling(7, min_periods=1).sum()
    return subset[["TIME", "lat", "lon", "rainfall", "cumi"]]

merged = None

# Process internal Lucknow points
for i, loc in enumerate(lucknow_coords, 1):
    print(f"Processing Lucknow station {i}...")
    temp = process_station(df, loc[0], loc[1])
    temp.columns = ["date", f"lat_orig_{i}", f"lon_orig_{i}", f"rainfall_orig_{i}", f"cumi_orig_{i}"]
    
    if merged is None:
        merged = temp
    else:
        merged = pd.merge(merged, temp, on="date", how="outer")

# Process river flow points
for i, loc in enumerate(river_inflow_coords, 1):
    print(f"Processing River Inflow station {i}...")
    temp = process_station(df, loc[0], loc[1])
    temp.columns = ["date", f"lat_adj_{i}", f"lon_adj_{i}", f"rainfall_adj_{i}", f"cumi_adj_{i}"]
    merged = pd.merge(merged, temp, on="date", how="outer")

# Basic cleaning and target init
merged = merged.sort_values("date").reset_index(drop=True)
merged["flood"] = 0 # Placeholder for target labels

merged.to_csv("newfile.csv", index=False)
print("\nDone. Final shape:", merged.shape)