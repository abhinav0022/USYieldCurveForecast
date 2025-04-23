import ssl
import certifi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from fredapi import Fred

# Fix SSL Certificate verification for macOS
ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = lambda: ssl_context

# -----------------------------
# Configuration
# -----------------------------
FRED_API_KEY = '6f7d78bc14c5e350a2727d0290e0bee7'  # <- Replace this with your actual FRED API key
FOMC_EVENTS_FILE = 'fomc_events.csv'  # <- Make sure this CSV exists

# Initialize Fred API client
fred = Fred(api_key=FRED_API_KEY)

# Function to fetch yield data
def fetch_yield_series(series_id):
    try:
        return fred.get_series(series_id)
    except Exception as e:
        print(f"Error fetching {series_id}: {e}")
        return None

# -----------------------------
# Step 1: Download Yield Data
# -----------------------------
yields = {
    '2Y': fetch_yield_series('GS2'),
    '5Y': fetch_yield_series('GS5'),
    '10Y': fetch_yield_series('GS10'),
    '30Y': fetch_yield_series('GS30')
}

# Check if any series failed
for key, value in yields.items():
    if value is None:
        print(f"Failed to download {key} yield data.")
    else:
        print(f"{key} yield data successfully downloaded.")

# Check if any yield data is None
if all(series is not None and not series.empty for series in yields.values()):
    df_yields = pd.DataFrame(yields)
    df_yields.index = pd.to_datetime(df_yields.index)
    df_yields = df_yields.dropna()
    print("✅ Yield data downloaded")
else:
    print("❌ Failed to download one or more yield series.")
    exit()

# -----------------------------
# Step 2: Load FOMC Events
# -----------------------------
df_fomc = pd.read_csv(FOMC_EVENTS_FILE, parse_dates=['Date'])

# Sample expected structure of fomc_events.csv:
# Date,RateChange,Sentiment
# 2023-03-22,0.25,Hawkish
# 2023-06-14,0.00,Dovish

print("✅ FOMC events loaded")

# -----------------------------
# Step 3: Merge & Calculate Yield Changes
# -----------------------------
def get_yield_changes(date):
    try:
        before = df_yields.loc[date - pd.Timedelta(days=1)]
        after = df_yields.loc[date + pd.Timedelta(days=1)]
        return after - before
    except KeyError:
        return pd.Series([np.nan] * 4, index=['2Y', '5Y', '10Y', '30Y'])

df_fomc[['Δ2Y', 'Δ5Y', 'Δ10Y', 'Δ30Y']] = df_fomc['Date'].apply(get_yield_changes).apply(pd.Series)
df_fomc.dropna(inplace=True)

print("✅ Yield changes calculated")

# -----------------------------
# Step 4: Regression Analysis
# -----------------------------
# Create dummy variables for sentiment
X = pd.get_dummies(df_fomc[['RateChange', 'Sentiment']], drop_first=True)
y = df_fomc['Δ2Y']
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print("✅ Regression completed")
print(model.summary())

# -----------------------------
# Optional: Plotting
# -----------------------------
plt.figure(figsize=(10, 6))
plt.scatter(df_fomc['RateChange'], df_fomc['Δ2Y'], c=df_fomc['Sentiment'].map({'Dovish': 'blue', 'Hawkish': 'red'}))
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Rate Change (%)")
plt.ylabel("Change in 2Y Yield (%)")
plt.title("2Y Yield Change vs FOMC Rate Decision")
plt.grid(True)
plt.show()