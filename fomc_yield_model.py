import pandas as pd
import numpy as np
from fredapi import Fred
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
import certifi

# ------------------------------
# Configuration
# ------------------------------
FRED_API_KEY = '6f7d78bc14c5e350a2727d0290e0bee7'  # Replace with your FRED API key
FOMC_CSV = 'fomc_events.csv'  # Input: FOMC event history
LOOKAHEAD_DAYS = 1  # Yield change measurement window (1 day after event)

# Fix SSL certificate issue on macOS
os.environ['SSL_CERT_FILE'] = certifi.where()

# ------------------------------
# Step 1: Load FOMC Event Data
# ------------------------------
df_fomc = pd.read_csv(FOMC_CSV, parse_dates=['Date'])
print("FOMC Data Loaded:")
print(df_fomc.head())

# Expected columns: Date, RateChange, Sentiment (e.g., Hawkish, Dovish), InflationExpectation (optional)

# ------------------------------
# Step 2: Load Yield Curve and Inflation Data
# ------------------------------
fred = Fred(api_key=FRED_API_KEY)

# Fetch yield curve data
yields = {
    '2Y': fred.get_series('GS2'),
    '5Y': fred.get_series('GS5'),
    '10Y': fred.get_series('GS10'),
    '30Y': fred.get_series('GS30')
}

df_yields = pd.DataFrame(yields)
df_yields.index = pd.to_datetime(df_yields.index)
df_yields = df_yields.dropna()

# Load 5-year breakeven inflation rate
df_infl = fred.get_series('T5YIE').to_frame(name='InflationExp')
df_infl.index = pd.to_datetime(df_infl.index)

# Print the first few rows of the yield data for debugging
print("Yield Data Loaded:")
print(df_yields.head())

# ------------------------------
# Step 3: Merge FOMC + Yield + Inflation
# ------------------------------
def calc_yield_delta(date):
    try:
        # Extract the month and year from the FOMC event date
        year_month = (date.year, date.month)
        
        # Find the matching month-year in the yield data
        before = df_yields[(df_yields.index.year == year_month[0]) & (df_yields.index.month == year_month[1])].iloc[0]
        
        # Find the next available yield data after the given FOMC event (same month)
        next_month = (year_month[0], year_month[1] + 1) if year_month[1] < 12 else (year_month[0] + 1, 1)
        after = df_yields[(df_yields.index.year == next_month[0]) & (df_yields.index.month == next_month[1])].iloc[0]
        
        print(f"Before: {before}")
        print(f"After: {after}")
        
        # If either 'before' or 'after' is empty, return NaN values
        if before.empty or after.empty:
            print(f"Empty slice found for date: {date}, returning NaN values")
            return pd.Series([np.nan] * 4, index=['2Y', '5Y', '10Y', '30Y'])
        
        return after - before
    except IndexError:
        print(f"IndexError: No data for {date}, returning NaN values")
        return pd.Series([np.nan] * 4, index=['2Y', '5Y', '10Y', '30Y'])


# Apply the function to calculate yield deltas
df_fomc[['Δ2Y', 'Δ5Y', 'Δ10Y', 'Δ30Y']] = df_fomc['Date'].apply(calc_yield_delta).apply(pd.Series)

# Merge inflation expectations
df_fomc = df_fomc.merge(df_infl, how='left', left_on='Date', right_index=True)
df_fomc.dropna(inplace=True)

print("Merged FOMC, Yield, and Inflation Data:")
print(df_fomc.head())

# ------------------------------
# Step 4: Feature Engineering
# ------------------------------
X = df_fomc[['RateChange', 'InflationExp']]
X = pd.concat([X, pd.get_dummies(df_fomc['Sentiment'], drop_first=True)], axis=1)
X = sm.add_constant(X)

# Define the target variables (yield deltas)
targets = {
    'Δ2Y': df_fomc['Δ2Y'],
    'Δ5Y': df_fomc['Δ5Y'],
    'Δ10Y': df_fomc['Δ10Y'],
    'Δ30Y': df_fomc['Δ30Y'],
}

# ------------------------------
# Step 5: Train Models and Predict
# ------------------------------
models = {}
predictions = {}

print("✅ Training regression models...\n")

# Train a separate model for each yield delta
for label, y in targets.items():
    model = sm.OLS(y, X).fit()
    models[label] = model
    print(f"--- {label} ---")
    print(model.summary())
    print("\n")

# ------------------------------
# Step 6: Forecast Current Curve Move
# ------------------------------
# Input your current scenario (adjust these values as needed)
scenario = {
    'RateChange': 0.00,
    'InflationExp': 2.2,
    'Sentiment_Hawkish': 1,  # 1 if Hawkish, 0 if not
    'Sentiment_Neutral': 0   # 1 if Neutral, 0 if not
}
x_new = pd.DataFrame([scenario])
x_new = sm.add_constant(x_new)

print("📈 Forecasted yield curve moves based on current scenario:\n")
for label, model in models.items():
    y_pred = model.predict(x_new)[0]
    print(f"{label}: {y_pred:.3f}%")

# ------------------------------
# Step 7: Plot Yield Curve Shift
# ------------------------------
current_curve = df_yields.iloc[-1]
forecast_shift = [models[f'Δ{x}Y'].predict(x_new)[0] for x in ['2', '5', '10', '30']]
forecasted_curve = current_curve + forecast_shift

# Plot the current and forecasted yield curves
plt.figure(figsize=(10, 6))
plt.plot([2, 5, 10, 30], current_curve.values, marker='o', label='Current Curve')
plt.plot([2, 5, 10, 30], forecasted_curve.values, marker='o', label='Forecasted Curve', linestyle='--')
plt.title('U.S. Yield Curve Forecast')
plt.xlabel('Maturity (Years)')
plt.ylabel('Yield (%)')
plt.grid(True)
plt.legend()
plt.show()