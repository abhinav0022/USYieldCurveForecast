import os
import certifi
import pandas as pd
import numpy as np
from fredapi import Fred
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import het_breuschpagan

# ------------------------------
# Configuration
# ------------------------------
FRED_API_KEY = '6f7d78bc14c5e350a2727d0290e0bee7'   # ← Replace with your FRED API key
os.environ['SSL_CERT_FILE'] = certifi.where()

# ------------------------------
# Step 1: Fetch Treasury Yields
# ------------------------------
fred = Fred(api_key=FRED_API_KEY)
yields_codes = {'2Y': 'GS2', '5Y': 'GS5', '10Y': 'GS10', '30Y': 'GS30'}
df_yields = pd.DataFrame({tenor: fred.get_series(code) for tenor, code in yields_codes.items()})
df_yields.index = pd.to_datetime(df_yields.index)
df_yields.dropna(inplace=True)
df_yield_m = df_yields.resample('M').last()  # month‐end yields

# ------------------------------
# Step 2: Compute Monthly Yield Changes
# ------------------------------
df_yield_delta = df_yield_m.diff().dropna()
df_yield_delta.columns = ['Δ2Y', 'Δ5Y', 'Δ10Y', 'Δ30Y']

# ------------------------------
# Step 3: Fetch Macro Variables
# ------------------------------
df_ffr  = fred.get_series('FEDFUNDS').to_frame('FedFundsRate')
df_infl = fred.get_series('T5YIE').to_frame('InflationExp')
df_vix  = fred.get_series('VIXCLS').to_frame('VIX')
df_gdp  = fred.get_series('GDPC1').to_frame('RealGDP')
df_gdp['GDPGrowth'] = df_gdp['RealGDP'].pct_change(periods=4) * 100

# Align macros to month‐end
macro = pd.concat([df_ffr, df_infl, df_vix, df_gdp['GDPGrowth']], axis=1)
macro = macro.resample('M').last().dropna()

# ------------------------------
# Step 4: Merge Yield Deltas & Macros
# ------------------------------
df = macro.join(df_yield_delta, how='inner').dropna()

# ------------------------------
# Step 5: Run Regressions with HC3 Robust SE
# ------------------------------
X = sm.add_constant(df[['FedFundsRate', 'InflationExp', 'GDPGrowth', 'VIX']])
models = {}
print("Regression results with HC3 robust standard errors:\n")
for tenor in ['Δ2Y', 'Δ5Y', 'Δ10Y', 'Δ30Y']:
    y = df[tenor]
    model = sm.OLS(y, X).fit(cov_type='HC3')
    models[tenor] = model
    print(f"--- {tenor} ---")
    print(model.summary())
    bp = het_breuschpagan(model.resid, model.model.exog)
    print(f"Breusch–Pagan p-value: {bp[1]:.4f}\n")

# ------------------------------
# Step 6: Forecast Next Month’s Yield Shift
# ------------------------------
latest = macro.iloc[-1]
x_new = pd.DataFrame([{
    'const':      1.0,
    'FedFundsRate': latest['FedFundsRate'],
    'InflationExp':  latest['InflationExp'],
    'GDPGrowth':     latest['GDPGrowth'],
    'VIX':           latest['VIX']
}])[X.columns]

forecast = {tenor: models[tenor].predict(x_new)[0] for tenor in models}
print("Forecasted monthly yield changes:")
for t, c in forecast.items():
    print(f"{t}: {c:.3f}%")

# ------------------------------
# Step 7: Plot Current vs Forecasted Curve
# ------------------------------
current   = df_yield_m.iloc[-1]
forecasted = current + pd.Series({
    '2Y':  forecast['Δ2Y'],
    '5Y':  forecast['Δ5Y'],
    '10Y': forecast['Δ10Y'],
    '30Y': forecast['Δ30Y']
})

plt.figure(figsize=(8, 5))
mats = [2, 5, 10, 30]
plt.plot(mats, current.values, marker='o', label='Current Yield Curve')
plt.plot(mats, forecasted.values, marker='o', linestyle='--', label='Forecasted Yield Curve')
plt.title('Monthly Yield Curve Forecast')
plt.xlabel('Maturity (Years)')
plt.ylabel('Yield (%)')
plt.grid(True)
plt.legend()
plt.ylim(0, max(current.max(), forecasted.max()) + 0.5)
plt.show()