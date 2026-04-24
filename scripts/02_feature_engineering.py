"""
Phase 2: Feature Engineering — ML-Ready Dataset
================================================
Transforms the harmonized daily exchange rate dataset into a feature-rich
dataset ready for supervised machine learning.

Input:  data/processed/merged_exchange_rates.csv
Output: data/processed/featured_exchange_rates.csv
        data/processed/feature_heatmap.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

INPUT_CSV = os.path.join(PROCESSED_DIR, 'merged_exchange_rates.csv')
OUTPUT_CSV = os.path.join(PROCESSED_DIR, 'featured_exchange_rates.csv')
OUTPUT_HEATMAP = os.path.join(PROCESSED_DIR, 'feature_heatmap.png')

# ---------------------------------------------------------------------------
# 1. Load merged dataset
# ---------------------------------------------------------------------------
print("=" * 65)
print("  Phase 2: Feature Engineering")
print("=" * 65)

df = pd.read_csv(INPUT_CSV, parse_dates=['Date'], index_col='Date')
print(f"\n[LOAD] Input dataset: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"       Date range: {df.index.min().date()} -> {df.index.max().date()}")
print(f"       Parallel_Rate non-null: {df['Parallel_Rate'].notna().sum()} rows")

# ---------------------------------------------------------------------------
# 2. Time-Series Lags — Parallel Premium (backward-looking only)
# ---------------------------------------------------------------------------
print("\n--- Engineering Features ---")

df['Premium_Lag_1d'] = df['Parallel_Premium'].shift(1)
df['Premium_Lag_3d'] = df['Parallel_Premium'].shift(3)
df['Premium_Lag_7d'] = df['Parallel_Premium'].shift(7)

print("[OK] Lag features: Premium_Lag_1d, Premium_Lag_3d, Premium_Lag_7d")

# ---------------------------------------------------------------------------
# 3. Momentum Indicators — Simple Moving Averages
# ---------------------------------------------------------------------------
df['Official_SMA_7'] = df['Official_Rate'].rolling(window=7, min_periods=7).mean()
df['Official_SMA_30'] = df['Official_Rate'].rolling(window=30, min_periods=30).mean()
df['Parallel_SMA_7'] = df['Parallel_Rate'].rolling(window=7, min_periods=7).mean()
df['Parallel_SMA_30'] = df['Parallel_Rate'].rolling(window=30, min_periods=30).mean()
df['Premium_SMA_7'] = df['Parallel_Premium'].rolling(window=7, min_periods=7).mean()

print("[OK] SMA features: Official_SMA_7/30, Parallel_SMA_7/30, Premium_SMA_7")

# ---------------------------------------------------------------------------
# 4. "Pressure Cooker" — Days Since Adjustment
# ---------------------------------------------------------------------------
# A day is "changed" if the absolute daily diff >= 0.001 ETB
# This threshold absorbs floating-point noise while detecting real moves
CHANGE_THRESHOLD = 0.001

daily_change = df['Official_Rate'].diff().abs()
is_changed = daily_change >= CHANGE_THRESHOLD

# Build groups: each change event starts a new group
# cumsum increments at every True, creating group IDs
change_groups = is_changed.cumsum()

# Within each group, cumcount gives 0, 1, 2, ... (days since the group started)
df['Days_Since_Adjustment'] = df.groupby(change_groups).cumcount()

# Convert to int (safe since cumcount produces integers, but NaN in first row)
# The first row has NaN from diff() — its group is 0, cumcount starts at 0, so it's fine
df['Days_Since_Adjustment'] = df['Days_Since_Adjustment'].astype(int)

max_pressure = df['Days_Since_Adjustment'].max()
print(f"[OK] Days_Since_Adjustment: max consecutive flat days = {max_pressure}")

# ---------------------------------------------------------------------------
# 5. Volatility — Rolling 7-Day Standard Deviation
# ---------------------------------------------------------------------------
df['Official_Vol_7d'] = df['Official_Rate'].rolling(window=7, min_periods=7).std(ddof=1)
df['Parallel_Vol_7d'] = df['Parallel_Rate'].rolling(window=7, min_periods=7).std(ddof=1)

print("[OK] Volatility features: Official_Vol_7d, Parallel_Vol_7d")

# ---------------------------------------------------------------------------
# 6. Target Variable — Price Jump Target (binary)
# ---------------------------------------------------------------------------
# "Will the official rate increase by > 1% within the next 7 days?"
# shift(-1): start from tomorrow (no same-day leakage)
# rolling(7).max(): highest rate in that 7-day forward window
future_max = df['Official_Rate'].shift(-1).rolling(window=7, min_periods=1).max()

# A jump = future max is > 1% above today's rate
df['Price_Jump_Target'] = ((future_max - df['Official_Rate']) / df['Official_Rate']) > 0.01

print("[OK] Price_Jump_Target: >1% official rate increase within 7 days")

# ---------------------------------------------------------------------------
# 7. Pre-Pruning NaN Audit
# ---------------------------------------------------------------------------
print("\n--- Pre-Pruning NaN Audit ---")
print(f"{'Column':<25} {'Non-Null':>10} {'NaN':>10}")
print("-" * 47)
for col in df.columns:
    non_null = df[col].notna().sum()
    null = df[col].isna().sum()
    print(f"{col:<25} {non_null:>10} {null:>10}")

# ---------------------------------------------------------------------------
# 8. Data Pruning
# ---------------------------------------------------------------------------
print("\n--- Data Pruning ---")
rows_before = len(df)

# Step 1: Filter to parallel overlap window (where Parallel_Rate is not NaN)
df = df[df['Parallel_Rate'].notna()].copy()
print(f"[1] Filter to parallel window: {rows_before} -> {len(df)} rows")

# Step 2: Drop any remaining NaN rows (from lags, rolling windows, target)
rows_before_dropna = len(df)
df = df.dropna().copy()
print(f"[2] Drop residual NaN:          {rows_before_dropna} -> {len(df)} rows")

# Convert target to proper boolean (dropna may have changed dtype)
df['Price_Jump_Target'] = df['Price_Jump_Target'].astype(bool)

# Convert Days_Since_Adjustment to int
df['Days_Since_Adjustment'] = df['Days_Since_Adjustment'].astype(int)

print(f"\n[OK] Final ML-ready dataset: {len(df)} rows x {len(df.columns)} columns")
print(f"     Date range: {df.index.min().date()} -> {df.index.max().date()}")

# ---------------------------------------------------------------------------
# 9. Automated Assertions
# ---------------------------------------------------------------------------
print("\n--- Validation Assertions ---")

# Assert 1: Zero NaN in final output
nan_total = df.isna().sum().sum()
assert nan_total == 0, f"Final dataset still contains {nan_total} NaN values!"
print("[PASS] Zero NaN in final output")

# Assert 2: Price_Jump_Target is boolean with both classes
assert df['Price_Jump_Target'].dtype == bool, \
    f"Price_Jump_Target is {df['Price_Jump_Target'].dtype}, expected bool!"
assert df['Price_Jump_Target'].nunique() == 2, \
    "Price_Jump_Target has only one class — both True and False must be present!"
print("[PASS] Price_Jump_Target is boolean with both classes")

# Assert 3: Days_Since_Adjustment is non-negative integer
assert (df['Days_Since_Adjustment'] >= 0).all(), \
    "Days_Since_Adjustment contains negative values!"
print("[PASS] Days_Since_Adjustment is non-negative")

# Assert 4: Spot-check lag — Premium_Lag_1d at row i == Parallel_Premium at row i-1
# Pick a row in the middle of the dataset for the spot check
spot_idx = len(df) // 2
spot_row = df.iloc[spot_idx]
prev_row = df.iloc[spot_idx - 1]
lag_value = spot_row['Premium_Lag_1d']
actual_prev = prev_row['Parallel_Premium']
assert abs(lag_value - actual_prev) < 1e-10, \
    f"Look-ahead bias detected! Lag_1d={lag_value} but prev premium={actual_prev}"
print(f"[PASS] Spot-check: Premium_Lag_1d[{spot_idx}] = {lag_value:.6f} "
      f"matches Parallel_Premium[{spot_idx - 1}] = {actual_prev:.6f}")

# Assert 5: Final row count in expected range
assert 2500 <= len(df) <= 3200, \
    f"Final row count {len(df)} is outside expected range [2500, 3200]!"
print(f"[PASS] Row count {len(df)} is within expected range")

# ---------------------------------------------------------------------------
# 10. Target Class Balance
# ---------------------------------------------------------------------------
print("\n--- Target Class Balance ---")
true_count = df['Price_Jump_Target'].sum()
false_count = len(df) - true_count
true_pct = (true_count / len(df)) * 100
false_pct = (false_count / len(df)) * 100

print(f"  Total samples     : {len(df)}")
print(f"  True  (Jump)      : {true_count:>6}  ({true_pct:.1f}%)")
print(f"  False (No Jump)   : {false_count:>6}  ({false_pct:.1f}%)")
print(f"  Imbalance ratio   : 1:{false_count / max(true_count, 1):.1f} (True:False)")

if true_pct < 20:
    print("  [NOTE] Dataset is imbalanced. Consider SMOTE, class weights, or")
    print("         stratified sampling in Phase 3 model training.")
elif true_pct > 40:
    print("  [NOTE] Dataset is relatively balanced. Standard training should work.")
else:
    print("  [NOTE] Moderate imbalance. Class weights recommended.")

# ---------------------------------------------------------------------------
# 11. Save ML-ready dataset
# ---------------------------------------------------------------------------
df.to_csv(OUTPUT_CSV)
print(f"\n[OK] Saved ML-ready dataset to: {OUTPUT_CSV}")

# ---------------------------------------------------------------------------
# 12. Diagnostic Statistics Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("  DIAGNOSTIC STATISTICS -- Phase 2 Feature Engineering")
print("=" * 65)
print(f"  Input rows               : 12498")
print(f"  Output rows              : {len(df)}")
print(f"  Output columns           : {len(df.columns)}")
print(f"  Date range               : {df.index.min().date()} -> {df.index.max().date()}")
print(f"  Features (model inputs)  : {len(df.columns) - 1}")
print(f"  Target column            : Price_Jump_Target")
print(f"  Target True%             : {true_pct:.1f}%")
print(f"  Max Days_Since_Adjustment: {df['Days_Since_Adjustment'].max()}")
print(f"  Output CSV               : {OUTPUT_CSV}")
print(f"  Output Heatmap           : {OUTPUT_HEATMAP}")
print("=" * 65)

# ---------------------------------------------------------------------------
# 13. Feature Correlation Heatmap
# ---------------------------------------------------------------------------
# Select only numeric columns for correlation (exclude boolean target)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 11), dpi=150)

# Create heatmap manually (no seaborn dependency required)
im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

# Add colorbar
cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Pearson Correlation', fontsize=12)

# Set ticks and labels
ax.set_xticks(range(len(numeric_cols)))
ax.set_yticks(range(len(numeric_cols)))
ax.set_xticklabels(numeric_cols, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(numeric_cols, fontsize=9)

# Annotate cells with correlation values
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        val = corr_matrix.values[i, j]
        color = 'white' if abs(val) > 0.6 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=7, color=color, fontweight='bold')

ax.set_title('Phase 2: Feature Correlation Heatmap\n(ML-Ready Dataset)',
             fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig(OUTPUT_HEATMAP, bbox_inches='tight')
plt.close()

print(f"\n[OK] Correlation heatmap saved to: {OUTPUT_HEATMAP}")
print("\n[DONE] Phase 2 Feature Engineering complete!")
