"""
Phase 1: Data Harmonization — Ethiopian FX Rate Merge
=====================================================
Merges two disparate exchange rate datasets into a single daily time-series:
  - wfp_official.csv  (daily official rates, ~8846 rows, 1992-2026)
  - wfp_parallel.csv  (monthly parallel/black-market rates, ~82 rows, 2017-2025)

Output:
  - data/processed/merged_exchange_rates.csv
  - data/processed/validation_chart.png
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import pchip_interpolate
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving charts
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

OFFICIAL_CSV = os.path.join(RAW_DIR, 'wfp_official.csv')
PARALLEL_CSV = os.path.join(RAW_DIR, 'wfp_parallel.csv')
OUTPUT_CSV = os.path.join(PROCESSED_DIR, 'merged_exchange_rates.csv')
OUTPUT_CHART = os.path.join(PROCESSED_DIR, 'validation_chart.png')

# ---------------------------------------------------------------------------
# 1. Create output directory
# ---------------------------------------------------------------------------
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Load raw data
# ---------------------------------------------------------------------------
official_df = pd.read_csv(OFFICIAL_CSV)
parallel_df = pd.read_csv(PARALLEL_CSV)

print(f"Raw Official rows : {len(official_df)}")
print(f"Raw Parallel rows : {len(parallel_df)}")

# ---------------------------------------------------------------------------
# 3. Date parsing — explicit dayfirst=True to prevent DD/MM ↔ MM/DD flip
# ---------------------------------------------------------------------------
official_df['Date'] = pd.to_datetime(official_df['Date'], dayfirst=True)
parallel_df['Date'] = pd.to_datetime(parallel_df['Date'], dayfirst=True)

# --- Assertions: catch date-parsing errors ---
assert official_df['Date'].min() == pd.Timestamp('1992-02-04'), \
    f"Date parsing error! Official min date is {official_df['Date'].min()}, expected 1992-02-04 (day/month may be flipped)"
assert parallel_df['Date'].min() == pd.Timestamp('2017-02-15'), \
    f"Date parsing error! Parallel min date is {parallel_df['Date'].min()}, expected 2017-02-15 (day/month may be flipped)"
assert official_df['Date'].notna().all(), \
    "Official dates contain NaT values after parsing!"
assert parallel_df['Date'].notna().all(), \
    "Parallel dates contain NaT values after parsing!"

print(f"\n[OK] Date parsing verified:")
print(f"   Official range : {official_df['Date'].min().date()} -> {official_df['Date'].max().date()}")
print(f"   Parallel range : {parallel_df['Date'].min().date()} -> {parallel_df['Date'].max().date()}")

# ---------------------------------------------------------------------------
# 4. Keep only the columns we need & sort by date
# ---------------------------------------------------------------------------
official_df = official_df[['Date', 'Value']].rename(columns={'Value': 'Official_Rate'})
official_df = official_df.sort_values('Date').drop_duplicates(subset='Date', keep='last').reset_index(drop=True)

parallel_df = parallel_df[['Date', 'Value']].rename(columns={'Value': 'Parallel_Rate'})
parallel_df = parallel_df.sort_values('Date').drop_duplicates(subset='Date', keep='last').reset_index(drop=True)

# ---------------------------------------------------------------------------
# 5. Build continuous daily index from official dataset's full range
# ---------------------------------------------------------------------------
full_index = pd.date_range(
    start=official_df['Date'].min(),
    end=official_df['Date'].max(),
    freq='D'
)

merged = pd.DataFrame(index=full_index)
merged.index.name = 'Date'

print(f"\n[OK] Continuous daily index: {len(merged)} days "
      f"({merged.index.min().date()} -> {merged.index.max().date()})")

# ---------------------------------------------------------------------------
# 6. Official Rate — reindex onto daily calendar, then forward-fill
# ---------------------------------------------------------------------------
official_series = official_df.set_index('Date')['Official_Rate']
merged['Official_Rate'] = official_series.reindex(full_index).ffill()

official_nan_count = merged['Official_Rate'].isna().sum()
print(f"\n[OK] Official_Rate NaN after forward-fill: {official_nan_count}")

# If there are NaN at the very start (before first observation), back-fill just those
if official_nan_count > 0:
    merged['Official_Rate'] = merged['Official_Rate'].bfill()
    print(f"   (Applied bfill for {official_nan_count} leading NaN values)")

assert merged['Official_Rate'].notna().all(), \
    "Official_Rate still contains NaN after forward-fill + back-fill!"

# ---------------------------------------------------------------------------
# 7. Parallel Rate — PCHIP interpolation (within observed range only)
# ---------------------------------------------------------------------------
parallel_start = parallel_df['Date'].min()
parallel_end = parallel_df['Date'].max()

# Convert known dates to ordinal (numeric) for scipy
known_dates_ordinal = parallel_df['Date'].map(pd.Timestamp.toordinal).values
known_values = parallel_df['Parallel_Rate'].values

# Target: all daily dates within the parallel observation window
target_mask = (full_index >= parallel_start) & (full_index <= parallel_end)
target_dates = full_index[target_mask]
target_dates_ordinal = pd.Series(target_dates).map(pd.Timestamp.toordinal).values

# PCHIP interpolation — smooth, monotonicity-preserving cubic
interpolated_values = pchip_interpolate(known_dates_ordinal, known_values, target_dates_ordinal)

# Assign: NaN outside the parallel range, interpolated within
merged['Parallel_Rate'] = np.nan
merged.loc[target_mask, 'Parallel_Rate'] = interpolated_values

parallel_nan_total = merged['Parallel_Rate'].isna().sum()
parallel_valid = merged['Parallel_Rate'].notna().sum()
print(f"\n[OK] Parallel_Rate:")
print(f"   Interpolated days (within 2017-2025 window): {parallel_valid}")
print(f"   NaN days (outside parallel range, expected) : {parallel_nan_total}")

# ---------------------------------------------------------------------------
# 8. Feature Engineering — Parallel Premium (%)
# ---------------------------------------------------------------------------
merged['Parallel_Premium'] = (
    (merged['Parallel_Rate'] - merged['Official_Rate']) / merged['Official_Rate']
) * 100

premium_valid = merged['Parallel_Premium'].notna().sum()
print(f"\n[OK] Parallel_Premium:")
print(f"   Computed for {premium_valid} days")
print(f"   Range: {merged['Parallel_Premium'].min():.2f}% -> {merged['Parallel_Premium'].max():.2f}%")

# ---------------------------------------------------------------------------
# 9. Save merged CSV
# ---------------------------------------------------------------------------
merged.to_csv(OUTPUT_CSV)
print(f"\n[OK] Saved merged dataset to: {OUTPUT_CSV}")
print(f"   Shape: {merged.shape[0]} rows x {merged.shape[1]} columns")

# ---------------------------------------------------------------------------
# 10. Diagnostic Statistics Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("  DIAGNOSTIC STATISTICS — Phase 1 Data Harmonization")
print("=" * 65)
print(f"  Official CSV rows parsed     : {len(official_df)}")
print(f"  Parallel CSV rows parsed     : {len(parallel_df)}")
print(f"  Official date range          : {official_df['Date'].min().date()} -> {official_df['Date'].max().date()}")
print(f"  Parallel date range          : {parallel_df['Date'].min().date()} -> {parallel_df['Date'].max().date()}")
print(f"  Merged daily index           : {len(merged)} rows")
print(f"  Official_Rate NaN after fill : 0")
print(f"  Parallel_Rate valid days     : {parallel_valid}")
print(f"  Parallel_Rate NaN (expected) : {parallel_nan_total}")
print(f"  Parallel_Premium range       : {merged['Parallel_Premium'].min():.2f}% -> {merged['Parallel_Premium'].max():.2f}%")
print(f"  Output CSV                   : {OUTPUT_CSV}")
print(f"  Output Chart                 : {OUTPUT_CHART}")
print("=" * 65)

# ---------------------------------------------------------------------------
# 11. Validation Chart — 3-panel figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(16, 14), dpi=150)
fig.suptitle('Phase 1: Data Harmonization — Validation Chart',
             fontsize=16, fontweight='bold', y=0.98)

# --- Panel 1: Full Timeline — Official Rate ---
ax1 = axes[0]
ax1.plot(merged.index, merged['Official_Rate'],
         color='#1f77b4', linewidth=0.8, label='Official Rate (forward-filled)')
ax1.set_title('Panel 1: Official Exchange Rate — Full Timeline (1992–2026)',
              fontsize=12, fontweight='bold')
ax1.set_ylabel('ETB per USD', fontsize=11)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_locator(mdates.YearLocator(5))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# --- Panel 2: Overlap Period — Official + Parallel + Original Scatter ---
ax2 = axes[1]
overlap_mask = merged['Parallel_Rate'].notna()
overlap_data = merged.loc[overlap_mask]

ax2.plot(overlap_data.index, overlap_data['Official_Rate'],
         color='#1f77b4', linewidth=1.2, label='Official Rate', zorder=2)
ax2.plot(overlap_data.index, overlap_data['Parallel_Rate'],
         color='#d62728', linewidth=1.2, label='Parallel Rate (PCHIP)', zorder=3)
ax2.scatter(parallel_df['Date'], parallel_df['Parallel_Rate'],
            color='#d62728', s=30, edgecolors='black', linewidths=0.5,
            zorder=4, label=f'Original observations (n={len(parallel_df)})')
ax2.set_title('Panel 2: Official vs Parallel Rate — Overlap Period (2017–2025)',
              fontsize=12, fontweight='bold')
ax2.set_ylabel('ETB per USD', fontsize=11)
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# --- Panel 3: Parallel Premium ---
ax3 = axes[2]
premium_data = merged.loc[overlap_mask, 'Parallel_Premium']
ax3.plot(premium_data.index, premium_data.values,
         color='#2ca02c', linewidth=1.0, label='Parallel Premium (%)')
ax3.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax3.set_title('Panel 3: Parallel Premium (%) — Overlap Period (2017–2025)',
              fontsize=12, fontweight='bold')
ax3.set_ylabel('Premium (%)', fontsize=11)
ax3.set_xlabel('Date', fontsize=11)
ax3.legend(loc='upper left', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_locator(mdates.YearLocator())
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_CHART, bbox_inches='tight')
plt.close()

print(f"\n[OK] Validation chart saved to: {OUTPUT_CHART}")
print("\n[DONE] Phase 1 Data Harmonization complete!")
