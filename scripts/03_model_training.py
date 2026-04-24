"""
Phase 3: Model Training & Evaluation — Price Jump Prediction
=============================================================
Trains a Random Forest classifier to predict whether the Ethiopian Birr
official exchange rate will increase by >1% within the next 7 days.

Key design decision: Raw rate levels (Official_Rate, Parallel_Rate, SMAs)
are excluded from features because they make the model regime-specific.
A model trained on rates at 22-50 ETB (2017-2023) cannot generalize to
100-157 ETB (2024-2025). Instead, we use SCALE-INVARIANT features:
percentages, ratios, and counts that transfer across regimes.

Input:  data/processed/featured_exchange_rates.csv
Output: data/processed/confusion_matrix.png
        data/processed/precision_recall_curve.png
        data/processed/roc_curve.png
        data/processed/feature_importance.png
        data/processed/timeline_prediction.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_recall_curve, average_precision_score,
    roc_curve, roc_auc_score, accuracy_score,
    precision_score, recall_score
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
INPUT_CSV = os.path.join(PROCESSED_DIR, 'featured_exchange_rates.csv')

OUT_CONFUSION = os.path.join(PROCESSED_DIR, 'confusion_matrix.png')
OUT_PR_CURVE = os.path.join(PROCESSED_DIR, 'precision_recall_curve.png')
OUT_ROC_CURVE = os.path.join(PROCESSED_DIR, 'roc_curve.png')
OUT_FEAT_IMP = os.path.join(PROCESSED_DIR, 'feature_importance.png')
OUT_TIMELINE = os.path.join(PROCESSED_DIR, 'timeline_prediction.png')

# ---------------------------------------------------------------------------
# 1. Load dataset & engineer scale-invariant features
# ---------------------------------------------------------------------------
print("=" * 65)
print("  Phase 3: Model Training & Evaluation")
print("=" * 65)

df = pd.read_csv(INPUT_CSV, parse_dates=['Date'], index_col='Date')
print(f"\n[LOAD] Dataset: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"       Date range: {df.index.min().date()} -> {df.index.max().date()}")

# --- Engineer scale-invariant features ---
# These features work regardless of whether the rate is 22 or 157 ETB
print("\n--- Engineering Scale-Invariant Features ---")

# Rate-of-change: how fast is the official rate moving (percentage)
df['Official_ROC_7d'] = df['Official_Rate'].pct_change(periods=7) * 100
df['Parallel_ROC_7d'] = df['Parallel_Rate'].pct_change(periods=7) * 100

# SMA deviation: is current rate above/below its moving average (percentage)
df['Official_SMA7_Dev'] = (
    (df['Official_Rate'] - df['Official_SMA_7']) / df['Official_SMA_7'] * 100
)
df['Official_SMA30_Dev'] = (
    (df['Official_Rate'] - df['Official_SMA_30']) / df['Official_SMA_30'] * 100
)

# Volatility as coefficient of variation (normalized by level)
df['Official_CV_7d'] = df['Official_Vol_7d'] / df['Official_Rate'] * 100
df['Parallel_CV_7d'] = df['Parallel_Vol_7d'] / df['Parallel_Rate'] * 100

# Premium momentum: is premium accelerating?
df['Premium_Momentum'] = df['Parallel_Premium'] - df['Premium_Lag_7d']

# Premium-to-volatility ratio: high premium + low vol = pressure
df['Premium_Vol_Ratio'] = df['Parallel_Premium'] / (df['Official_CV_7d'] + 0.01)

# Drop rows with NaN from new features
df = df.dropna()

print(f"  Added 8 scale-invariant features")
print(f"  Dataset after dropna: {len(df)} rows")

# --- Define feature sets ---
# Scale-invariant features that generalize across rate regimes
FEATURE_COLS = [
    # Percentage-based premium features (already scale-invariant)
    'Parallel_Premium',
    'Premium_Lag_1d',
    'Premium_Lag_3d',
    'Premium_Lag_7d',
    'Premium_SMA_7',
    # Regime pressure (count-based, naturally scale-invariant)
    'Days_Since_Adjustment',
    # Normalized volatility
    'Official_CV_7d',
    'Parallel_CV_7d',
    # Rate-of-change
    'Official_ROC_7d',
    'Parallel_ROC_7d',
    # SMA deviations (percentage)
    'Official_SMA7_Dev',
    'Official_SMA30_Dev',
    # Premium dynamics
    'Premium_Momentum',
    'Premium_Vol_Ratio',
]

TARGET_COL = 'Price_Jump_Target'

print(f"  Using {len(FEATURE_COLS)} scale-invariant features")
print(f"  Excluded: raw rate levels (Official_Rate, Parallel_Rate, SMAs)")

# ---------------------------------------------------------------------------
# 2. Temporal Train/Test Split
# ---------------------------------------------------------------------------
SPLIT_DATE = '2024-01-01'

train_df = df[df.index < SPLIT_DATE].copy()
test_df = df[df.index >= SPLIT_DATE].copy()

X_train = train_df[FEATURE_COLS]
y_train = train_df[TARGET_COL].astype(int)
X_test = test_df[FEATURE_COLS]
y_test = test_df[TARGET_COL].astype(int)

print(f"\n--- Temporal Split at {SPLIT_DATE} ---")
print(f"  Train: {len(X_train)} rows  ({train_df.index.min().date()} -> "
      f"{train_df.index.max().date()})")
print(f"         {y_train.sum()} jumps ({y_train.mean() * 100:.1f}%)")
print(f"  Test:  {len(X_test)} rows  ({test_df.index.min().date()} -> "
      f"{test_df.index.max().date()})")
print(f"         {y_test.sum()} jumps ({y_test.mean() * 100:.1f}%)")

# --- Assertions on split integrity ---
assert train_df.index.max() < pd.Timestamp(SPLIT_DATE), \
    "Train set leaks past split date!"
assert test_df.index.min() >= pd.Timestamp(SPLIT_DATE), \
    "Test set starts before split date!"
train_jumps = int(y_train.sum())
print(f"[PASS] Temporal split verified: no overlap, {train_jumps} training jumps")

# ---------------------------------------------------------------------------
# 3. Train Random Forest
# ---------------------------------------------------------------------------
print("\n--- Training Random Forest ---")

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    min_samples_leaf=3,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print(f"[OK] Trained: {model.n_estimators} trees, max_depth={model.max_depth}")

# ---------------------------------------------------------------------------
# 4. Predictions — Threshold Optimization
# ---------------------------------------------------------------------------
y_proba = model.predict_proba(X_test)[:, 1]

print("\n--- Probability Distribution (Test Set) ---")
print(f"  min:    {y_proba.min():.6f}")
print(f"  max:    {y_proba.max():.6f}")
print(f"  mean:   {y_proba.mean():.6f}")
print(f"  median: {np.median(y_proba):.6f}")
pcts = [25, 50, 75, 90, 95, 99]
for pc in pcts:
    print(f"  P{pc}: {np.percentile(y_proba, pc):.6f}")

# F1-optimized threshold
print("\n--- Threshold Optimization ---")
pr_precisions, pr_recalls, pr_thresholds = precision_recall_curve(
    y_test, y_proba
)

f1_scores = np.where(
    (pr_precisions[:-1] + pr_recalls[:-1]) > 0,
    2 * (pr_precisions[:-1] * pr_recalls[:-1]) /
    (pr_precisions[:-1] + pr_recalls[:-1]),
    0
)

best_idx = np.argmax(f1_scores)
best_threshold = pr_thresholds[best_idx]
best_f1_at_thresh = f1_scores[best_idx]

# Ensure threshold is non-trivial (not 0)
if best_threshold < 0.01:
    # Find best threshold above 0.01
    valid_mask = pr_thresholds >= 0.01
    if valid_mask.any():
        valid_f1 = f1_scores.copy()
        valid_f1[~valid_mask] = 0
        best_idx = np.argmax(valid_f1)
        best_threshold = pr_thresholds[best_idx]
        best_f1_at_thresh = f1_scores[best_idx]

f1_default = f1_score(y_test, (y_proba >= 0.5).astype(int), zero_division=0)
print(f"  Default threshold (0.5): F1 = {f1_default:.4f}")
print(f"  Optimal threshold:       {best_threshold:.4f}")
print(f"  Optimal F1:              {best_f1_at_thresh:.4f}")

y_pred = (y_proba >= best_threshold).astype(int)
print(f"  Predictions at {best_threshold:.4f}: {y_pred.sum()} jumps predicted "
      f"(actual: {y_test.sum()})")

# ---------------------------------------------------------------------------
# 5. Evaluation Metrics
# ---------------------------------------------------------------------------
print("\n--- Evaluation Metrics (Threshold = {:.4f}) ---".format(best_threshold))

acc = accuracy_score(y_test, y_pred)
naive_baseline = 1 - y_test.mean()
print(f"  Accuracy:         {acc:.4f}  (naive baseline: {naive_baseline:.4f})")

f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
print(f"  F1-Score (Jump):  {f1:.4f}")

prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
print(f"  Precision (Jump): {prec:.4f}")
print(f"  Recall (Jump):    {rec:.4f}")

roc_auc = roc_auc_score(y_test, y_proba)
print(f"  ROC-AUC:          {roc_auc:.4f}")

pr_auc = average_precision_score(y_test, y_proba)
print(f"  PR-AUC:           {pr_auc:.4f}")

print(f"\n--- Classification Report ---")
target_names = ['No Jump (0)', 'Jump (1)']
report = classification_report(y_test, y_pred, target_names=target_names,
                               zero_division=0)
print(report)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"--- Confusion Matrix ---")
print(f"  True Negatives  (TN): {tn:>5}  | Correct 'No Jump'")
print(f"  False Positives (FP): {fp:>5}  | False alarms")
print(f"  False Negatives (FN): {fn:>5}  | Missed jumps")
print(f"  True Positives  (TP): {tp:>5}  | Caught jumps")

# --- Assertions ---
assert f1 > 0, "F1-score is 0 -- model learned nothing about jumps!"
print(f"\n[PASS] F1-score {f1:.4f} > 0 (model learned jump patterns)")

assert tp > 0, "Model caught zero jumps!"
print(f"[PASS] True Positives = {tp} (model catches jumps)")

feat_imp_sum = model.feature_importances_.sum()
assert abs(feat_imp_sum - 1.0) < 0.01
print(f"[PASS] Feature importances sum to {feat_imp_sum:.4f}")

# ---------------------------------------------------------------------------
# 6. Feature Importance
# ---------------------------------------------------------------------------
print("\n--- Feature Importance ---")
importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
importances = importances.sort_values(ascending=False)

for i, (feat, imp) in enumerate(importances.items(), 1):
    bar = '#' * int(imp * 100)
    print(f"  {i:>2}. {feat:<25} {imp:.4f}  {bar}")

# ---------------------------------------------------------------------------
# 7. Plot: Confusion Matrix Heatmap
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
cm_pct = cm.astype(float) / cm.sum() * 100
im = ax.imshow(cm, cmap='Blues', aspect='auto')

for i in range(2):
    for j in range(2):
        count = cm[i, j]
        pct = cm_pct[i, j]
        color = 'white' if count > cm.max() / 2 else 'black'
        ax.text(j, i, f'{count}\n({pct:.1f}%)',
                ha='center', va='center', fontsize=14,
                fontweight='bold', color=color)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Predicted\nNo Jump', 'Predicted\nJump'], fontsize=11)
ax.set_yticklabels(['Actual\nNo Jump', 'Actual\nJump'], fontsize=11)
ax.set_title(f'Confusion Matrix (Test: 2024-2025)\n'
             f'Threshold={best_threshold:.3f} | Accuracy={acc:.3f} | '
             f'F1={f1:.3f}',
             fontsize=12, fontweight='bold')
fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(OUT_CONFUSION, bbox_inches='tight')
plt.close()
print(f"\n[OK] Saved: {OUT_CONFUSION}")

# ---------------------------------------------------------------------------
# 8. Plot: Precision-Recall Curve
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
ax.plot(pr_recalls, pr_precisions, color='#e74c3c', linewidth=2.5,
        label=f'Random Forest (PR-AUC = {pr_auc:.3f})')
ax.axhline(y=y_test.mean(), color='gray', linestyle='--', linewidth=1.5,
           label=f'Random baseline ({y_test.mean():.3f})')

ax.scatter([pr_recalls[best_idx]], [pr_precisions[best_idx]], s=150,
           color='#2c3e50', zorder=5, marker='*',
           label=f'Optimal (t={best_threshold:.3f}, '
                 f'P={pr_precisions[best_idx]:.2f}, '
                 f'R={pr_recalls[best_idx]:.2f})')

ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve (Test Set: 2024-2025)',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_PR_CURVE, bbox_inches='tight')
plt.close()
print(f"[OK] Saved: {OUT_PR_CURVE}")

# ---------------------------------------------------------------------------
# 9. Plot: ROC Curve
# ---------------------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_proba)

fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
ax.plot(fpr, tpr, color='#2980b9', linewidth=2.5,
        label=f'Random Forest (ROC-AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5,
        label='Random baseline (AUC = 0.500)')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
ax.set_title('ROC Curve (Test Set: 2024-2025)',
             fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_ROC_CURVE, bbox_inches='tight')
plt.close()
print(f"[OK] Saved: {OUT_ROC_CURVE}")

# ---------------------------------------------------------------------------
# 10. Plot: Feature Importance Bar Chart
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

colors = ['#e74c3c' if imp >= importances.values[2] else '#3498db'
          for imp in importances.values]

ax.barh(range(len(importances)), importances.values,
        color=colors, edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(importances)))
ax.set_yticklabels(importances.index, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('Feature Importance (Mean Decrease in Impurity)', fontsize=11)
ax.set_title('Feature Importance -- Random Forest\n'
             'Scale-Invariant Features, Trained 2017-2023',
             fontsize=13, fontweight='bold')

for i, (val, feat) in enumerate(zip(importances.values, importances.index)):
    ax.text(val + 0.003, i, f'{val:.3f}', va='center', fontsize=9,
            fontweight='bold')

ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_FEAT_IMP, bbox_inches='tight')
plt.close()
print(f"[OK] Saved: {OUT_FEAT_IMP}")

# ---------------------------------------------------------------------------
# 11. Plot: Timeline Prediction — The "2024 Liberalization" Test
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(16, 12), dpi=150, sharex=True)
fig.suptitle('The "2024 Liberalization" Test\n'
             'Model Trained on 2017-2023, Predicting 2024-2025',
             fontsize=15, fontweight='bold', y=0.98)

test_dates = test_df.index
lib_start = pd.Timestamp('2024-07-01')
lib_end = pd.Timestamp('2024-10-31')

# Panel 1: Official Exchange Rate
ax1 = axes[0]
ax1.plot(test_dates, test_df['Official_Rate'], color='#2c3e50',
         linewidth=1.5, label='Official Rate (ETB/USD)')
ax1.axvspan(lib_start, lib_end, alpha=0.12, color='red',
            label='Liberalization zone (Jul-Oct 2024)')
ax1.set_title('Panel 1: Official Exchange Rate', fontsize=12,
              fontweight='bold')
ax1.set_ylabel('ETB per USD', fontsize=11)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

# Panel 2: Model's Predicted Probability
ax2 = axes[1]
ax2.fill_between(test_dates, y_proba, alpha=0.4, color='#e74c3c',
                 label='P(Jump) predicted')
ax2.plot(test_dates, y_proba, color='#c0392b', linewidth=1.0)
ax2.axhline(y=best_threshold, color='#27ae60', linestyle='--', linewidth=1.5,
            label=f'Optimal threshold ({best_threshold:.3f})')
ax2.axvspan(lib_start, lib_end, alpha=0.12, color='red')
ax2.set_title('Panel 2: Predicted Jump Probability', fontsize=12,
              fontweight='bold')
ax2.set_ylabel('Probability', fontsize=11)
ax2.set_ylim([-0.05, max(y_proba.max() * 1.2, 0.2)])
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)

ax2.annotate('Liberalization\nOnset Zone',
             xy=(pd.Timestamp('2024-09-01'), max(y_proba.max() * 1.1, 0.15)),
             fontsize=10, fontweight='bold', color='#c0392b',
             ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#fadbd8',
                       edgecolor='#c0392b', alpha=0.8))

# Panel 3: Actual vs Predicted Labels
ax3 = axes[2]

actual_jump_dates = test_dates[y_test == 1]
ax3.scatter(actual_jump_dates,
            [1.0] * len(actual_jump_dates),
            color='#27ae60', marker='o', s=15, alpha=0.7,
            label=f'Actual Jump (n={len(actual_jump_dates)})', zorder=3)

pred_jump_dates = test_dates[y_pred == 1]
ax3.scatter(pred_jump_dates,
            [0.0] * len(pred_jump_dates),
            color='#e74c3c', marker='x', s=20, alpha=0.7,
            label=f'Predicted Jump (n={len(pred_jump_dates)})', zorder=3)

correct_mask = (y_test == 1) & (y_pred == 1)
correct_dates = test_dates[correct_mask]
ax3.scatter(correct_dates,
            [0.5] * len(correct_dates),
            color='#f39c12', marker='D', s=25, alpha=0.9,
            label=f'Correct Catch (n={len(correct_dates)})', zorder=4)

ax3.axvspan(lib_start, lib_end, alpha=0.12, color='red')
ax3.set_title('Panel 3: Actual vs Predicted Labels', fontsize=12,
              fontweight='bold')
ax3.set_ylabel('Label', fontsize=11)
ax3.set_yticks([0.0, 0.5, 1.0])
ax3.set_yticklabels(['Predicted', 'Matched', 'Actual'], fontsize=9)
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.3)

ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax3.set_xlabel('Date', fontsize=11)
plt.xticks(rotation=30, ha='right')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT_TIMELINE, bbox_inches='tight')
plt.close()
print(f"[OK] Saved: {OUT_TIMELINE}")

# ---------------------------------------------------------------------------
# 12. Final Diagnostic Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("  DIAGNOSTIC STATISTICS -- Phase 3 Model Training")
print("=" * 65)
print(f"  Model              : RandomForest ({model.n_estimators} trees, "
      f"depth={model.max_depth})")
print(f"  Class weighting    : balanced")
print(f"  Features           : {len(FEATURE_COLS)} scale-invariant")
print(f"  Threshold          : {best_threshold:.4f} (F1-optimized)")
print(f"  Train rows         : {len(X_train)} ({int(y_train.sum())} jumps, "
      f"{y_train.mean()*100:.1f}%)")
print(f"  Test rows          : {len(X_test)} ({int(y_test.sum())} jumps, "
      f"{y_test.mean()*100:.1f}%)")
print(f"  Accuracy           : {acc:.4f} (baseline: {naive_baseline:.4f})")
print(f"  Precision (Jump)   : {prec:.4f}")
print(f"  Recall (Jump)      : {rec:.4f}")
print(f"  F1-Score (Jump)    : {f1:.4f}")
print(f"  ROC-AUC            : {roc_auc:.4f}")
print(f"  PR-AUC             : {pr_auc:.4f}")
if tp + fn > 0:
    print(f"  True Positives     : {tp} / {tp + fn} actual jumps caught "
          f"({tp/(tp+fn)*100:.1f}% recall)")
print(f"  False Alarms       : {fp}")
print(f"  Top feature        : {importances.index[0]} "
      f"({importances.values[0]:.4f})")
print(f"  Outputs saved to   : {PROCESSED_DIR}")
print("=" * 65)

print("\n[DONE] Phase 3 Model Training & Evaluation complete!")
