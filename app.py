"""
Ethiopia FX Early-Warning System — Streamlit Dashboard
========================================================
Interactive dashboard for monitoring Ethiopian Birr exchange rate
regime stability and visualizing model predictions.

Run: streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Ethiopia FX Early-Warning System",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FEATURED_CSV = os.path.join(PROCESSED_DIR, "featured_exchange_rates.csv")

# Image paths
IMG_TIMELINE = os.path.join(PROCESSED_DIR, "timeline_prediction.png")
IMG_CONFUSION = os.path.join(PROCESSED_DIR, "confusion_matrix.png")
IMG_PR_CURVE = os.path.join(PROCESSED_DIR, "precision_recall_curve.png")
IMG_ROC_CURVE = os.path.join(PROCESSED_DIR, "roc_curve.png")
IMG_FEAT_IMP = os.path.join(PROCESSED_DIR, "feature_importance.png")
IMG_HEATMAP = os.path.join(PROCESSED_DIR, "feature_heatmap.png")

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Header styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #e74c3c;
        margin-bottom: 0;
        line-height: 1.2;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #7f8c8d;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }

    /* Status indicators */
    .status-green {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        color: white;
        padding: 12px 16px;
        border-radius: 10px;
        text-align: center;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .status-yellow {
        background: linear-gradient(135deg, #f39c12, #f1c40f);
        color: #2c3e50;
        padding: 12px 16px;
        border-radius: 10px;
        text-align: center;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .status-red {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        color: white;
        padding: 12px 16px;
        border-radius: 10px;
        text-align: center;
        font-weight: 700;
        font-size: 0.95rem;
    }

    /* Sidebar section headers */
    .sidebar-header {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #95a5a6;
        margin-top: 1.2rem;
        margin-bottom: 0.5rem;
    }

    /* Finding cards */
    .finding-card {
        background-color: rgba(39, 174, 96, 0.08);
        border-left: 4px solid #27ae60;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
    }
    .finding-card-warn {
        background-color: rgba(243, 156, 18, 0.08);
        border-left: 4px solid #f39c12;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
    }
    .finding-card-alert {
        background-color: rgba(231, 76, 60, 0.08);
        border-left: 4px solid #e74c3c;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
    }

    /* Divider */
    .sidebar-divider {
        border: none;
        border-top: 1px solid rgba(150,150,150,0.2);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load the featured exchange rates dataset."""
    df = pd.read_csv(FEATURED_CSV, parse_dates=["Date"], index_col="Date")
    return df


df = load_data()
latest = df.iloc[-1]
prev = df.iloc[-2]

# Derived values
latest_official = latest["Official_Rate"]
prev_official = prev["Official_Rate"]
official_delta = latest_official - prev_official
official_delta_pct = (official_delta / prev_official) * 100

latest_premium = latest["Parallel_Premium"]
prev_premium = prev["Parallel_Premium"]
premium_delta = latest_premium - prev_premium

latest_dsa = int(latest["Days_Since_Adjustment"])

# System status logic
if latest_premium < 20:
    status_level = "green"
    status_text = "Monitoring Regime Stability"
    status_icon = "🟢"
elif latest_premium <= 50:
    status_level = "yellow"
    status_text = "Elevated Premium — Watch"
    status_icon = "🟡"
else:
    status_level = "red"
    status_text = "High Premium — Alert"
    status_icon = "🔴"


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<p class="main-title">📉 Ethiopia FX</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Early-Warning System</p>', unsafe_allow_html=True
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # --- Latest Data ---
    st.markdown(
        '<p class="sidebar-header">📊 Latest Data</p>', unsafe_allow_html=True
    )

    st.metric(
        label="Official Rate (ETB/USD)",
        value=f"{latest_official:.2f}",
        delta=f"{official_delta:+.2f} ({official_delta_pct:+.1f}%)",
    )

    st.metric(
        label="Parallel Premium",
        value=f"{latest_premium:.1f}%",
        delta=f"{premium_delta:+.1f}pp",
        delta_color="inverse",
    )

    st.metric(
        label="Days Since Adjustment",
        value=f"{latest_dsa} days",
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # --- System Status ---
    st.markdown(
        '<p class="sidebar-header">🔔 System Status</p>', unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="status-{status_level}">'
        f"{status_icon} {status_text}</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # --- Model Stats ---
    st.markdown(
        '<p class="sidebar-header">📈 Model Performance</p>', unsafe_allow_html=True
    )

    st.caption("Random Forest (500 trees, depth=8)")
    col_a, col_b = st.columns(2)
    col_a.metric("F1-Score", "0.535")
    col_b.metric("Recall", "55.2%")
    col_a.metric("ROC-AUC", "0.706")
    col_b.metric("PR-AUC", "0.505")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.caption(f"Data: WFP VAM | Range: {df.index.min().date()} to {df.index.max().date()}")


# ---------------------------------------------------------------------------
# MAIN CONTENT — Header
# ---------------------------------------------------------------------------
st.markdown(
    '<p class="main-title">Ethiopia FX Early-Warning System</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtitle">'
    "Predictive Modeling of FX Regime Shifts &mdash; Monitoring the "
    "2024 Liberalization and Beyond"
    "</p>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📈 The Forecast", "🧠 The Logic", "📋 Raw Data"])


# ========================== TAB 1: THE FORECAST ============================
with tab1:
    st.markdown("### The \"2024 Liberalization\" Test")
    st.markdown(
        "Can a model trained on **2017-2023** (managed peg era) detect the onset "
        "of Ethiopia's historic FX liberalization in **2024**?"
    )

    # Hero image
    if os.path.exists(IMG_TIMELINE):
        st.image(
            IMG_TIMELINE,
            caption=(
                "Figure 1: Model Predicted Probability vs. Actual Official Rate "
                "during the 2024 FX Liberalization. The model, trained exclusively "
                "on 2017-2023 data, shows a probability spike to 0.42 in late "
                "June 2024 — weeks before Ethiopia's historic exchange rate reform."
            ),
            use_container_width=True,
        )
    else:
        st.warning("Timeline prediction chart not found. Run `03_model_training.py` first.")

    # Key findings
    st.markdown("### Key Findings")

    st.markdown(
        '<div class="finding-card">'
        "<strong>Early Warning Confirmed</strong> — The model's predicted "
        "probability spiked to <strong>0.42</strong> in late June 2024, "
        "<em>weeks before</em> the official rate jumped from 58 to 120+ ETB/USD."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="finding-card">'
        "<strong>55.2% Recall</strong> — 80 out of 145 actual jump events "
        "were correctly flagged by a model that has never seen a "
        "liberalization event."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="finding-card-warn">'
        "<strong>74 False Alarms</strong> — Acceptable for an early-warning "
        "system where the cost of missing a devaluation far exceeds the cost "
        "of a false alert."
        "</div>",
        unsafe_allow_html=True,
    )

    # Metrics row
    st.markdown("### Performance Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", "52.0%")
    m2.metric("Recall", "55.2%")
    m3.metric("F1-Score", "0.535")
    m4.metric("ROC-AUC", "0.706")

    # Confusion matrix + PR curve side by side
    st.markdown("### Detailed Evaluation")
    col1, col2 = st.columns(2)

    with col1:
        if os.path.exists(IMG_CONFUSION):
            st.image(
                IMG_CONFUSION,
                caption=(
                    "Figure 2: Confusion Matrix — 313 True Negatives, "
                    "80 True Positives, 74 False Alarms, 65 Missed Jumps."
                ),
                use_container_width=True,
            )

    with col2:
        if os.path.exists(IMG_PR_CURVE):
            st.image(
                IMG_PR_CURVE,
                caption=(
                    "Figure 3: Precision-Recall Curve (PR-AUC = 0.505). "
                    "The star marks the F1-optimized operating threshold."
                ),
                use_container_width=True,
            )


# ========================== TAB 2: THE LOGIC ===============================
with tab2:
    st.markdown("### The Scale-Invariant Pivot")

    st.error(
        "**The initial model using raw rate levels COMPLETELY FAILED** "
        "(ROC-AUC = 0.478, worse than random). A model trained on rates at "
        "22-50 ETB cannot extrapolate to 100-157 ETB. All test probabilities "
        "were below 0.005."
    )

    st.success(
        "**The fix:** Replace absolute rate levels with **percentage-based, "
        "scale-invariant features** that work at any rate level. A \"2% "
        "deviation from the 7-day moving average\" means the same thing "
        "whether the rate is 30 ETB or 130 ETB."
    )

    # Feature importance chart
    st.markdown("### What Drives the Model's Predictions?")

    if os.path.exists(IMG_FEAT_IMP):
        st.image(
            IMG_FEAT_IMP,
            caption=(
                "Figure 4: Feature Importance (Mean Decrease in Impurity). "
                "Top 3 features — all scale-invariant — account for 55.5% "
                "of total importance."
            ),
            use_container_width=True,
        )

    # Top 3 features
    st.markdown("### Top 3 Features")
    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown("##### 🥇 Official\\_SMA7\\_Dev")
        st.markdown("**27.0%** importance")
        st.caption(
            "How far the current rate deviates from its 7-day moving average. "
            "The strongest signal for imminent adjustments."
        )

    with f2:
        st.markdown("##### 🥈 Official\\_CV\\_7d")
        st.markdown("**15.5%** importance")
        st.caption(
            "Coefficient of variation — volatility normalized by rate level. "
            "Captures market nervousness independent of scale."
        )

    with f3:
        st.markdown("##### 🥉 Official\\_ROC\\_7d")
        st.markdown("**13.0%** importance")
        st.caption(
            "7-day rate of change in percentage terms. "
            "Momentum signal that transfers across any rate level."
        )

    # Surprise finding
    st.markdown("### The Days\\_Since\\_Adjustment Surprise")

    st.markdown(
        '<div class="finding-card-alert">'
        '<strong>Ranked LAST at 0.6% importance</strong> — Our "Pressure '
        'Cooker" hypothesis was not confirmed. The 2021-2023 training period '
        "had <strong>3 consecutive years</strong> of complete rate stasis "
        "with <strong>zero jumps</strong>, teaching the model that extended "
        "flat periods do <em>not</em> predict imminent regime change. In a "
        "managed peg, stasis is the <em>norm</em>, not a warning sign."
        "</div>",
        unsafe_allow_html=True,
    )

    # Correlation heatmap
    st.markdown("### Feature Correlation Matrix")

    if os.path.exists(IMG_HEATMAP):
        st.image(
            IMG_HEATMAP,
            caption=(
                "Figure 5: Pearson correlation matrix of engineered features. "
                "Note the tight clustering of premium features (r > 0.99) and "
                "the independence of Days_Since_Adjustment."
            ),
            use_container_width=True,
        )


# ========================== TAB 3: RAW DATA ================================
with tab3:
    st.markdown("### Latest 10 Days — Featured Exchange Rates")

    # Display last 10 rows
    latest_10 = df.tail(10).sort_index(ascending=False)

    # Format for display
    display_df = latest_10.copy()
    display_df.index = display_df.index.strftime("%Y-%m-%d")
    display_df = display_df.round(4)

    st.dataframe(
        display_df,
        use_container_width=True,
        height=420,
    )

    # Dataset stats
    st.markdown("### Dataset Overview")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Rows", f"{len(df):,}")
    s2.metric("Features", f"{len(df.columns) - 1}")
    s3.metric("Date Range", f"{df.index.min().year}-{df.index.max().year}")
    s4.metric(
        "Target Balance",
        f"{df['Price_Jump_Target'].mean() * 100:.1f}% jumps",
    )

    # Download button
    st.markdown("### Download")
    csv_data = df.to_csv()
    st.download_button(
        label="📥 Download Full Dataset (CSV)",
        data=csv_data,
        file_name="featured_exchange_rates.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# TECHNICAL BACKGROUND (expander)
# ---------------------------------------------------------------------------
st.markdown("---")

with st.expander("📚 Technical Background — Methodology Deep Dive"):
    st.markdown("""
### Data Harmonization (Phase 1)

Ethiopia's exchange rate data comes from two World Food Programme (WFP) sources
with fundamentally different frequencies:

| Dataset | Frequency | Rows | Date Range |
|---|---|---|---|
| **Official Rate** | Daily | ~8,847 | 1992 - 2025 |
| **Parallel Rate** | Monthly | ~83 | 2017 - 2025 |

**The problem:** You can't train a daily prediction model with monthly labels.

**Our solution:**

- **Official Rate gaps (weekends/holidays):** Forward-fill. In a managed peg
  system, the rate *is* the previous day's rate until the National Bank of
  Ethiopia (NBE) announces a change. This is financially correct.

- **Parallel Rate (monthly to daily):** PCHIP (Piecewise Cubic Hermite
  Interpolating Polynomial) interpolation via `scipy.interpolate.pchip_interpolate`.
  Unlike linear interpolation, PCHIP produces a smooth, **monotonic** curve that
  respects the non-linear dynamics of informal FX markets. It won't create
  artificial overshoots between observation points.

---

### The Scale-Invariant Pivot (Phase 3)

The most critical technical decision in this project. The initial model used
raw rate levels as features (Official_Rate at 22-50 ETB during 2017-2023).
When tested on 2024 data where rates were 100-157 ETB, **every single test
sample fell outside the model's learned decision boundaries**, producing a
ROC-AUC of 0.478 (worse than random).

**The fix:** Replace all absolute rate features with percentage-based,
scale-invariant alternatives:

| Raw Feature (Failed) | Scale-Invariant Replacement |
|---|---|
| `Official_Rate` (22-50 ETB) | `Official_SMA7_Dev` (% deviation from 7-day SMA) |
| `Official_Vol_7d` (std dev in ETB) | `Official_CV_7d` (coefficient of variation) |
| `Parallel_Rate` | `Parallel_ROC_7d` (7-day % change) |
| Premium level | `Premium_Momentum` (7-day premium change) |

This single change improved ROC-AUC from **0.478 to 0.706**.

---

### Model Architecture

**Random Forest Classifier** with 500 trees, trained on 2017-2023 data:

```python
RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    min_samples_leaf=3,
    max_features='sqrt',
    class_weight='balanced',  # Each positive sample weighted ~57x
    random_state=42
)
```

**Why `class_weight='balanced'`?** With only 43 positive samples in 2,475
training rows (1.7%), a naive model achieves 98.3% accuracy by always
predicting "No Jump." The balanced weighting forces each tree to treat
rare jump events as 57x more important during construction.

**Why not SMOTE?** SMOTE creates synthetic minority samples by interpolating
between existing ones — producing temporally incoherent "days" that never
existed in the time-series. Class weighting achieves the same rebalancing
without generating artificial data.

**Threshold optimization:** The default 0.5 decision threshold fails because
the model was trained on 1.7% positives, so predicted probabilities are
naturally low. We optimize the threshold by maximizing F1-score on the
Precision-Recall curve, yielding an optimal threshold of 0.030.
    """)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #95a5a6; font-size: 0.85rem;">'
    "Ethiopia FX Early-Warning System &mdash; Built by "
    '<a href="https://github.com/rebiraolin" style="color: #3498db;">'
    "@rebiraolin</a> &mdash; "
    "Data: WFP VAM | Model: Random Forest (scikit-learn)"
    "</p>",
    unsafe_allow_html=True,
)
