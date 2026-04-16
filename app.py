# ============================================================
# Football Player Fitness & Injury Prediction System
# Enhanced Multi-Model Comparison Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             r2_score, mean_absolute_error,
                             mean_squared_error, confusion_matrix)

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Football Fitness Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS (preserved from base + minor additions) ─────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Barlow:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Barlow', sans-serif; }

.stApp {
    background: linear-gradient(160deg, #0a0e1a 0%, #0f1b2d 50%, #071024 100%);
    color: #e8eaf0;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1520 0%, #111d2e 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #c8d8e8 !important; }

.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 5.5rem;
    letter-spacing: 6px;
    background: linear-gradient(90deg, #00d4ff, #ffffff, #00d4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1.1;
}
.hero-sub {
    font-size: 0.95rem; color: #7ba3c8;
    letter-spacing: 2px; text-transform: uppercase; margin-top: 4px;
}
.hero-divider {
    height: 3px;
    background: linear-gradient(90deg, #00d4ff, transparent);
    margin: 16px 0 28px 0; border: none;
}

.section-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.4rem; letter-spacing: 3px;
    color: #00d4ff; text-transform: uppercase;
    border-left: 4px solid #00d4ff;
    padding-left: 10px; margin: 24px 0 14px 0;
}

.metric-grid { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 20px; }
.metric-card {
    flex: 1; min-width: 130px;
    background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(0,80,160,0.12));
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 12px; padding: 16px 18px; text-align: center;
}
.metric-card .label {
    font-size: 0.72rem; letter-spacing: 1.5px; text-transform: uppercase;
    color: #7ba3c8; margin-bottom: 6px;
}
.metric-card .value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem; color: #00d4ff; line-height: 1;
}
.metric-card .unit { font-size: 0.75rem; color: #5a8aaa; }

/* Best model highlight card */
.metric-card.best {
    border-color: #00e676;
    background: linear-gradient(135deg, rgba(0,230,118,0.1), rgba(0,80,40,0.12));
}
.metric-card.best .value { color: #00e676; }

.result-fit {
    background: linear-gradient(135deg, rgba(0,200,100,0.15), rgba(0,100,60,0.1));
    border: 2px solid #00c864; border-radius: 16px; padding: 22px 28px; text-align: center;
}
.result-notfit {
    background: linear-gradient(135deg, rgba(255,60,60,0.15), rgba(120,0,0,0.1));
    border: 2px solid #ff4444; border-radius: 16px; padding: 22px 28px; text-align: center;
}
.result-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4rem; letter-spacing: 4px;
}
.result-fit .result-title { color: #00e676; }
.result-notfit .result-title { color: #ff5252; }
.result-minutes { font-size: 1rem; color: #a0b8cc; margin-top: 6px; }
.result-minutes span { color: #00d4ff; font-weight: 700; font-size: 1.2rem; }

.dec-table { width: 100%; border-collapse: collapse; margin-top: 12px; }
.dec-table th {
    background: rgba(0,212,255,0.1); color: #00d4ff;
    font-size: 0.78rem; letter-spacing: 1.5px; text-transform: uppercase;
    padding: 10px 14px; text-align: left; border-bottom: 1px solid rgba(0,212,255,0.2);
}
.dec-table td {
    padding: 9px 14px; border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 0.88rem; color: #c8d8e8;
}
.dec-table tr:hover td { background: rgba(0,212,255,0.05); }

.info-box {
    background: rgba(0,212,255,0.06);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 10px; padding: 14px 18px;
    font-size: 0.88rem; color: #a0b8cc; margin-bottom: 18px;
}

/* Best model banner */
.best-model-banner {
    background: linear-gradient(135deg, rgba(0,230,118,0.12), rgba(0,100,60,0.08));
    border: 1px solid #00e676;
    border-radius: 12px; padding: 14px 20px; margin-bottom: 20px;
    font-size: 0.95rem; color: #00e676;
    font-family: 'Bebas Neue', sans-serif; letter-spacing: 2px; font-size: 1.1rem;
}

.stButton > button {
    background: linear-gradient(90deg, #0080c0, #00d4ff);
    color: #000 !important; font-weight: 700;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem; letter-spacing: 2px;
    border: none; border-radius: 8px;
    padding: 10px 30px; width: 100%;
    transition: transform 0.15s, box-shadow 0.15s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,212,255,0.35);
}
div[data-testid="stTextInput"] input {
    background: rgba(0,30,60,0.6) !important;
    border: 1px solid rgba(0,212,255,0.35) !important;
    color: #e8eaf0 !important; border-radius: 8px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1rem; letter-spacing: 2px; color: #7ba3c8 !important;
}
.stTabs [aria-selected="true"] { color: #00d4ff !important; }

.plot-container {
    background: rgba(10,20,40,0.7);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 12px; padding: 6px; overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════
DARK_BG  = "#0a0e1a"
CARD_BG  = "#0d1520"
CYAN     = "#00d4ff"
CYAN2    = "#0088bb"
GREEN    = "#00e676"
RED      = "#ff5252"
TEXT_CLR = "#c8d8e8"
GRID_CLR = "#1a2a3a"


# ══════════════════════════════════════════════════════════
# 1. DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════

@st.cache_data
def load_and_prepare():
    """Load CSV, fill missing values, engineer target labels."""
    # Try to load from current directory, then common paths
    import os
    possible_paths = [
        "ml_ready_fifa_dataset.csv",
        r"D:\ml_ready_fifa_dataset.csv",
        "data/ml_ready_fifa_dataset.csv",
    ]
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break

    if df is None:
        # Generate synthetic demo dataset so the app always runs
        st.warning("⚠️ Dataset not found — using synthetic demo data for display purposes.")
        np.random.seed(42)
        n = 2000
        positions = ["GK", "CB", "LB", "RB", "CM", "CAM", "LW", "RW", "ST", "CDM"]
        wr_options = ["Low/Low","Low/Medium","Medium/Low","Medium/Medium",
                      "Medium/High","High/Medium","High/High"]
        df = pd.DataFrame({
            "p_id2":   [f"player_{i:04d}" for i in range(n)],
            "position": np.random.choice(positions, n),
            "age":      np.random.randint(17, 38, n).astype(float),
            "pace":     np.random.normal(70, 12, n).clip(30, 99),
            "physic":   np.random.normal(68, 10, n).clip(30, 99),
            "fifa_rating": np.random.normal(72, 8, n).clip(50, 99),
            "cumulative_minutes_played": np.random.normal(5500, 2500, n).clip(0, 12000),
            "season_days_injured":       np.random.exponential(18, n).clip(0, 90),
            "avg_days_injured_prev_seasons": np.random.exponential(15, n).clip(0, 60),
            "work_rate": np.random.choice(wr_options, n),
            "bmi":       np.random.normal(22.5, 1.8, n).clip(18, 30),
            "nationality": np.random.choice(["Brazil","France","Germany","Spain","England"], n),
            "start_year": np.random.choice([2019,2020,2021,2022,2023], n),
            "minutes_per_game_prev_seasons": np.random.normal(70, 18, n).clip(0, 90),
            "season_minutes_played": np.random.normal(2200, 900, n).clip(0, 3800),
        })

    # Numeric columns to fill
    num_cols = ["age", "pace", "physic", "fifa_rating",
                "cumulative_minutes_played", "season_days_injured",
                "avg_days_injured_prev_seasons", "bmi",
                "minutes_per_game_prev_seasons", "season_minutes_played"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # Work-rate text → numeric
    wr_map = {"Low/Low": 1, "Low/Medium": 1.5, "Medium/Low": 2,
              "Medium/Medium": 2.5, "Medium/High": 3,
              "High/Medium": 3.5, "High/High": 4}
    if "work_rate_numeric" not in df.columns:
        df["work_rate_numeric"] = df["work_rate"].map(wr_map).fillna(2.5)
    else:
        df["work_rate_numeric"] = pd.to_numeric(df["work_rate_numeric"], errors="coerce").fillna(2.5)

    # ── Rule-based fitness label ──────────────────────────
    def fitness_label(row):
        c1 = row["season_days_injured"] < 30
        c2 = row["pace"] > 60
        c3 = row["physic"] > 60
        c4 = row["cumulative_minutes_played"] < 8000
        c5 = row["avg_days_injured_prev_seasons"] < 20
        c6 = row["work_rate_numeric"] >= 2
        return int(sum([c1, c2, c3, c4, c5, c6]) >= 4)

    df["fitness_label"] = df.apply(fitness_label, axis=1)

    # ── Recommended minutes target ─────────────────────────
    def recommended_minutes(row):
        base = 90
        base -= min(row["season_days_injured"] * 0.4, 40)
        base -= min(row["cumulative_minutes_played"] / 1000, 20)
        base += (row["physic"] - 60) * 0.3
        base += (row["pace"] - 60) * 0.2
        return float(np.clip(base, 0, 90))

    df["recommended_minutes"] = df.apply(recommended_minutes, axis=1)
    return df


# ══════════════════════════════════════════════════════════
# 2. MULTI-MODEL TRAINING
# ══════════════════════════════════════════════════════════

@st.cache_resource
def train_all_models(df):
    """
    Train four models: RandomForest, XGBoost, SVM, GradientBoosting.
    Returns metrics for each + the primary (best accuracy) clf/reg.
    """
    features = ["age", "pace", "physic", "fifa_rating",
                "cumulative_minutes_played", "season_days_injured",
                "avg_days_injured_prev_seasons",
                "work_rate_numeric", "bmi"]

    X = df[features].fillna(df[features].median())
    y_cls = df["fitness_label"]
    y_reg = df["recommended_minutes"]

    # 80/20 split (same seed for all models — fair comparison)
    X_tr, X_te, yc_tr, yc_te = train_test_split(X, y_cls, test_size=0.2, random_state=42)
    _,    _,    yr_tr, yr_te  = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    # SVM needs scaling
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    # ── Define model zoo ──────────────────────────────────
    model_configs = {
        "Random Forest": {
            "clf": RandomForestClassifier(n_estimators=100, random_state=42),
            "reg": RandomForestRegressor(n_estimators=100, random_state=42),
            "scaled": False,
        },
        "Gradient Boosting": {
            "clf": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "reg": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "scaled": False,
        },
        "SVM": {
            "clf": SVC(kernel="rbf", random_state=42, probability=True),
            "reg": SVR(kernel="rbf"),
            "scaled": True,  # uses scaled data
        },
    }

    # Conditionally add XGBoost
    if XGBOOST_AVAILABLE:
        model_configs["XGBoost"] = {
            "clf": XGBClassifier(n_estimators=100, random_state=42,
                                 eval_metric="logloss", verbosity=0),
            "reg": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
            "scaled": False,
        }

    # ── Train each model & collect metrics ───────────────
    all_cls_metrics = {}
    all_reg_metrics = {}
    trained_clfs    = {}
    trained_regs    = {}
    cms             = {}

    for name, cfg in model_configs.items():
        Xtr = X_tr_sc if cfg["scaled"] else X_tr
        Xte = X_te_sc if cfg["scaled"] else X_te

        # Classification
        clf = cfg["clf"]
        clf.fit(Xtr, yc_tr)
        yc_pred = clf.predict(Xte)
        all_cls_metrics[name] = {
            "Accuracy":  round(accuracy_score(yc_te, yc_pred), 3),
            "Precision": round(precision_score(yc_te, yc_pred, zero_division=0), 3),
            "Recall":    round(recall_score(yc_te, yc_pred, zero_division=0), 3),
            "F1 Score":  round(f1_score(yc_te, yc_pred, zero_division=0), 3),
        }
        cms[name] = confusion_matrix(yc_te, yc_pred)
        trained_clfs[name] = (clf, cfg["scaled"])

        # Regression
        reg = cfg["reg"]
        reg.fit(Xtr, yr_tr)
        yr_pred = reg.predict(Xte)
        all_reg_metrics[name] = {
            "R² Score": round(r2_score(yr_te, yr_pred), 3),
            "MAE":      round(mean_absolute_error(yr_te, yr_pred), 2),
            "RMSE":     round(np.sqrt(mean_squared_error(yr_te, yr_pred)), 2),
        }
        trained_regs[name] = (reg, cfg["scaled"])

    # ── Pick best model (highest accuracy) ───────────────
    best_model_name = max(all_cls_metrics,
                          key=lambda m: all_cls_metrics[m]["Accuracy"])

    # For the primary prediction, use Random Forest (always available + has feature importance)
    rf_clf = trained_clfs["Random Forest"][0]
    rf_reg = trained_regs["Random Forest"][0]
    rf_cm  = cms["Random Forest"]

    return (rf_clf, rf_reg,
            all_cls_metrics, all_reg_metrics,
            rf_cm, features,
            trained_clfs, trained_regs, cms,
            best_model_name, scaler)


# ══════════════════════════════════════════════════════════
# 3. PLOT HELPERS
# ══════════════════════════════════════════════════════════

def style_ax(ax, title=""):
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_CLR, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.title.set_color(CYAN)
    ax.title.set_fontsize(9)
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.xaxis.label.set_fontsize(7)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_fontsize(7)
    if title:
        ax.set_title(title.upper(), fontsize=9, fontweight="bold", color=CYAN, pad=8)


def make_collage(df):
    """4-panel collage: distribution | injury by position | correlation | feature importance"""
    fig = plt.figure(figsize=(13, 9), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                            left=0.07, right=0.97, top=0.93, bottom=0.07)

    # 1. Pace distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(CARD_BG)
    ax1.hist(df["pace"].dropna(), bins=28, color=CYAN, alpha=0.8, edgecolor=DARK_BG)
    ax1.axvline(df["pace"].median(), color="#ff6b6b", lw=1.5, linestyle="--", label="Median")
    ax1.legend(fontsize=7, facecolor=CARD_BG, labelcolor=TEXT_CLR)
    style_ax(ax1, "Pace Distribution")

    # 2. Injury days by position
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(CARD_BG)
    pos_inj = (df.groupby("position")["season_days_injured"]
                 .mean().sort_values(ascending=False).head(6))
    colors = [CYAN if i == 0 else CYAN2 for i in range(len(pos_inj))]
    bars = ax2.bar(pos_inj.index, pos_inj.values, color=colors, edgecolor=DARK_BG)
    for b in bars:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                 f"{b.get_height():.0f}", ha="center", va="bottom",
                 color=TEXT_CLR, fontsize=7)
    ax2.set_ylabel("Avg Injury Days")
    style_ax(ax2, "Avg Injury Days by Position")

    # 3. Correlation heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(CARD_BG)
    corr_cols = ["age", "pace", "physic", "fifa_rating",
                 "season_days_injured", "cumulative_minutes_played", "fitness_label"]
    corr = df[corr_cols].corr()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, ax=ax3, cmap=cmap, annot=True, fmt=".2f",
                linewidths=0.5, linecolor=DARK_BG,
                annot_kws={"size": 6}, cbar_kws={"shrink": 0.7})
    ax3.tick_params(axis="x", rotation=30, labelsize=6)
    ax3.tick_params(axis="y", rotation=0, labelsize=6)
    style_ax(ax3, "Correlation Heatmap")

    # 4. Feature importance placeholder (filled later)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(CARD_BG)
    style_ax(ax4, "Feature Importance")

    return fig, ax4


def fill_feature_importance(ax, clf, features):
    importances = clf.feature_importances_
    idx = np.argsort(importances)
    colors = [CYAN if i == len(idx)-1 else CYAN2 for i in range(len(idx))]
    ax.barh([features[i] for i in idx], importances[idx],
            color=colors, edgecolor=DARK_BG)
    ax.set_xlabel("Importance")
    style_ax(ax, "Feature Importance")


def make_confusion_matrix_fig(cm, title="CONFUSION MATRIX"):
    fig, ax = plt.subplots(figsize=(4, 3.2), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)
    labels = ["Not Fit", "Fit"]
    cmap = sns.light_palette(CYAN, as_cmap=True)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                xticklabels=labels, yticklabels=labels,
                linewidths=1, linecolor=DARK_BG,
                annot_kws={"size": 13, "weight": "bold", "color": "#000"})
    ax.set_xlabel("Predicted", color=TEXT_CLR, fontsize=8)
    ax.set_ylabel("Actual", color=TEXT_CLR, fontsize=8)
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    ax.set_title(title, color=CYAN, fontsize=9, fontweight="bold", pad=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    fig.tight_layout()
    return fig


def make_comparison_bar(metric_name, model_names, values, best_name):
    """Single horizontal bar chart for a metric comparison."""
    fig, ax = plt.subplots(figsize=(5.5, 2.8), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)
    bar_colors = [GREEN if m == best_name else CYAN for m in model_names]
    bars = ax.barh(model_names, values, color=bar_colors, edgecolor=DARK_BG, height=0.5)
    for b, v in zip(bars, values):
        ax.text(v + 0.005, b.get_y() + b.get_height()/2,
                f"{v:.3f}", va="center", color=TEXT_CLR, fontsize=8, fontweight="bold")
    ax.set_xlim(0, max(values) * 1.2 if max(values) > 0 else 1.0)
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.set_title(f"{metric_name.upper()} COMPARISON",
                 color=CYAN, fontsize=9, fontweight="bold", pad=8)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    fig.tight_layout(pad=1.2)
    return fig


# ══════════════════════════════════════════════════════════
# 4. PREDICTION HELPERS
# ══════════════════════════════════════════════════════════

def predict_player(player_row, clf, reg, features):
    """Run classifier + regressor on a single player row."""
    X = player_row[features].fillna(0).values.reshape(1, -1)
    fit_pred  = clf.predict(X)[0]
    mins_pred = reg.predict(X)[0]
    return int(fit_pred), round(float(mins_pred), 1)


def build_decision_table(row):
    """Return list of dicts for each fitness rule."""
    rules = [
        {"Condition": "Injury Days < 30",
         "Your Value": f"{row['season_days_injured']:.0f} days",
         "Result": "✅ Pass" if row["season_days_injured"] < 30 else "❌ Fail"},
        {"Condition": "Pace > 60",
         "Your Value": f"{row['pace']:.1f}",
         "Result": "✅ Pass" if row["pace"] > 60 else "❌ Fail"},
        {"Condition": "Physic > 60",
         "Your Value": f"{row['physic']:.1f}",
         "Result": "✅ Pass" if row["physic"] > 60 else "❌ Fail"},
        {"Condition": "Workload < 8000 mins",
         "Your Value": f"{row['cumulative_minutes_played']:.0f} mins",
         "Result": "✅ Pass" if row["cumulative_minutes_played"] < 8000 else "❌ Fail"},
        {"Condition": "Avg Injury History < 20 days",
         "Your Value": f"{row['avg_days_injured_prev_seasons']:.1f} days",
         "Result": "✅ Pass" if row["avg_days_injured_prev_seasons"] < 20 else "❌ Fail"},
        {"Condition": "Work Rate ≥ Moderate",
         "Your Value": row.get("work_rate", f"{row['work_rate_numeric']}"),
         "Result": "✅ Pass" if row["work_rate_numeric"] >= 2 else "❌ Fail"},
    ]
    return rules


# ══════════════════════════════════════════════════════════
# 5. STREAMLIT UI — MAIN
# ══════════════════════════════════════════════════════════

def main():
    # ── Load data & train ─────────────────────────────────
    with st.spinner("Loading dataset and training models…"):
        df = load_and_prepare()
        (clf, reg,
         all_cls_metrics, all_reg_metrics,
         cm, features,
         trained_clfs, trained_regs, cms,
         best_model_name, scaler) = train_all_models(df)

    # Shorthand for primary model metrics (Random Forest)
    cls_metrics = all_cls_metrics["Random Forest"]
    reg_metrics = all_reg_metrics["Random Forest"]

    # ── Hero header ───────────────────────────────────────
    st.markdown("""
    <div style="padding: 10px 0 0 0">
        <p class="hero-title">⚽ Football Fitness Predictor</p>
        <p class="hero-sub">AI-powered Player Fitness &amp; Injury Risk System · Multi-Model Comparison</p>
    </div>
    <hr class="hero-divider">
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🏟️ Player Lookup")
        st.markdown('<div class="info-box">Enter a Player ID to get their fitness prediction and breakdown.</div>',
                    unsafe_allow_html=True)

        player_id   = st.text_input("Player ID (p_id2)", placeholder="e.g. player_0001")
        predict_btn = st.button("⚡ Predict Fitness")

        st.markdown("---")
        st.markdown("### 📋 Dataset Info")
        st.markdown(
            f"<div class='info-box'>Total records: <b>{len(df):,}</b><br>"
            f"Unique players: <b>{df['p_id2'].nunique():,}</b><br>"
            f"Fit players: <b>{df['fitness_label'].sum():,}</b> "
            f"({df['fitness_label'].mean()*100:.1f}%)</div>",
            unsafe_allow_html=True)

        st.markdown("### 🏆 Best Model")
        st.markdown(
            f"<div class='info-box' style='border-color:#00e676; color:#00e676'>"
            f"<b>{best_model_name}</b><br>"
            f"Accuracy: <b>{all_cls_metrics[best_model_name]['Accuracy']:.3f}</b></div>",
            unsafe_allow_html=True)

        st.markdown("### 🔬 Features Used")
        for f in features:
            st.markdown(f"<div style='font-size:0.82rem; color:#7ba3c8; padding:2px 0'>▸ {f}</div>",
                        unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊  Dashboard",
        "🔮  Prediction",
        "📈  Model Metrics",
        "📊  Model Comparison"
    ])

    # ════════════════════════════════
    # TAB 1 — DASHBOARD
    # ════════════════════════════════
    with tab1:
        st.markdown('<div class="section-label">Dataset Overview</div>', unsafe_allow_html=True)

        total    = len(df)
        fit_pct  = df["fitness_label"].mean() * 100
        avg_inj  = df["season_days_injured"].mean()
        avg_pace = df["pace"].mean()

        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="label">Total Records</div>
                <div class="value">{total:,}</div>
            </div>
            <div class="metric-card">
                <div class="label">Fit Players</div>
                <div class="value">{fit_pct:.1f}<span style="font-size:1rem">%</span></div>
            </div>
            <div class="metric-card">
                <div class="label">Avg Injury Days</div>
                <div class="value">{avg_inj:.0f}</div>
                <div class="unit">days/season</div>
            </div>
            <div class="metric-card">
                <div class="label">Avg Pace</div>
                <div class="value">{avg_pace:.1f}</div>
                <div class="unit">/ 100</div>
            </div>
            <div class="metric-card">
                <div class="label">Models Trained</div>
                <div class="value">{len(all_cls_metrics)}</div>
                <div class="unit">ML models</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Statistical insights
        st.markdown('<div class="section-label">Statistical Insights</div>', unsafe_allow_html=True)
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.markdown(f"""
            <div class="info-box">
                <b>Age Range:</b> {df['age'].min():.0f} – {df['age'].max():.0f} yrs<br>
                <b>Mean Age:</b> {df['age'].mean():.1f} yrs<br>
                <b>Peak Pace (95th pct):</b> {df['pace'].quantile(0.95):.1f}
            </div>""", unsafe_allow_html=True)
        with col_s2:
            st.markdown(f"""
            <div class="info-box">
                <b>Mean Workload:</b> {df['cumulative_minutes_played'].mean():.0f} mins<br>
                <b>Mean BMI:</b> {df['bmi'].mean():.2f}<br>
                <b>Mean FIFA Rating:</b> {df['fifa_rating'].mean():.1f}
            </div>""", unsafe_allow_html=True)
        with col_s3:
            corr_val = df[["pace","fitness_label"]].corr().iloc[0,1]
            st.markdown(f"""
            <div class="info-box">
                <b>Pace–Fitness Correlation:</b> {corr_val:.3f}<br>
                <b>Not Fit Players:</b> {(df['fitness_label']==0).sum():,}<br>
                <b>Fit Players:</b> {df['fitness_label'].sum():,}
            </div>""", unsafe_allow_html=True)

        # Collage
        st.markdown('<div class="section-label">Visualisations</div>', unsafe_allow_html=True)
        fig, ax4 = make_collage(df)
        fill_feature_importance(ax4, clf, features)
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        plt.close(fig)

    # ════════════════════════════════
    # TAB 2 — PREDICTION
    # ════════════════════════════════
    with tab2:
        st.markdown('<div class="section-label">Player Fitness Prediction</div>',
                    unsafe_allow_html=True)

        if predict_btn:
            if not player_id.strip():
                st.warning("⚠️ Please enter a Player ID in the sidebar.")
            else:
                pid = player_id.strip().lower()
                player_rows = df[df["p_id2"].str.lower() == pid]

                if player_rows.empty:
                    st.error(f"❌ Player **{player_id}** not found in the dataset.")
                    st.markdown("**Sample IDs you can try:**")
                    sample = df["p_id2"].dropna().unique()[:8]
                    st.code("  |  ".join(sample))
                else:
                    row = player_rows.sort_values("start_year", ascending=False).iloc[0]
                    fit_pred, mins_pred = predict_player(row, clf, reg, features)
                    decision = build_decision_table(row)
                    passes = sum(1 for d in decision if "✅" in d["Result"])

                    result_class = "result-fit" if fit_pred == 1 else "result-notfit"
                    result_text  = "✅ FIT TO PLAY" if fit_pred == 1 else "❌ NOT FIT"
                    st.markdown(f"""
                    <div class="{result_class}">
                        <div class="result-title">{result_text}</div>
                        <div class="result-minutes">
                            Recommended Playing Time: <span>{mins_pred} mins</span>
                            &nbsp;|&nbsp; Conditions Passed: <span>{passes}/6</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

                    col_a, col_b = st.columns([1, 1.6])
                    with col_a:
                        st.markdown('<div class="section-label" style="font-size:1rem">Player Profile</div>',
                                    unsafe_allow_html=True)
                        stats = {
                            "Name / ID":         row["p_id2"],
                            "Position":          row.get("position", "—"),
                            "Age":               f"{row['age']:.0f} yrs",
                            "Nationality":       row.get("nationality", "—"),
                            "FIFA Rating":       f"{row['fifa_rating']:.0f}",
                            "Pace":              f"{row['pace']:.1f}",
                            "Physic":            f"{row['physic']:.1f}",
                            "Injury Days":       f"{row['season_days_injured']:.0f} days",
                            "Workload (career)": f"{row['cumulative_minutes_played']:.0f} mins",
                            "Work Rate":         row.get("work_rate", "—"),
                            "BMI":               f"{row['bmi']:.2f}",
                        }
                        for k, v in stats.items():
                            st.markdown(
                                f"<div style='display:flex; justify-content:space-between; "
                                f"padding:5px 0; border-bottom:1px solid #1a2a3a; font-size:0.86rem'>"
                                f"<span style='color:#7ba3c8'>{k}</span>"
                                f"<span style='color:#e8eaf0; font-weight:600'>{v}</span></div>",
                                unsafe_allow_html=True)

                    with col_b:
                        st.markdown('<div class="section-label" style="font-size:1rem">Decision Breakdown</div>',
                                    unsafe_allow_html=True)
                        rows_html = "".join(
                            f"<tr><td>{d['Condition']}</td>"
                            f"<td>{d['Your Value']}</td>"
                            f"<td>{d['Result']}</td></tr>"
                            for d in decision
                        )
                        st.markdown(f"""
                        <table class="dec-table">
                            <thead><tr>
                                <th>Condition</th><th>Value</th><th>Status</th>
                            </tr></thead>
                            <tbody>{rows_html}</tbody>
                        </table>
                        """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="info-box" style="text-align:center; padding:30px; font-size:0.95rem">
                👈 Enter a <b>Player ID</b> in the sidebar and click <b>⚡ Predict Fitness</b>
                to see the full fitness analysis.
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-label">Sample Players</div>', unsafe_allow_html=True)
            sample_df = (df.groupby("p_id2")
                           .agg(position=("position","first"),
                                age=("age","max"),
                                pace=("pace","mean"),
                                physic=("physic","mean"),
                                fitness_label=("fitness_label","last"))
                           .reset_index()
                           .head(12)
                           [["p_id2","position","age","pace","physic","fitness_label"]])
            sample_df.columns = ["Player ID","Position","Age","Pace","Physic","Fit?"]
            sample_df["Fit?"] = sample_df["Fit?"].map({1:"✅ Fit", 0:"❌ Not Fit"})
            st.dataframe(sample_df, use_container_width=True, hide_index=True,
                         column_config={
                             "Pace":  st.column_config.ProgressColumn(min_value=0, max_value=100),
                             "Physic": st.column_config.ProgressColumn(min_value=0, max_value=100)
                         })

    # ════════════════════════════════
    # TAB 3 — MODEL METRICS (All Models)
    # ════════════════════════════════
    with tab3:
        st.markdown('<div class="section-label">Per-Model Metrics</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            Classification and regression metrics for every trained model.
            Best model is highlighted with a <span style="color:#00e676">green border</span>.
        </div>
        """, unsafe_allow_html=True)

        # ── Loop through every model and render its metric cards + confusion matrix ──
        for mname in all_cls_metrics.keys():
            is_best = (mname == best_model_name)
            badge   = " 🏆" if is_best else ""
            border  = "border-left: 4px solid #00e676;" if is_best else "border-left: 4px solid #00d4ff;"

            # Model name header
            st.markdown(
                f'<div class="section-label" style="{border}">{mname}{badge}</div>',
                unsafe_allow_html=True)

            # ── Classification metric cards ──────────────
            st.markdown(
                f'<div style="font-size:0.78rem; letter-spacing:1.5px; color:#7ba3c8; '
                f'text-transform:uppercase; margin-bottom:8px;">Classification Metrics</div>',
                unsafe_allow_html=True)

            cls_m = all_cls_metrics[mname]
            c_cols = st.columns(4)
            cls_card_color = "metric-card best" if is_best else "metric-card"
            for col, (metric_name, metric_val) in zip(c_cols, cls_m.items()):
                with col:
                    st.markdown(f"""
                    <div class="{cls_card_color}">
                        <div class="label">{metric_name}</div>
                        <div class="value">{metric_val}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

            # ── Regression metric cards ───────────────────
            st.markdown(
                f'<div style="font-size:0.78rem; letter-spacing:1.5px; color:#7ba3c8; '
                f'text-transform:uppercase; margin-bottom:8px;">Regression Metrics</div>',
                unsafe_allow_html=True)

            reg_m = all_reg_metrics[mname]
            r_cols = st.columns(3)
            for col, (metric_name, metric_val) in zip(r_cols, reg_m.items()):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="label">{metric_name}</div>
                        <div class="value">{metric_val}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

            # ── Confusion matrix (centred, not too wide) ──
            st.markdown(
                f'<div style="font-size:0.78rem; letter-spacing:1.5px; color:#7ba3c8; '
                f'text-transform:uppercase; margin-bottom:8px;">Confusion Matrix</div>',
                unsafe_allow_html=True)

            cm_left, cm_mid, cm_right = st.columns([1, 1.4, 1])
            with cm_mid:
                cm_fig = make_confusion_matrix_fig(cms[mname], title=mname.upper())
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.pyplot(cm_fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                plt.close(cm_fig)

            # Divider between models
            st.markdown("<hr style='border:none; border-top:1px solid #1a2a3a; margin:24px 0'>",
                        unsafe_allow_html=True)

        # ── Shared model details note ─────────────────────
        st.markdown('<div class="section-label">Model Details</div>', unsafe_allow_html=True)
        xgb_note = "XGBoost included (XGBClassifier + XGBRegressor)." if XGBOOST_AVAILABLE \
                   else "XGBoost not installed — install via <code>pip install xgboost</code>."
        st.markdown(f"""
        <div class="info-box">
            <b>Random Forest:</b> 100 trees — no scaling needed. Provides feature importance.<br>
            <b>Gradient Boosting:</b> 100 estimators — sequential boosting on residuals.<br>
            <b>SVM:</b> RBF kernel — StandardScaler applied before training/prediction.<br>
            <b>XGBoost:</b> {xgb_note}<br><br>
            <b>Target (cls):</b> Fit / Not Fit (rule-based label: ≥4/6 conditions met)<br>
            <b>Target (reg):</b> Recommended playing minutes (0–90)<br>
            <b>Split:</b> 80/20 train-test, random_state=42 — same for all models.<br>
            <b>Missing values:</b> Imputed with column median.
        </div>
        """, unsafe_allow_html=True)

    # ════════════════════════════════
    # TAB 4 — MODEL COMPARISON (NEW)
    # ════════════════════════════════
    with tab4:
        st.markdown('<div class="section-label">Multi-Model Comparison</div>',
                    unsafe_allow_html=True)

        # Best model banner
        best_acc = all_cls_metrics[best_model_name]["Accuracy"]
        st.markdown(
            f'<div class="best-model-banner">🏆 Best Model: {best_model_name} '
            f'— Accuracy {best_acc:.3f}</div>',
            unsafe_allow_html=True)

        model_names = list(all_cls_metrics.keys())

        # ── Classification comparison table ──────────────
        st.markdown('<div class="section-label">Classification Metrics — All Models</div>',
                    unsafe_allow_html=True)

        cls_rows = []
        for mname in model_names:
            m = all_cls_metrics[mname]
            is_best = "🏆 " if mname == best_model_name else ""
            cls_rows.append({
                "Model":     is_best + mname,
                "Accuracy":  m["Accuracy"],
                "Precision": m["Precision"],
                "Recall":    m["Recall"],
                "F1 Score":  m["F1 Score"],
            })
        cls_df = pd.DataFrame(cls_rows)
        st.dataframe(cls_df, use_container_width=True, hide_index=True,
                     column_config={
                         "Accuracy":  st.column_config.ProgressColumn(min_value=0, max_value=1),
                         "F1 Score":  st.column_config.ProgressColumn(min_value=0, max_value=1),
                         "Precision": st.column_config.ProgressColumn(min_value=0, max_value=1),
                         "Recall":    st.column_config.ProgressColumn(min_value=0, max_value=1),
                     })

        # ── Regression comparison table ───────────────────
        st.markdown('<div class="section-label">Regression Metrics — All Models</div>',
                    unsafe_allow_html=True)

        reg_rows = []
        for mname in model_names:
            m = all_reg_metrics[mname]
            reg_rows.append({
                "Model":    mname,
                "R² Score": m["R² Score"],
                "MAE":      m["MAE"],
                "RMSE":     m["RMSE"],
            })
        reg_df = pd.DataFrame(reg_rows)
        st.dataframe(reg_df, use_container_width=True, hide_index=True)

        # ── Bar charts ────────────────────────────────────
        st.markdown('<div class="section-label">Visual Comparison</div>', unsafe_allow_html=True)

        col_b1, col_b2 = st.columns(2)

        with col_b1:
            acc_vals = [all_cls_metrics[m]["Accuracy"] for m in model_names]
            fig_acc = make_comparison_bar("Accuracy", model_names, acc_vals, best_model_name)
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.pyplot(fig_acc, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            plt.close(fig_acc)

        with col_b2:
            f1_vals = [all_cls_metrics[m]["F1 Score"] for m in model_names]
            fig_f1 = make_comparison_bar("F1 Score", model_names, f1_vals, best_model_name)
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.pyplot(fig_f1, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            plt.close(fig_f1)

        # ── Precision & Recall ────────────────────────────
        col_b3, col_b4 = st.columns(2)

        with col_b3:
            prec_vals = [all_cls_metrics[m]["Precision"] for m in model_names]
            fig_prec = make_comparison_bar("Precision", model_names, prec_vals, best_model_name)
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.pyplot(fig_prec, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            plt.close(fig_prec)

        with col_b4:
            rec_vals = [all_cls_metrics[m]["Recall"] for m in model_names]
            fig_rec = make_comparison_bar("Recall", model_names, rec_vals, best_model_name)
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.pyplot(fig_rec, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            plt.close(fig_rec)

        # ── Confusion matrices for all models ─────────────
        st.markdown('<div class="section-label">Confusion Matrices — All Models</div>',
                    unsafe_allow_html=True)

        cm_cols = st.columns(len(model_names))
        for i, mname in enumerate(model_names):
            with cm_cols[i]:
                cm_fig = make_confusion_matrix_fig(cms[mname], title=mname.upper())
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.pyplot(cm_fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                plt.close(cm_fig)

        # ── Summary notes ─────────────────────────────────
        st.markdown('<div class="section-label">Model Notes</div>', unsafe_allow_html=True)
        xgb_note = "XGBoost included." if XGBOOST_AVAILABLE else "XGBoost not installed (pip install xgboost)."
        st.markdown(f"""
        <div class="info-box">
            <b>Random Forest:</b> 100 trees, no scaling needed. Provides feature importance.<br>
            <b>Gradient Boosting:</b> Sequential boosting; strong on tabular data.<br>
            <b>SVM:</b> Radial Basis Function kernel; StandardScaler applied before training.<br>
            <b>XGBoost:</b> {xgb_note}<br><br>
            All models use the same 80/20 train-test split (random_state=42) for a fair comparison.<br>
            🏆 Best model is highlighted in <span style="color:#00e676">green</span>.
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()