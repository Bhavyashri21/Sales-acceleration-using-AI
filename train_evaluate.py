"""
train_evaluate.py
Trains and evaluates the Hawkes model alongside baseline models.
Reproduces the comparison from Table 2 of the paper:
  - Subjective (random) baseline
  - Logistic Regression
  - Hawkes Win Predictor (our model)
Prints AUC scores and saves plots.
"""

import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

from data_generator import generate_dataset
from hawkes_model import HawkesWinPredictor


# ── 1. Load or Generate Data ──────────────────────────────────────────────────

def load_data():
    try:
        df = pd.read_csv("leads_data.csv")
        df["interaction_times"] = df["interaction_times"].apply(ast.literal_eval)
        print(f"[Data] Loaded leads_data.csv ({len(df)} leads)")
    except FileNotFoundError:
        df = generate_dataset(600)
        df.to_csv("leads_data.csv", index=False)
        print(f"[Data] Generated and saved leads_data.csv ({len(df)} leads)")
    return df


# ── 2. Feature Engineering ────────────────────────────────────────────────────

def encode_features(df: pd.DataFrame):
    """Encode categorical profile features into numeric vectors."""
    cat_cols = ["geography", "deal_size", "sector", "product_line"]
    encoders = {}
    df_enc = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col + "_enc"] = le.fit_transform(df[col])
        encoders[col] = le

    feature_cols = (
        [c + "_enc" for c in cat_cols]
        + ["lead_age_days", "sales_stage", "num_interactions",
           "clustering_score", "mean_gap_days", "recency_days"]
    )
    X = df_enc[feature_cols].values.astype(float)
    y = df["win"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, feature_cols


# ── 3. Train & Evaluate ───────────────────────────────────────────────────────

def evaluate_models(df: pd.DataFrame, X: np.ndarray, y: np.ndarray):
    # Train/test split (70/30)
    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(idx, test_size=0.30, random_state=42, stratify=y)

    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    train_seqs = [df.iloc[i]["interaction_times"] for i in idx_train]
    test_seqs  = [df.iloc[i]["interaction_times"] for i in idx_test]

    results = {}

    # ── Baseline: "Subjective" random scores (simulating biased human ratings) ──
    np.random.seed(0)
    subj_scores = np.random.beta(2, 5, size=len(y_test))  # skewed low, like biased ratings
    results["Subjective (Baseline)"] = roc_auc_score(y_test, subj_scores)

    # ── Model A: Logistic Regression ──
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_train, y_train)
    lr_scores = lr.predict_proba(X_test)[:, 1]
    results["Logistic Regression"] = roc_auc_score(y_test, lr_scores)

    # ── Model B: Hawkes Win Predictor ──
    hawkes = HawkesWinPredictor(max_iter=30, lr=0.005)
    hawkes.fit(train_seqs, X_train, y_train.tolist(), T=90.0)
    hawkes_scores = hawkes.predict_proba(test_seqs, X_test, T=90.0)
    results["Hawkes Model (Ours)"] = roc_auc_score(y_test, hawkes_scores)

    # ── Print results ──
    print("\n" + "=" * 50)
    print("  MODEL COMPARISON (AUC)")
    print("=" * 50)
    for model, auc in results.items():
        marker = " ◀ BEST" if auc == max(results.values()) else ""
        print(f"  {model:<30s}  AUC = {auc:.4f}{marker}")
    print("=" * 50)

    # Classification report for Hawkes model
    hawkes_preds = (hawkes_scores > 0.5).astype(int)
    print("\n[Hawkes Model] Classification Report:")
    print(classification_report(y_test, hawkes_preds, target_names=["Lost", "Won"]))

    return results, y_test, {
        "Subjective (Baseline)": subj_scores,
        "Logistic Regression":   lr_scores,
        "Hawkes Model (Ours)":   hawkes_scores,
    }


# ── 4. Plots ──────────────────────────────────────────────────────────────────

def plot_results(results: dict, y_test: np.ndarray, score_dict: dict, df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Sales Pipeline AI – Results Dashboard", fontsize=15, fontweight="bold")

    # ── Plot 1: AUC Bar Chart ──
    ax = axes[0]
    models = list(results.keys())
    aucs   = list(results.values())
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    bars   = ax.bar(models, aucs, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUC Score", fontsize=12)
    ax.set_title("Model AUC Comparison", fontsize=12)
    ax.set_xticklabels(models, rotation=12, ha="right", fontsize=9)
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{auc:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

    # ── Plot 2: ROC Curves ──
    ax = axes[1]
    line_styles = ["--", "-.", "-"]
    for (model, scores), ls, color in zip(score_dict.items(), line_styles, colors):
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc = results[model]
        ax.plot(fpr, tpr, lw=2, ls=ls, color=color, label=f"{model} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Plot 3: Win propensity score distribution ──
    ax = axes[2]
    hawkes_scores = score_dict["Hawkes Model (Ours)"]
    won_scores  = hawkes_scores[y_test == 1]
    lost_scores = hawkes_scores[y_test == 0]
    ax.hist(lost_scores, bins=25, alpha=0.6, color="#e74c3c", label="Lost Leads", edgecolor="black")
    ax.hist(won_scores,  bins=25, alpha=0.7, color="#2ecc71", label="Won Leads",  edgecolor="black")
    ax.axvline(0.5, color="black", ls="--", lw=1.5, label="Threshold 0.5")
    ax.set_xlabel("Predicted Win Propensity Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution (Hawkes Model)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results_dashboard.png", dpi=150, bbox_inches="tight")
    print("\n[Plot] Saved → results_dashboard.png")
    plt.show()

    # ── Extra Plot: Interaction Clustering ──
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4))
    fig2.suptitle("Hawkes Process Insights", fontsize=13, fontweight="bold")

    ax = axes2[0]
    won_mask  = df["win"] == 1
    lost_mask = df["win"] == 0
    ax.scatter(df.loc[lost_mask, "recency_days"], df.loc[lost_mask, "num_interactions"],
               alpha=0.4, color="#e74c3c", label="Lost", s=20)
    ax.scatter(df.loc[won_mask,  "recency_days"], df.loc[won_mask,  "num_interactions"],
               alpha=0.6, color="#2ecc71", label="Won",  s=30, marker="^")
    ax.set_xlabel("Recency (days since last interaction)")
    ax.set_ylabel("Number of Interactions")
    ax.set_title("Interaction Recency vs Frequency")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes2[1]
    ax.hist(df.loc[lost_mask, "clustering_score"], bins=30, alpha=0.6,
            color="#e74c3c", label="Lost", edgecolor="black")
    ax.hist(df.loc[won_mask, "clustering_score"], bins=30, alpha=0.7,
            color="#2ecc71", label="Won",  edgecolor="black")
    ax.set_xlabel("Clustering Score (lower = more clustered)")
    ax.set_ylabel("Count")
    ax.set_title("Temporal Clustering of Seller Interactions")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("hawkes_insights.png", dpi=150, bbox_inches="tight")
    print("[Plot] Saved → hawkes_insights.png")
    plt.show()


# ── 5. Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Sales Acceleration using AI – Pipeline Win Predictor")
    print("  Based on: Yan et al. (AAAI 2015)")
    print("=" * 60)

    df = load_data()
    X, y, feature_cols = encode_features(df)
    print(f"[Features] {len(feature_cols)} features: {feature_cols}")

    results, y_test, score_dict = evaluate_models(df, X, y)
    plot_results(results, y_test, score_dict, df)

    print("\n Done! Check results_dashboard.png and hawkes_insights.png")