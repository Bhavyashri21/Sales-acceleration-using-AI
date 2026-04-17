"""
dashboard.py
Interactive CLI dashboard – scores new leads in real time.
Run after train_evaluate.py has been executed.
"""

import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from data_generator import generate_dataset, GEOGRAPHY, DEAL_SIZE, SECTOR, PRODUCT_LINE
from hawkes_model import HawkesWinPredictor


def prepare_and_train():
    """Load data, encode features, train Hawkes model, return model + scaler."""
    try:
        df = pd.read_csv("leads_data.csv")
        df["interaction_times"] = df["interaction_times"].apply(ast.literal_eval)
    except FileNotFoundError:
        df = generate_dataset(500)
        df.to_csv("leads_data.csv", index=False)

    cat_cols = ["geography", "deal_size", "sector", "product_line"]
    encoders = {}
    df_enc = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(df[col])
        df_enc[col + "_enc"] = le.transform(df[col])
        encoders[col] = le

    feature_cols = (
        [c + "_enc" for c in cat_cols]
        + ["lead_age_days", "sales_stage", "num_interactions",
           "clustering_score", "mean_gap_days", "recency_days"]
    )
    X = df_enc[feature_cols].values.astype(float)
    y = df["win"].values
    seqs = df["interaction_times"].tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = HawkesWinPredictor(max_iter=25, lr=0.005)
    model.fit(seqs, X_scaled, y.tolist())

    return model, scaler, encoders, feature_cols


def get_user_lead():
    """Prompt user to enter a new lead interactively."""
    print("\n" + "─" * 50)
    print("  ENTER NEW LEAD DETAILS")
    print("─" * 50)

    geo_options = list(enumerate(GEOGRAPHY, 1))
    print("Geography: " + " | ".join(f"[{i}] {g}" for i, g in geo_options))
    geo_idx = int(input("  Choose (1-4): ").strip()) - 1
    geo = GEOGRAPHY[geo_idx]

    deal_options = list(enumerate(DEAL_SIZE, 1))
    print("Deal Size: " + " | ".join(f"[{i}] {d}" for i, d in deal_options))
    deal_idx = int(input("  Choose (1-4): ").strip()) - 1
    deal = DEAL_SIZE[deal_idx]

    sec_options = list(enumerate(SECTOR, 1))
    print("Sector:    " + " | ".join(f"[{i}] {s}" for i, s in sec_options))
    sec_idx = int(input("  Choose (1-4): ").strip()) - 1
    sector = SECTOR[sec_idx]

    prod_options = list(enumerate(PRODUCT_LINE, 1))
    print("Product:   " + " | ".join(f"[{i}] {p}" for i, p in prod_options))
    prod_idx = int(input("  Choose (1-4): ").strip()) - 1
    product = PRODUCT_LINE[prod_idx]

    lead_age   = int(input("Lead age (days, e.g. 30): ").strip())
    sales_stage = int(input("Sales stage (1-4):        ").strip())

    print("\nEnter interaction timestamps (days from lead creation, e.g.: 10 25 26 27 80)")
    raw = input("  Times: ").strip()
    times = sorted([float(x) for x in raw.split()])

    return {
        "geography":    geo,
        "deal_size":    deal,
        "sector":       sector,
        "product_line": product,
        "lead_age_days":  lead_age,
        "sales_stage":    sales_stage,
        "interaction_times": times,
    }


def featurize_lead(lead: dict, encoders: dict, scaler: StandardScaler) -> tuple:
    """Convert a raw lead dict into a scaled feature vector."""
    cat_enc = [
        encoders["geography"].transform([lead["geography"]])[0],
        encoders["deal_size"].transform([lead["deal_size"]])[0],
        encoders["sector"].transform([lead["sector"]])[0],
        encoders["product_line"].transform([lead["product_line"]])[0],
    ]

    times = lead["interaction_times"]
    n = len(times)
    if n > 1:
        diffs = np.diff(times)
        clust = float(np.std(diffs))
        mean_gap = float(np.mean(diffs))
        recency = float(90.0 - times[-1])
    else:
        clust, mean_gap, recency = 90.0, 90.0, 90.0

    num_feats = [lead["lead_age_days"], lead["sales_stage"], n, clust, mean_gap, recency]
    raw = np.array(cat_enc + num_feats, dtype=float).reshape(1, -1)
    scaled = scaler.transform(raw)
    return scaled, times


def score_bar(prob: float, width: int = 40) -> str:
    filled = int(prob * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {prob:.1%}"


def recommendation(score: float) -> str:
    if score >= 0.70:
        return " HIGH PRIORITY – Assign senior rep, schedule call this week"
    elif score >= 0.45:
        return " MEDIUM PRIORITY – Follow up within 2 weeks"
    elif score >= 0.25:
        return " LOW PRIORITY – Monitor; increase interaction frequency"
    else:
        return "  COLD LEAD – Re-evaluate qualification or park"


def run_dashboard():
    print("\n" + "=" * 60)
    print("    Sales Pipeline AI – Real-Time Lead Scorer")
    print("  Based on Hawkes Process Win-Propensity Model")
    print("=" * 60)

    print("\n[System] Training model on existing pipeline data...")
    model, scaler, encoders, _ = prepare_and_train()
    print("[System] Model ready!\n")

    while True:
        print("\n" + "=" * 60)
        choice = input("Options: [1] Score a new lead  [2] Score random leads  [3] Quit\n→ ").strip()

        if choice == "3":
            print("\nGoodbye! ")
            break

        elif choice == "2":
            # Score a batch of 5 random leads and rank them
            print("\n[Generating 5 random leads and scoring...]\n")
            from data_generator import generate_dataset
            sample_df = generate_dataset(5)
            sample_df["interaction_times"] = sample_df["interaction_times"]

            rows = []
            for _, row in sample_df.iterrows():
                lead_dict = {
                    "geography": row["geography"], "deal_size": row["deal_size"],
                    "sector": row["sector"], "product_line": row["product_line"],
                    "lead_age_days": row["lead_age_days"], "sales_stage": row["sales_stage"],
                    "interaction_times": row["interaction_times"],
                }
                X_lead, times = featurize_lead(lead_dict, encoders, scaler)
                score = model.predict_proba([times], X_lead)[0]
                rows.append((row["lead_id"], lead_dict["geography"],
                             lead_dict["deal_size"], len(times), score))

            rows.sort(key=lambda r: r[4], reverse=True)
            print(f"{'Rank':<5} {'Lead ID':<12} {'Geo':<15} {'Deal':<12} {'Interactions':<14} {'Win Propensity'}")
            print("─" * 80)
            for rank, (lid, geo, deal, n_int, score) in enumerate(rows, 1):
                print(f"  {rank:<4} {lid:<12} {geo:<15} {deal:<12} {n_int:<14} {score_bar(score, 20)}")

        elif choice == "1":
            try:
                lead = get_user_lead()
                X_lead, times = featurize_lead(lead, encoders, scaler)
                score = model.predict_proba([times], X_lead)[0]

                print("\n" + "─" * 50)
                print("   LEAD SCORING RESULT")
                print("─" * 50)
                print(f"  Geography    : {lead['geography']}")
                print(f"  Deal Size    : {lead['deal_size']}")
                print(f"  Sector       : {lead['sector']}")
                print(f"  Product      : {lead['product_line']}")
                print(f"  Interactions : {len(times)} events → {times}")
                print(f"\n  Win Propensity (next 2 weeks):")
                print(f"  {score_bar(score)}")
                print(f"\n  → {recommendation(score)}")
                print("─" * 50)

            except (ValueError, IndexError) as e:
                print(f"[Error] Invalid input: {e}. Please try again.")

        else:
            print("Invalid option, please enter 1, 2, or 3.")


if __name__ == "__main__":
    run_dashboard()