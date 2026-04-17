"""
data_generator.py
Generates synthetic sales pipeline data inspired by the paper.
Each lead has: profile features + seller interaction timestamps + win/loss outcome
"""

import numpy as np
import pandas as pd
import random

np.random.seed(42)
random.seed(42)

GEOGRAPHY    = ["Greater China", "Southeast Asia", "South Asia", "Europe"]
DEAL_SIZE    = ["Small", "Medium", "Large", "Enterprise"]
SECTOR       = ["General Business", "Industry", "Healthcare", "Energy"]
PRODUCT_LINE = ["Hardware", "Software", "Services", "Analytics"]


def generate_interaction_times(win: bool, T: float = 90.0) -> list:
    """
    Simulate seller-lead interaction timestamps.
    Won leads show temporal clustering near the end (Hawkes-inspired).
    """
    times = []
    if win:
        # Burst of activity near win event (temporal clustering as in paper)
        num_bursts = np.random.randint(3, 7)
        for _ in range(num_bursts):
            center = np.random.uniform(T * 0.5, T * 0.85)
            cluster_size = np.random.randint(2, 5)
            for _ in range(cluster_size):
                t = center + np.random.exponential(2.0)
                if t < T:
                    times.append(round(t, 2))
        # A few random early interactions
        for _ in range(np.random.randint(1, 3)):
            times.append(round(np.random.uniform(0, T * 0.4), 2))
    else:
        # Sparse, unclustered interactions
        num_events = np.random.randint(1, 6)
        times = [round(np.random.uniform(0, T), 2) for _ in range(num_events)]

    return sorted(set(times))


def generate_dataset(n_leads: int = 500) -> pd.DataFrame:
    records = []

    for i in range(n_leads):
        win = np.random.random() < 0.20          # ~20% win rate (matches paper)
        geo = random.choice(GEOGRAPHY)
        deal = random.choice(DEAL_SIZE)
        sector = random.choice(SECTOR)
        product = random.choice(PRODUCT_LINE)
        lead_age = np.random.randint(10, 90)      # days old
        sales_stage = np.random.randint(1, 5)

        interaction_times = generate_interaction_times(win)
        num_interactions = len(interaction_times)

        # Temporal clustering score: std dev of inter-arrival times (low = clustered)
        if num_interactions > 1:
            diffs = np.diff(interaction_times)
            clustering_score = float(np.std(diffs))
            mean_gap = float(np.mean(diffs))
            recency = float(90.0 - interaction_times[-1])  # days since last touch
        else:
            clustering_score = 90.0
            mean_gap = 90.0
            recency = 90.0

        records.append({
            "lead_id":           f"LEAD_{i:04d}",
            "geography":         geo,
            "deal_size":         deal,
            "sector":            sector,
            "product_line":      product,
            "lead_age_days":     lead_age,
            "sales_stage":       sales_stage,
            "num_interactions":  num_interactions,
            "clustering_score":  round(clustering_score, 3),
            "mean_gap_days":     round(mean_gap, 3),
            "recency_days":      round(recency, 3),
            "interaction_times": interaction_times,
            "win":               int(win),
        })

    df = pd.DataFrame(records)
    print(f"[DataGenerator] Generated {n_leads} leads | Win rate: {df['win'].mean():.1%}")
    return df


if __name__ == "__main__":
    df = generate_dataset(500)
    df.to_csv("leads_data.csv", index=False)
    print(df[["lead_id", "num_interactions", "clustering_score", "recency_days", "win"]].head(10))
    print("\nSaved → leads_data.csv")