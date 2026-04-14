from model.preprocessing import *


def prepare_training_data(donors_df):
    train_df = donors_df.copy()
    train_df["recipient_age"] = 40
    train_df["recipient_bg"] = "A+"
    train_df["urgency_score"] = 5
    train_df["organ_type"] = train_df["donor_organ"].fillna("Kidney").str.lower()
    train_df["dataset_source"] = "live"
    train_df["donor_health_score"] = train_df["health_score"]
    train_df["recipient_health_score"] = 0.72
    train_df["wait_time_days"] = 30
    train_df["distance_km"] = train_df["distance"].fillna(0.5).apply(lambda value: value * 100 if value <= 1 else value)
    train_df["compatibility_score"] = train_df.apply(
        lambda row: compatibility_score(str(row["donor_bg"]).replace("+", "").replace("-", ""), "A"),
        axis=1,
    )
    train_df["success_probability"] = (
        0.5 * train_df["compatibility_score"]
        + 0.3 * (train_df["urgency_score"] / 10)
        + 0.2 * train_df["health_score"]
        - 0.1 * train_df["distance"].fillna(0.5)
    )
    return train_df
