def rank_recipients(df, model):
    FEATURES = [
        "donor_age",
        "recipient_age",
        "donor_bg",
        "recipient_bg",
        "health_score",
        "urgency_score",
        "distance",
        "compatibility_score",
        "organ_type",
        "dataset_source"
    ]

    X = df[FEATURES]

    df["ml_score"] = model.predict(X)

    df["final_score"] = (
        0.4 * df["ml_score"] +
        0.3 * df["urgency_score"] +
        0.2 * df["compatibility_score"] -
        0.1 * df["distance"]
    )

    return df.sort_values(by="final_score", ascending=False)