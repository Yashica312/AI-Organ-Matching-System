"""Generate a synthetic dataset for organ donor-recipient matching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROW_COUNT = 1000
BLOOD_GROUPS = ["A", "B", "AB", "O"]
OUTPUT_PATH = Path(__file__).resolve().parent / "dataset.csv"
RANDOM_SEED = 42


def calculate_compatibility(donor_bg: str, recipient_bg: str) -> float:
    """Return the rule-based blood group compatibility score."""
    if donor_bg == recipient_bg:
        return 1.0
    if donor_bg == "O":
        return 0.9
    return 0.4


def haversine_distance(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Calculate great-circle distance between two latitude/longitude points."""
    earth_radius = 6371.0

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return earth_radius * c


def normalize_to_unit_interval(values: np.ndarray) -> np.ndarray:
    """Scale an array to the 0-1 range."""
    min_value = values.min()
    max_value = values.max()

    if np.isclose(min_value, max_value):
        return np.ones_like(values)
    return (values - min_value) / (max_value - min_value)


def main() -> None:
    # Create a reproducible random number generator.
    rng = np.random.default_rng(RANDOM_SEED)

    # Randomly assign donor and recipient blood groups from the allowed set.
    donor_bg = rng.choice(BLOOD_GROUPS, size=ROW_COUNT)
    recipient_bg = rng.choice(BLOOD_GROUPS, size=ROW_COUNT)

    # Generate urgency scores from 1 to 10 and health scores from 0.5 to 1.0.
    urgency = rng.integers(1, 11, size=ROW_COUNT)
    health = rng.uniform(0.5, 1.0, size=ROW_COUNT)

    # Generate random donor and recipient geographic coordinates.
    donor_lat = rng.uniform(-90, 90, size=ROW_COUNT)
    donor_lon = rng.uniform(-180, 180, size=ROW_COUNT)
    recipient_lat = rng.uniform(-90, 90, size=ROW_COUNT)
    recipient_lon = rng.uniform(-180, 180, size=ROW_COUNT)

    # Calculate travel distance between donor and recipient using lat/lon pairs.
    raw_distance_km = haversine_distance(donor_lat, donor_lon, recipient_lat, recipient_lon)

    # Scale distance to a smaller 0-1 penalty range for the success formula.
    distance = normalize_to_unit_interval(raw_distance_km)

    # Compute compatibility for each donor-recipient pair using the given rules.
    compatibility = np.array(
        [calculate_compatibility(donor, recipient) for donor, recipient in zip(donor_bg, recipient_bg, strict=True)]
    )

    # Compute the raw target score from compatibility, urgency, health, and distance penalty.
    raw_success_prob = (
        (0.5 * compatibility)
        + (0.3 * (urgency / 10.0))
        + (0.2 * health)
        - (0.1 * distance)
    )

    # Normalize the target so the final success probability lies between 0 and 1.
    success_prob = normalize_to_unit_interval(raw_success_prob)

    # Build the final dataset with the required columns.
    dataset = pd.DataFrame(
        {
            "donor_bg": donor_bg,
            "recipient_bg": recipient_bg,
            "urgency": urgency,
            "health": np.round(health, 4),
            "distance": np.round(distance, 4),
            "compatibility": np.round(compatibility, 4),
            "success_prob": np.round(success_prob, 4),
        }
    )

    # Save the synthetic dataset as CSV in the project folder.
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
