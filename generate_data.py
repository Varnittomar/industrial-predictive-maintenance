"""Generate reproducible synthetic machine sensor data for a demo."""
from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_ROWS = 5000

def main():
    rng = np.random.default_rng(RANDOM_SEED)
    machine_type = rng.choice(["L", "M", "H"], size=N_ROWS, p=[0.6, 0.3, 0.1])
    air_temp = rng.normal(300.0, 2.0, N_ROWS)
    process_temp = air_temp + rng.normal(10.0, 1.5, N_ROWS)
    speed = np.clip(rng.normal(1500, 180, N_ROWS), 800, 2500)
    torque = np.clip(rng.normal(40, 9, N_ROWS), 5, 80)
    wear = rng.integers(0, 251, N_ROWS)

    # Illustrative synthetic failure mechanism; not a real-world physical model.
    logit = (
        -5.2
        + 0.035 * (torque - 40)
        + 0.012 * (wear - 100)
        + 0.65 * (machine_type == "H")
        + 0.0015 * np.maximum(speed - 1750, 0)
        + 0.45 * (process_temp - 310)
    )
    probability = 1 / (1 + np.exp(-logit))
    failure = rng.binomial(1, probability)

    df = pd.DataFrame({
        "machine_id": [f"M{i:05d}" for i in range(1, N_ROWS + 1)],
        "machine_type": machine_type,
        "air_temperature_k": air_temp.round(2),
        "process_temperature_k": process_temp.round(2),
        "rotational_speed_rpm": speed.round(1),
        "torque_nm": torque.round(2),
        "tool_wear_min": wear,
        "failure": failure,
    })

    # Add a small number of missing sensor readings to demonstrate cleaning.
    for col in ["air_temperature_k", "torque_nm"]:
        missing_idx = rng.choice(df.index, size=20, replace=False)
        df.loc[missing_idx, col] = np.nan

    out = Path("data")
    out.mkdir(exist_ok=True)
    path = out / "machine_sensor_data.csv"
    df.to_csv(path, index=False)
    print(f"Saved {len(df):,} rows to {path}")
    print(f"Failure rate (excluding missing labels): {df['failure'].mean():.2%}")

if __name__ == "__main__":
    main()
