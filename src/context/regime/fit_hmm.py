from pathlib import Path
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


def main() -> None:

    data_dir = Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    data_path = data_dir / "spy_features.csv"

    if not data_path.exists():
        raise FileNotFoundError("Feature file not found. Run build_features.py first.")

    df = pd.read_csv(data_path)

    # --- Prepare data ---
    X = df[["return", "volatility"]].values

    # --- Scale features ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Fit HMM ---
    model = GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=5000,
        random_state=42
    )

    model.fit(X_scaled)

    # --- Predict regimes ---
    hidden_states = model.predict(X_scaled)

    df["regime"] = hidden_states

    # --- Save results ---
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "spy_regimes.csv"

    df.to_csv(output_path, index=False)

    print("Model fitted successfully.")
    print("Regime counts:")
    print(df["regime"].value_counts())

    print("\nMeans by regime:")
    print(df.groupby("regime")[["return", "volatility"]].mean())


if __name__ == "__main__":
    main()