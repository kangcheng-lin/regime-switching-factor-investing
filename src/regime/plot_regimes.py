from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:

    data_dir = Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "spy_regimes.csv"

    if not data_path.exists():
        raise FileNotFoundError("Regime file not found. Run fit_hmm.py first.")

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["close"], linewidth=1)

    for regime in sorted(df["regime"].unique()):
        regime_data = df[df["regime"] == regime]
        plt.scatter(
            regime_data["date"],
            regime_data["close"],
            s=8,
            label=f"Regime {regime}"
        )

    plt.title("SPY Price with HMM Regimes")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()

    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "spy_regimes.png"

    plt.savefig(output_path)
    plt.show()


if __name__ == "__main__":
    main()