from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    df = pd.read_csv("data/spy_features.csv")
    df["date"] = pd.to_datetime(df["date"])

    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

    # --- Plot returns and volatility ---
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(df["date"], df["return"])
    plt.title("SPY Daily Returns")

    plt.subplot(2, 1, 2)
    plt.plot(df["date"], df["volatility"])
    plt.title("SPY Rolling Volatility (10-day)")

    plt.tight_layout()

    plt.savefig(output_dir / "sanity_check.png")
    plt.show()


if __name__ == "__main__":
    main()