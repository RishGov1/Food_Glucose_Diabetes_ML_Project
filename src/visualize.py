from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def make_d1namo_response_plot(csv_path, output_path):
    df = pd.read_csv(csv_path)
    cols = [f"g_{x}" for x in [0, 15, 40, 45, 60, 120]]
    means = df.groupby("label")[cols].mean(numeric_only=True)
    means.index = ["Non-diabetic / healthy" if i == 0 else "Type 1 diabetes" for i in means.index]

    x = [0, 15, 40, 45, 60, 120]
    plt.figure(figsize=(10, 6))
    for label, row in means.iterrows():
        plt.plot(x, row.values, marker="o", linewidth=2, label=label)
    plt.xlabel("Minutes after meal")
    plt.ylabel("Glucose (mg/dL)")
    plt.title("D1NAMO post-meal glucose response")
    plt.grid(alpha=0.25)
    plt.legend()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

def make_class_distribution(csv_path, output_path):
    df = pd.read_csv(csv_path)
    counts = df.groupby("label")["subject_id"].nunique()
    labels = ["Healthy", "Type 1 diabetes"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    plt.figure(figsize=(7, 5))
    sns.barplot(x=labels, y=values)
    plt.ylabel("Unique subjects")
    plt.title("D1NAMO subject distribution")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()

if __name__ == "__main__":
    make_d1namo_response_plot(
        "outputs/d1namo_meal_samples.csv",
        "outputs/figures/d1namo_response.png",
    )
    make_class_distribution(
        "outputs/d1namo_meal_samples.csv",
        "outputs/figures/d1namo_class_distribution.png",
    )
