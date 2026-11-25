"""
Section 4.5 Efficiency and deployment realism
Creates:
  - Table6_efficiency_report_by_testing_version.csv
  - Table6_efficiency_report_with_mean_std.csv
  - Accuracy_Efficiency_tradeoff_latency_vs_dice.png (500 DPI)
  - Accuracy_Efficiency_tradeoff_params_vs_dice.png (500 DPI)

Inputs:
  - efficiency_report_Data_1.csv
  - efficiency_report_Data_2.csv
  - efficiency_report_Data_3.csv

Notes:
  - The provided efficiency CSVs contain a single row (LiteSSMPolypNet). If you have multiple models/baselines,
    append them as additional rows with a 'Model' column and adapt the scatter plotting section.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

paths = {'Dataset 1': '/mnt/data/efficiency_report_Data_1.csv', 'Dataset 2': '/mnt/data/efficiency_report_Data_2.csv', 'Dataset 3': '/mnt/data/efficiency_report_Data_3.csv'}

# (Optional) replace with your real aggregate Dice if different:
dice_notta = {'Dataset 1': [0.9062, 0.9305, 0.7935, 0.7807, 0.8987], 'Dataset 2': [0.9194, 0.9454, 0.8227, 0.8061, 0.9109], 'Dataset 3': [0.9071, 0.9525, 0.8236, 0.7933, 0.8862]}
dice_tta = {'Dataset 1': [0.9168, 0.9406, 0.8066, 0.8114, 0.8991], 'Dataset 2': [0.919, 0.947, 0.8277, 0.8076, 0.9199], 'Dataset 3': [0.9082, 0.9527, 0.8204, 0.7951, 0.8924]}

OUT_DIR = Path(".")

def main():
    rows = []
    for tv, p in paths.items():
        df = pd.read_csv(p)
        if len(df) != 1:
            raise ValueError(f"{tv} expected 1 row, got {len(df)}")
        r = df.iloc[0].to_dict()
        r["testing_version"] = tv
        rows.append(r)

    table6 = pd.DataFrame(rows)[[
        "testing_version",
        "Params(M)",
        "Latency_ms_per_img_NoTTA",
        "PeakMem_MB_NoTTA",
        "Latency_ms_per_img_TTA",
        "PeakMem_MB_TTA",
        "LatencyMultiplier_TTA/NoTTA",
    ]].sort_values("testing_version").reset_index(drop=True)

    num_cols = [c for c in table6.columns if c != "testing_version"]
    mean_row = {"testing_version": "MEAN"}
    std_row = {"testing_version": "STD"}
    for c in num_cols:
        mean_row[c] = float(pd.to_numeric(table6[c], errors="coerce").mean())
        std_row[c] = float(pd.to_numeric(table6[c], errors="coerce").std(ddof=1))

    table6_summary = pd.concat([table6, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    table6.to_csv(OUT_DIR / "Table6_efficiency_report_by_testing_version.csv", index=False)
    table6_summary.to_csv(OUT_DIR / "Table6_efficiency_report_with_mean_std.csv", index=False)

    avg_dice_notta = np.mean([np.mean(v) for v in dice_notta.values()])
    avg_dice_tta = np.mean([np.mean(v) for v in dice_tta.values()])

    lat_notta = float(mean_row["Latency_ms_per_img_NoTTA"])
    lat_tta = float(mean_row["Latency_ms_per_img_TTA"])
    params_m = float(mean_row["Params(M)"])

    plt.rcParams.update({
        "font.size": 15,
        "font.weight": "normal",
        "axes.labelweight": "normal",
        "axes.titleweight": "normal",
        "axes.linewidth": 2.0,
        "xtick.major.width": 2.0,
        "ytick.major.width": 2.0,
    })

    # Latency vs Dice
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter([lat_notta], [avg_dice_notta], s=120, marker="o", label="LiteSSMPolypNet (NoTTA)")
    ax.scatter([lat_tta], [avg_dice_tta], s=140, marker="^", label="LiteSSMPolypNet (TTA)")
    ax.annotate("NoTTA", (lat_notta, avg_dice_notta), textcoords="offset points", xytext=(8, 8))
    ax.annotate("TTA", (lat_tta, avg_dice_tta), textcoords="offset points", xytext=(8, 8))
    ax.set_xlabel("Latency (ms / image)")
    ax.set_ylabel("Average Dice (mean over datasets)")
    ax.set_title("Accuracy–Efficiency trade-off (latency vs Dice)")
    ax.grid(True, linewidth=1.2, alpha=0.35)
    ax.legend(loc="best", frameon=True, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Accuracy_Efficiency_tradeoff_latency_vs_dice.png", dpi=500, bbox_inches="tight")
    plt.close(fig)

    # Params vs Dice
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter([params_m], [avg_dice_notta], s=120, marker="o", label="LiteSSMPolypNet (NoTTA)")
    ax.scatter([params_m], [avg_dice_tta], s=140, marker="^", label="LiteSSMPolypNet (TTA)")
    ax.annotate("NoTTA", (params_m, avg_dice_notta), textcoords="offset points", xytext=(8, 8))
    ax.annotate("TTA", (params_m, avg_dice_tta), textcoords="offset points", xytext=(8, 8))
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Average Dice (mean over datasets)")
    ax.set_title("Accuracy–Efficiency trade-off (params vs Dice)")
    ax.grid(True, linewidth=1.2, alpha=0.35)
    ax.legend(loc="best", frameon=True, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Accuracy_Efficiency_tradeoff_params_vs_dice.png", dpi=500, bbox_inches="tight")
    plt.close(fig)

    print("Saved Table 6 + trade-off figures (500 DPI).")

if __name__ == "__main__":
    main()
