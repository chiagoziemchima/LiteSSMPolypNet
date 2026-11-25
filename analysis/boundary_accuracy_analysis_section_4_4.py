"""
Section 4.4 Boundary accuracy analysis (HD95/ASSD) — reproducible tables + figures.

What it produces:
  - Table5_boundary_summary_by_testing_version.csv
  - Table5_boundary_summary_pooled.csv
  - Boxplot_HD95_grouped_by_testing_version.png (500 DPI)
  - Boxplot_ASSD_grouped_by_testing_version.png (500 DPI)
  - Scatter_Dice_vs_HD95_colored_by_dataset.png (500 DPI)

Assumptions:
  - Pairing is per-image within each CSV.
  - These inputs are per-image TTA metrics (NoTTA vs TTA not included here).

Dependencies:
  pip install numpy pandas matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

paths = {'Kvasir-SEG': {'Dataset 1': '/mnt/data/Kvasir_per_image_TTA_Data_1.csv', 'Dataset 2': '/mnt/data/Kvasir_per_image_TTA_Data_2.csv', 'Dataset 3': '/mnt/data/Kvasir_per_image_TTA_Data_3.csv'}, 'CVC-300': {'Dataset 1': '/mnt/data/CVC-300_per_image_TTA_Data_1.csv', 'Dataset 2': '/mnt/data/CVC-300_per_image_TTA_Data_2.csv', 'Dataset 3': '/mnt/data/CVC-300_per_image_TTA_Data_3.csv'}, 'CVC-ClinicDB': {'Dataset 1': '/mnt/data/CVC-ClinicDB_per_image_TTA_Data_1.csv', 'Dataset 2': '/mnt/data/CVC-ClinicDB_per_image_TTA_Data_2.csv', 'Dataset 3': '/mnt/data/CVC-ClinicDB_per_image_TTA_Data_3.csv'}, 'CVC-ColonDB': {'Dataset 1': '/mnt/data/CVC-ColonDB_per_image_TTA_Data_1.csv', 'Dataset 2': '/mnt/data/CVC-ColonDB_per_image_TTA_Data_2.csv', 'Dataset 3': '/mnt/data/CVC-ColonDB_per_image_TTA_Data_3.csv'}, 'ETIS-Larib': {'Dataset 1': '/mnt/data/ETIS-LaribPolypDB_per_image_TTA_Data_1.csv', 'Dataset 2': '/mnt/data/ETIS-LaribPolypDB_per_image_TTA_Data_2.csv', 'Dataset 3': '/mnt/data/ETIS-LaribPolypDB_per_image_TTA_Data_3.csv'}}

datasets = list(paths.keys())
versions = ["Dataset 1", "Dataset 2", "Dataset 3"]
OUT_DIR = Path(".")

def median_iqr(x: pd.Series):
    x = pd.to_numeric(x, errors="coerce")
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    med = float(np.median(x))
    q1 = float(np.percentile(x, 25))
    q3 = float(np.percentile(x, 75))
    return med, q1, q3, int(len(x))

def load_long():
    dfs = []
    for ds in datasets:
        for v in versions:
            df = pd.read_csv(paths[ds][v])
            df["dataset"] = ds
            df["testing_version"] = v
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def build_table5(all_df: pd.DataFrame):
    rows = []
    for (ds, v), sub in all_df.groupby(["dataset", "testing_version"]):
        hd_med, hd_q1, hd_q3, n1 = median_iqr(sub["Hausdorff95"])
        as_med, as_q1, as_q3, n2 = median_iqr(sub["ASSD"])
        rows.append({
            "dataset": ds,
            "testing_version": v,
            "n_images": min(n1, n2),
            "HD95_median": hd_med,
            "HD95_IQR_Q1": hd_q1,
            "HD95_IQR_Q3": hd_q3,
            "ASSD_median": as_med,
            "ASSD_IQR_Q1": as_q1,
            "ASSD_IQR_Q3": as_q3,
            "HD95_median_[IQR]": f"{hd_med:.3f} [{hd_q1:.3f}, {hd_q3:.3f}]",
            "ASSD_median_[IQR]": f"{as_med:.3f} [{as_q1:.3f}, {as_q3:.3f}]",
        })
    table5_by_version = pd.DataFrame(rows).sort_values(["dataset", "testing_version"]).reset_index(drop=True)

    rows2 = []
    for ds, sub in all_df.groupby("dataset"):
        hd_med, hd_q1, hd_q3, n1 = median_iqr(sub["Hausdorff95"])
        as_med, as_q1, as_q3, n2 = median_iqr(sub["ASSD"])
        rows2.append({
            "dataset": ds,
            "testing_version": "POOLED_ALL_VERSIONS",
            "n_images": min(n1, n2),
            "HD95_median": hd_med,
            "HD95_IQR_Q1": hd_q1,
            "HD95_IQR_Q3": hd_q3,
            "ASSD_median": as_med,
            "ASSD_IQR_Q1": as_q1,
            "ASSD_IQR_Q3": as_q3,
            "HD95_median_[IQR]": f"{hd_med:.3f} [{hd_q1:.3f}, {hd_q3:.3f}]",
            "ASSD_median_[IQR]": f"{as_med:.3f} [{as_q1:.3f}, {as_q3:.3f}]",
        })
    table5_pooled = pd.DataFrame(rows2).sort_values(["dataset"]).reset_index(drop=True)

    table5_by_version.to_csv(OUT_DIR / "Table5_boundary_summary_by_testing_version.csv", index=False)
    table5_pooled.to_csv(OUT_DIR / "Table5_boundary_summary_pooled.csv", index=False)

def set_style():
    plt.rcParams.update({
        "font.size": 15,
        "font.weight": "normal",
        "axes.labelweight": "normal",
        "axes.titleweight": "normal",
        "axes.linewidth": 2.0,
        "xtick.major.width": 2.0,
        "ytick.major.width": 2.0,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
    })

def grouped_boxplot(all_df: pd.DataFrame, metric: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(14, 6))

    base = np.arange(len(datasets))
    offsets = [-0.25, 0.0, 0.25]
    linestyles = ["-", "--", ":"]

    for j, v in enumerate(versions):
        data = []
        for ds in datasets:
            vals = all_df[(all_df["dataset"] == ds) & (all_df["testing_version"] == v)][metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            data.append(vals)
        bp = ax.boxplot(
            data,
            positions=base + offsets[j],
            widths=0.22,
            showfliers=False,
            manage_ticks=False,
        )
        for element in ["boxes", "whiskers", "caps", "medians"]:
            for line in bp[element]:
                line.set_linewidth(2.2)
                line.set_linestyle(linestyles[j])

    ax.set_xticks(base, datasets, rotation=0)
    ax.set_xlabel("Dataset")
    ax.set_ylabel(metric)
    ax.set_title(f"Boxplots of per-image {metric} grouped by testing version")
    ax.grid(True, axis="y", linewidth=1.2, alpha=0.35)

    handles = [Line2D([0], [0], linewidth=2.2, linestyle=ls, label=v) for ls, v in zip(linestyles, versions)]
    ax.legend(handles=handles, title="Testing version", frameon=True, fontsize=12, title_fontsize=12, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=500, bbox_inches="tight")
    plt.close(fig)

def scatter_dice_vs_hd95(all_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 7))
    for ds in datasets:
        sub = all_df[all_df["dataset"] == ds]
        x = pd.to_numeric(sub["Dice"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub["Hausdorff95"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[mask], y[mask], s=14, alpha=0.35, label=ds)

    ax.set_xlabel("Dice")
    ax.set_ylabel("HD95 (Hausdorff95)")
    ax.set_title("Scatter: Dice vs HD95 (all images), colored by dataset")
    ax.grid(True, linewidth=1.2, alpha=0.35)
    ax.legend(loc="best", frameon=True, fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=500, bbox_inches="tight")
    plt.close(fig)

def main():
    set_style()
    all_df = load_long()
    build_table5(all_df)
    grouped_boxplot(all_df, "Hausdorff95", OUT_DIR / "Boxplot_HD95_grouped_by_testing_version.png")
    grouped_boxplot(all_df, "ASSD", OUT_DIR / "Boxplot_ASSD_grouped_by_testing_version.png")
    scatter_dice_vs_hd95(all_df, OUT_DIR / "Scatter_Dice_vs_HD95_colored_by_dataset.png")
    print("Saved Table 5 + boundary figures (500 DPI).")

if __name__ == "__main__":
    main()
