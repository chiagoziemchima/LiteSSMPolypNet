"""
Paired statistical testing (Wilcoxon signed-rank) + effect sizes (rank-biserial correlation)
for per-image segmentation metrics across 3 regimes (Data_1/Data_2/Data_3), and summary tables.

Outputs:
  - Table_model_comparison_means_std.csv
  - Table3_wilcoxon_pvalues_effectsizes.csv
  - Table_overall_pooled_tests.csv

Assumptions:
  - Pairing is by row order (same image order across regimes).

Dependencies:
  pip install numpy pandas scipy
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata

datasets = {'Kvasir': {'Data_1': '/mnt/data/Kvasir_per_image_TTA_Data_1.csv', 'Data_2': '/mnt/data/Kvasir_per_image_TTA_Data_2.csv', 'Data_3': '/mnt/data/Kvasir_per_image_TTA_Data_3.csv'}, 'CVC-300': {'Data_1': '/mnt/data/CVC-300_per_image_TTA_Data_1.csv', 'Data_2': '/mnt/data/CVC-300_per_image_TTA_Data_2.csv', 'Data_3': '/mnt/data/CVC-300_per_image_TTA_Data_3.csv'}, 'CVC-ClinicDB': {'Data_1': '/mnt/data/CVC-ClinicDB_per_image_TTA_Data_1.csv', 'Data_2': '/mnt/data/CVC-ClinicDB_per_image_TTA_Data_2.csv', 'Data_3': '/mnt/data/CVC-ClinicDB_per_image_TTA_Data_3.csv'}, 'CVC-ColonDB': {'Data_1': '/mnt/data/CVC-ColonDB_per_image_TTA_Data_1.csv', 'Data_2': '/mnt/data/CVC-ColonDB_per_image_TTA_Data_2.csv', 'Data_3': '/mnt/data/CVC-ColonDB_per_image_TTA_Data_3.csv'}, 'ETIS-LaribPolypDB': {'Data_1': '/mnt/data/ETIS-LaribPolypDB_per_image_TTA_Data_1.csv', 'Data_2': '/mnt/data/ETIS-LaribPolypDB_per_image_TTA_Data_2.csv', 'Data_3': '/mnt/data/ETIS-LaribPolypDB_per_image_TTA_Data_3.csv'}}

REGIMES = ['Data_1', 'Data_2', 'Data_3']
COMPARISONS = [('Data_2', 'Data_1'), ('Data_3', 'Data_1'), ('Data_3', 'Data_2')]

METRICS = {'Dice': {'higher_is_better': True}, 'Hausdorff95': {'higher_is_better': False}}
ALPHA = 0.05

def _to_oriented(x: np.ndarray, higher_is_better: bool) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x if higher_is_better else -x

def rank_biserial_effect(x: np.ndarray, y: np.ndarray, higher_is_better: bool) -> float:
    xo = _to_oriented(x, higher_is_better)
    yo = _to_oriented(y, higher_is_better)
    d = xo - yo
    d = d[~np.isnan(d)]
    d = d[d != 0.0]
    if d.size == 0:
        return np.nan
    r = rankdata(np.abs(d))
    w_plus = r[d > 0].sum()
    w_minus = r[d < 0].sum()
    return (w_plus - w_minus) / (w_plus + w_minus)

def paired_wilcoxon_p(x: np.ndarray, y: np.ndarray, higher_is_better: bool):
    xo = _to_oriented(x, higher_is_better)
    yo = _to_oriented(y, higher_is_better)
    d = xo - yo
    mask = (~np.isnan(d)) & (d != 0.0)
    if mask.sum() == 0:
        return np.nan, 0
    res = wilcoxon(xo[mask], yo[mask], alternative="two-sided", zero_method="wilcox")
    return float(res.pvalue), int(mask.sum())

def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n, dtype=float)
    prev = 0.0
    for k, idx in enumerate(order):
        mult = n - k
        val = mult * pvals[idx]
        val = max(val, prev)
        prev = val
        adj[idx] = min(val, 1.0)
    return adj

def main():
    dfs = {}
    for ds, paths in datasets.items():
        dfs[ds] = {r: pd.read_csv(paths[r]) for r in REGIMES}
        nrows = {r: len(dfs[ds][r]) for r in REGIMES}
        if len(set(nrows.values())) != 1:
            raise ValueError(f"[{ds}] Row counts differ across regimes: {nrows}")
        for metric in METRICS:
            for r in REGIMES:
                if metric not in dfs[ds][r].columns:
                    raise KeyError(f"[{ds}/{r}] Missing '{metric}'. Found: {list(dfs[ds][r].columns)}")

    # Model/regime performance summary
    perf_rows = []
    for ds in datasets.keys():
        for r in REGIMES:
            for metric in METRICS:
                x = dfs[ds][r][metric].to_numpy(dtype=float)
                perf_rows.append({
                    "dataset": ds,
                    "regime": r,
                    "metric": metric,
                    "mean": float(np.nanmean(x)),
                    "std": float(np.nanstd(x, ddof=1)),
                    "median": float(np.nanmedian(x)),
                    "n": int(np.sum(~np.isnan(x))),
                })
    table_perf = pd.DataFrame(perf_rows).sort_values(["dataset", "metric", "regime"]).reset_index(drop=True)

    # Table 3 tests
    test_rows = []
    for ds in datasets.keys():
        for (rA, rB) in COMPARISONS:
            for metric, cfg in METRICS.items():
                x = dfs[ds][rA][metric].to_numpy(dtype=float)
                y = dfs[ds][rB][metric].to_numpy(dtype=float)

                p, n_eff = paired_wilcoxon_p(x, y, cfg["higher_is_better"])
                rbc = rank_biserial_effect(x, y, cfg["higher_is_better"])

                xo = _to_oriented(x, cfg["higher_is_better"])
                yo = _to_oriented(y, cfg["higher_is_better"])
                med_diff = float(np.nanmedian(xo - yo))

                test_rows.append({
                    "dataset": ds,
                    "comparison": f"{rA} vs {rB}",
                    "metric": metric,
                    "n_effective": n_eff,
                    "p_value": p,
                    "effect_size_rbc": rbc,
                    "median_diff_oriented": med_diff,
                    "significant_(p<0.05)": (p < ALPHA) if np.isfinite(p) else False,
                })
    table3 = pd.DataFrame(test_rows).sort_values(["dataset", "metric", "comparison"]).reset_index(drop=True)

    # Holm adjust within each dataset+metric family of 3 comparisons
    table3["p_value_holm_within_dataset_metric"] = np.nan
    for (ds, metric), sub_idx in table3.groupby(["dataset", "metric"]).groups.items():
        pvals = table3.loc[sub_idx, "p_value"].to_numpy(dtype=float)
        adj = holm_adjust(pvals)
        table3.loc[sub_idx, "p_value_holm_within_dataset_metric"] = adj
    table3["significant_holm_(p<0.05)"] = table3["p_value_holm_within_dataset_metric"] < ALPHA

    # Pooled tests
    pooled_rows = []
    for (rA, rB) in COMPARISONS:
        for metric, cfg in METRICS.items():
            xs, ys = [], []
            for ds in datasets.keys():
                xs.append(dfs[ds][rA][metric].to_numpy(dtype=float))
                ys.append(dfs[ds][rB][metric].to_numpy(dtype=float))
            x_all = np.concatenate(xs, axis=0)
            y_all = np.concatenate(ys, axis=0)

            p, n_eff = paired_wilcoxon_p(x_all, y_all, cfg["higher_is_better"])
            rbc = rank_biserial_effect(x_all, y_all, cfg["higher_is_better"])

            xo = _to_oriented(x_all, cfg["higher_is_better"])
            yo = _to_oriented(y_all, cfg["higher_is_better"])
            med_diff = float(np.nanmedian(xo - yo))

            pooled_rows.append({
                "dataset": "POOLED_ALL",
                "comparison": f"{rA} vs {rB}",
                "metric": metric,
                "n_effective": n_eff,
                "p_value": p,
                "effect_size_rbc": rbc,
                "median_diff_oriented": med_diff,
                "significant_(p<0.05)": (p < ALPHA) if np.isfinite(p) else False,
            })
    table_overall = pd.DataFrame(pooled_rows).sort_values(["metric", "comparison"]).reset_index(drop=True)

    table_perf.to_csv("Table_model_comparison_means_std.csv", index=False)
    table3.to_csv("Table3_wilcoxon_pvalues_effectsizes.csv", index=False)
    table_overall.to_csv("Table_overall_pooled_tests.csv", index=False)

    print("Saved: Table_model_comparison_means_std.csv")
    print("Saved: Table3_wilcoxon_pvalues_effectsizes.csv")
    print("Saved: Table_overall_pooled_tests.csv")

if __name__ == "__main__":
    main()
