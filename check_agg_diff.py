import numpy as np
from itertools import combinations

NPZ = "results/xai/umamba_mtl/fold_0/10022_1000022.npz"
d = np.load(NPZ, allow_pickle=True)

methods = ["saliency", "integrated_gradients", "gradient_shap", "occlusion",
           "occlusion_tz", "occlusion_pz", "occlusion_ch_baseline", "ablation"]
agg_suffixes = ["", "mean", "abs_sum", "abs_avg"]
agg_labels = {"": "sum", "mean": "mean", "abs_sum": "abs\_sum", "abs_avg": "abs\_avg"}

rows = []

for method in methods:
    key_suffix_pairs = [(method if s == "" else f"{method}_{s}", s)
                        for s in agg_suffixes
                        if (method if s == "" else f"{method}_{s}") in d]
    if len(key_suffix_pairs) < 2:
        continue
    for (ka, sa), (kb, sb) in combinations(key_suffix_pairs, 2):
        a, b = d[ka], d[kb]
        if a.shape != b.shape or a.size == 0:
            continue
        abs_diff = np.abs(a - b)
        qs = np.nanquantile(abs_diff, [0.5, 0.95, 0.99, 1.0])
        rows.append({
            "method": method.replace("_", r"\_"),
            "a": agg_labels[sa],
            "b": agg_labels[sb],
            "mean": float(np.nanmean(abs_diff)),
            "q50":  float(qs[0]),
            "q95":  float(qs[1]),
            "q99":  float(qs[2]),
            "max":  float(qs[3]),
            "n_diff": int(np.sum(~np.isclose(a, b, equal_nan=True))),
            "total": a.size,
        })

# LaTeX table
print(r"\begin{table}[ht]")
print(r"\centering")
print(r"\begin{tabular}{llcrrrrrr}")
print(r"\toprule")
print(r"Method & Pair & $N_{\neq}$ & Mean & Median & $q_{95}$ & $q_{99}$ & Max \\")
print(r"\midrule")

prev_method = None
for r in rows:
    if r["method"] != prev_method and prev_method is not None:
        print(r"\midrule")
    prev_method = r["method"]
    pair = f"{r['a']} vs {r['b']}"
    print(
        f"  {r['method']} & {pair} & {r['n_diff']}/{r['total']} & "
        f"{r['mean']:.4f} & {r['q50']:.4f} & {r['q95']:.4f} & {r['q99']:.4f} & {r['max']:.4f} \\\\"
    )

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\caption{Pixel-wise absolute differences between aggregation methods per XAI method.}")
print(r"\label{tab:agg_diff}")
print(r"\end{table}")
