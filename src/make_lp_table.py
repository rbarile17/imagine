import json

import pandas as pd

from . import PREDICTIONS_PATH

with open(PREDICTIONS_PATH / "lp_metrics.json", 'r') as f:
    data = json.load(f)

rows = []
for model, datasets in data.items():
    for dataset, metrics in datasets.items():
        row = {
            'model': model,
            'dataset': dataset,
            'mrr': metrics['mrr'],
            'h1': metrics['h1']
        }
        rows.append(row)

df = pd.DataFrame(rows)
df["mrr"] = df["mrr"].map(lambda x: f"{x:.3f}")
df["h1"] = df["h1"].map(lambda x: f"{x:.3f}")
df = df.rename(columns={"h1": "$H@1$", "mrr": "$MRR$"})
df = df.loc[df["dataset"].isin(["DB50K", "DB100K", "YAGO4-20"])]
df = df.reset_index(drop=True)
df = df.pivot_table(
    index="model",
    columns="dataset",
    values=["$H@1$", "$MRR$"],
    aggfunc=lambda x: x,
)
df = df.swaplevel(axis=1).sort_index(axis=1, level=0)
df.to_latex("lp_results.tex")
