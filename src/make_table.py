import json
import os
import pandas as pd

from . import RESULTS_PATH

def make_table():
    list_of_dirs = os.listdir(RESULTS_PATH)

    results = []
    for dir in list_of_dirs:
        metrics_path = RESULTS_PATH / dir / "metrics.json"
        dir = dir.split('_')
        experiment = {
            "model": dir[1],
            "dataset": dir[2],
            "mode": dir[3],
            "summarization": dir[4],
            "entity_density": dir[5],
            "pred_rank": dir[6]
        }
        if experiment["mode"] == "aw":
            experiment["mode"] = "Imagine-W"
        elif experiment["mode"] == "ri":
            experiment["mode"] = "Kelpie-I"
        elif experiment["mode"] == "necessary":
            experiment["mode"] = "Kelpie-W"
        elif experiment["mode"] == "imagine":
            experiment["mode"] = "Imagine-I"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            experiment.update(metrics)
            results.append(experiment)

    results = pd.DataFrame(results)
    # results = results.loc[results["summarization"] != "no"]
    results = results.loc[results["entity_density"] == "any"]
    results = results.loc[results["pred_rank"] == "first"]
    results = results.replace("Imagine-I", r"\textsc{Imagine-I}")
    results = results.replace("Imagine-W", r"\textsc{Imagine-W}")
    results = results.replace("Kelpie-I", r"\textsc{Kelpie-I}")
    results = results.replace("Kelpie-W", r"\textsc{Kelpie-W}")
    results = results.replace("TransE", r"\textsc{TransE}")
    results = results.replace("ConvE", r"\textsc{ConvE}")
    results = results.replace("ComplEx", r"\textsc{ComplEx}")
    results = results.replace("bisimulation", "2bisimulation")
    results = results.replace("simulation", "3simulation")
    results = results.replace("no", "1no")


    results = results.drop(columns=["time"])
    results["delta_h1"] = results["delta_h1"].map(lambda x: f"{x:.3f}")
    results["delta_mrr"] = results["delta_mrr"].map(lambda x: f"{x:.3f}")
    results = results.rename(columns={"delta_h1": "$\Delta H@1$", "delta_mrr": "$\Delta MRR$"})
    results = results.reset_index(drop=True)
    # sort by model, mode, summarization
    results = results.pivot_table(
        index=["model", "mode", "summarization"],
        columns="dataset",
        values=["$\Delta H@1$", "$\Delta MRR$"],
        aggfunc=lambda x: x,
    )
    results = results.swaplevel(axis=1).sort_index(axis=1, level=0)

    results.to_latex("results.tex")

if __name__ == '__main__':
    make_table()
