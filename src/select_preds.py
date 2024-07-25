import click

import pandas as pd

from . import DATASETS, MODELS, PRED_RANKS
from . import PREDICTIONS_PATH, SELECTED_PREDICTIONS_PATH

from . import FIRST, NOT_FIRST

from .dataset import Dataset

@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
@click.option("--model", type=click.Choice(MODELS))
@click.option("--pred-rank", type=click.Choice(PRED_RANKS))
def main(dataset, model, pred_rank):
    dataset_name = dataset
    dataset = Dataset(dataset)

    preds_path = PREDICTIONS_PATH / f"{model}_{dataset_name}.csv"
    preds = pd.read_csv(preds_path, sep=";")
    preds.drop("s_rank", axis=1, inplace=True)

    if pred_rank == FIRST:
        preds = preds[preds["o_rank"] == 1]
    elif pred_rank == NOT_FIRST:
        preds = preds[preds["o_rank"] != 1]
    preds.drop(["o_rank"], axis=1, inplace=True)

    preds = preds.sample(100)
    preds = preds.reset_index(drop=True)

    output_path = SELECTED_PREDICTIONS_PATH / f"{model}_{dataset_name}_{pred_rank}.csv"
    preds.to_csv(output_path, sep="\t", index=False, header=False)


if __name__ == "__main__":
    main()
