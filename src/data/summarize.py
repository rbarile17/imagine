import click

import pandas as pd

from .. import DATASETS
from .. import DATA_PATH

from ..dataset import Dataset
from ..summarization import Simulation

@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
def main(dataset):
    dataset = Dataset(dataset=dataset)
    summarization = Simulation(dataset)
    q_triples, part_map = summarization.summarize()

    output_path = DATA_PATH / dataset.name
    if dataset.name == "DB100K":
        output_path = output_path / "mapped"
    pd.DataFrame(q_triples).to_csv(output_path / "train_summarization.txt", index=False, header=False, sep="\t")
    pd.DataFrame(part_map.items()).to_csv(output_path / "part_map.csv", index=False, header=False, sep="\t")

if __name__ == "__main__":
    main()
