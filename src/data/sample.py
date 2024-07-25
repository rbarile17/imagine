import click

import pandas as pd

from .. import DATASETS
from .. import DATA_PATH

@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
def main(dataset):
    triples = pd.read_csv(DATA_PATH / dataset / "test_named.txt", sep="\t", names=["s", "p", "o"])

    triples = triples.sample(300)
    triples = triples.reset_index(drop=True)

    triples.to_csv(DATA_PATH / dataset / f"sample_test_named.txt", sep="\t", index=False, header=False)


if __name__ == "__main__":
    main()
