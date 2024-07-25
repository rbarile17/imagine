import click

import pandas as pd

from .. import DATASETS
from .. import WN18RR
from .. import DATA_PATH

@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
def main(dataset):
    entity_to_id = pd.read_csv(DATA_PATH / dataset / "entity2id.txt", sep="\t", names=["entity", "id"])
    id_to_name = pd.read_csv(DATA_PATH / dataset / "entityid2name.txt", sep="\t", names=["id", "name"])

    if dataset == WN18RR:
        id_to_name["name"] = id_to_name["name"].str.split(",")
        id_to_name["name"] = id_to_name["name"].map(lambda x: x[0][:-1])

    entity_to_name = pd.merge(entity_to_id, id_to_name, on="id")
    entity_to_name["name"] = entity_to_name["name"].str.split(" ")
    entity_to_name["name"] = entity_to_name["name"].map(lambda x: "_".join(x)) 
    entity_to_name = entity_to_name.set_index("entity")["name"].to_dict()
    for split in ["train", "valid", "test"]:
        triples = pd.read_csv(DATA_PATH / dataset / f"{split}.txt", sep="\t", names=["s", "p", "o"])

        triples["s"] = triples["s"].map(entity_to_name.get)
        triples["o"] = triples["o"].map(entity_to_name.get)

        triples.to_csv(DATA_PATH / dataset / f"{split}_named.txt", sep="\t", index=False, header=False)

        print(triples["s"].isnull().sum())
        print(triples["o"].isnull().sum())

if __name__ == "__main__":
    main()
