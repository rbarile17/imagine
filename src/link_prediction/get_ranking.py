import click
import json

import torch

from tqdm import tqdm

from . import MODEL_REGISTRY

from .. import DATASETS, MODELS
from .. import CONFIGS_PATH, MODELS_PATH

from ..dataset import Dataset
from ..utils import set_seeds


@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
@click.option("--model", type=click.Choice(MODELS))
def main(dataset, model):
    set_seeds(42)

    model_config_file = CONFIGS_PATH / f"{model}_{dataset}.json"
    with open(model_config_file, "r") as f:
        model_config = json.load(f)
    model_name = model_config["model"]
    model_path = model_config.get("model_path", MODELS_PATH / f"{model_name}_{dataset}.pt")

    print(f"Loading dataset {dataset}...")
    dataset_name = dataset
    dataset = Dataset(dataset=dataset)

    print(f"Initializing model {model_name}...")
    model_class = MODEL_REGISTRY[model_name]["class"]
    hyperparams_class = model_class.get_hyperparams_class()
    model_hp = hyperparams_class(**model_config["model_params"])
    model = model_class(dataset=dataset, hp=model_hp, init_random=True)
    model.to("cuda")

    model.load_state_dict(torch.load(model_path))
    model.eval()

    triples = dataset.testing_triples
    batch_size = 2048
    if len(triples) > batch_size:
        batch_start = 0
        all_scores = []
        num_triples = len(triples)
        with tqdm(total=num_triples, unit="ex", leave=False) as p:
            while batch_start < num_triples:
                batch_end = min(len(triples), batch_start + batch_size)
                cur_batch = triples[batch_start:batch_end]
                all_scores += model.all_scores(cur_batch).cpu()
                batch_start += batch_size
                p.update(batch_size)
            all_scores = torch.stack(all_scores) 
    else:
        all_scores = model.all_scores(cur_batch)

    all_scores = torch.argsort(all_scores, dim=1)

    rankings = []
    for i, triple in enumerate(triples):
        labels_triple = dataset.labels_triple(triple)
        ranking = [dataset.id_to_entity[j.item()] for j in all_scores[i]]
        rankings.append({"triple": labels_triple, "ranking": ranking})

    with open("test.json", "w") as f: 
        json.dump(rankings, f, indent=4)

if __name__ == "__main__":
    main()
