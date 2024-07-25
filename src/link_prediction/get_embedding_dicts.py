import click
import json
import pickle

import torch

import numpy as np

from . import MODEL_REGISTRY

from .. import DATASETS, MODELS
from .. import CONFIGS_PATH, EMBEDDINGS_PATH, MODELS_PATH

from ..dataset import Dataset
from ..utils import set_seeds

def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

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

    entity_dict = {
        k: np.squeeze(model.entity_embeddings[v].cpu().detach().numpy())
        for k, v in dataset.entity_to_id.items()
    }

    relations_dict = {
        k: np.squeeze(model.relation_embeddings[v].cpu().detach().numpy())
        for k, v in dataset.relation_to_id.items()
    }

    output_path = EMBEDDINGS_PATH / f"{model_name}_{dataset_name}"
    output_path.mkdir(exist_ok=True, parents=True)

    save_obj(entity_dict, output_path / "entity_embeddings.pkl")
    save_obj(relations_dict, output_path / "relation_embeddings.pkl")


if __name__ == "__main__":
    main()
