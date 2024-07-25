import click
import json

import torch

from . import MODEL_REGISTRY
from .evaluation import Evaluator

from .. import DATASETS, MODELS
from .. import CONFIGS_PATH, MODELS_PATH, PREDICTIONS_PATH

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

    print("Evaluating model...")
    evaluator = Evaluator(model=model)
    ranks = evaluator.evaluate(triples=dataset.testing_triples)
    metrics = evaluator.get_metrics(ranks)
    df_output = evaluator.get_df_output(triples=dataset.testing_triples, ranks=ranks)

    # entity_dict = {k: model.entity_embeddings[v].cpu().detach().numpy() for k, v in dataset.entity_to_id.items()}

    with open(PREDICTIONS_PATH / "lp_metrics.json", "r") as f:
        lp_metrics = json.load(f)
    if model_name not in lp_metrics:
        lp_metrics[model_name] = {}
    if dataset_name not in lp_metrics[model_name]:
        lp_metrics[model_name][dataset_name] = {}
    lp_metrics[model_name][dataset_name] = metrics    

    with open(PREDICTIONS_PATH / "lp_metrics.json", "w") as f:
        json.dump(lp_metrics, f, indent=4)

    df_output.to_csv(PREDICTIONS_PATH / f"{model_name}_{dataset_name}.csv", index=False, sep=";")


if __name__ == "__main__":
    main()
