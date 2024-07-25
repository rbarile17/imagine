import click
import json

from . import MODEL_REGISTRY

from .. import DATASETS, MODELS
from .. import CONFIGS_PATH, MODELS_PATH

from ..dataset import Dataset
from ..utils import set_seeds


@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
@click.option("--model", type=click.Choice(MODELS))
@click.option("--valid", type=float, default=-1, help="Number of epochs before valid.")
def main(dataset, model, valid):
    set_seeds(42)

    model_config_file = CONFIGS_PATH / f"{model}_{dataset}.json"
    with open(model_config_file, "r") as f:
        model_config = json.load(f)
    model = model_config["model"]
    model_path = model_config.get("model_path", MODELS_PATH / f"{model}_{dataset}.pt")

    print(f"Loading dataset {dataset}...")
    dataset = Dataset(dataset=dataset)

    print(f"Initializing model {model}...")
    model_class = MODEL_REGISTRY[model]["class"]
    optimizer_class = MODEL_REGISTRY[model]["optimizer"]
    hyperparams_class = model_class.get_hyperparams_class()
    model_hp = hyperparams_class(**model_config["model_params"])
    model = model_class(dataset=dataset, hp=model_hp, init_random=True)
    model.to("cuda")

    print("Training model...")
    hp = model_config["training"]
    optimizer_params = optimizer_class.get_hyperparams_class()(**hp)
    optimizer = optimizer_class(model=model, hp=optimizer_params)

    training_triples = dataset.training_triples
    validation_triples = dataset.validation_triples

    epochs = optimizer.train(training_triples, model_path, valid, validation_triples)
    model_config["kelpie"] = model_config["training"].copy()
    model_config["evaluation"] = model_config["training"].copy()
    model_config["kelpie"]["epochs"] = epochs
    model_config["evaluation"]["epochs"] = epochs 
    if model_config["model"] == "TransE":
        model_config["kelpie"]["lr"] = 0.01
    with open(model_config_file, "w") as f:
        json.dump(model_config, f, indent=4)

if __name__ == "__main__":
    main()
