import click
import json
import optuna

from . import MODEL_REGISTRY

from .. import DATASETS, MODELS
from .. import CONFIGS_PATH

from .evaluation import Evaluator

from ..dataset import Dataset
from ..utils import set_seeds

def map_model_trial_hp(model, trial):
    scat = trial.suggest_categorical

    map = {
        "ComplEx": {
            "model": {
                "dimension": 2 ** trial.suggest_int("dimension", 6, 8),  # powers of 2
                "init_scale": 1e-3,  # from KBC https://github.com/facebookresearch/kbc/blob/main/kbc/models.py
            },
            "optimizer": {
                "batch_size": 8192,  # exploit the GPU memory
                "optimizer_name": "Adagrad",  # standard choice
                "epochs": 1000,
                "regularizer_weight": 0,  # avoid regularization implemented in Kelpie
                "regularizer_name": "N3",  # from KBC https://github.com/facebookresearch/kbc/blob/main/kbc/models.py
                "lr": trial.suggest_float("lr", 0, 0.05),  # standard range
                "decay1": 0.9,  # from KBC https://github.com/facebookresearch/kbc/blob/main/kbc/models.py
                "decay2": 0.999,  # from KBC https://github.com/facebookresearch/kbc/blob/main/kbc/models.py
            },
        },
        "ConvE": {
            "model": {
                # from original paper
                "dimension": 200,
                # from original paper
                "input_do_rate": scat("input_do_rate", [0, 0.1, 0.2]),
                # from original paper
                "fmap_do_rate": scat("fmap_do_rate", [0, 0.1, 0.2, 0.3]),
                # from original paper
                "hid_do_rate": scat("hid_do_rate", [0, 0.1, 0.3, 0.5]),
                "hidden_layer_size": 9728,  # from original code
            },
            "optimizer": {
                "batch_size": 8192,  # exploit the GPU memory
                # from original paper
                "label_smoothing": scat("label_smoothing", [0, 0.1, 0.2, 0.3]),
                "epochs": 1000,
                "lr": trial.suggest_float("lr", 0, 0.05),  # standard range
                "decay": 0.995,  # from original code
            },
        },
        "TransE": {
            "model": {
                "norm": trial.suggest_int("norm", 1, 2),  # from original paper
                "dimension": 2 ** trial.suggest_int("dimension", 6, 8),  # powers of 2
            },
            "optimizer": {
                "batch_size": 8192,  # exploit the GPU memory
                "epochs": 1000,
                "lr": trial.suggest_float("lr", 0, 0.05),  # standard range
                "margin": scat("margin", [1, 2, 10]),  # from original paper
                # standard values
                "negative_triples_ratio": scat("negative_triples_ratio", [5, 10, 15]),
                "regularizer_weight": 0,  # avoid regularization implemented in Kelpie
            },
        },
    }

    return map[model]

def map_model_best_hp(model, best):
    map = {
        "ComplEx": {
            "model": {
                "dimension": 2 ** best["dimension"],
                "init_scale": 1e-3,
            },
            "optimizer": {
                "batch_size": 8192,
                "optimizer_name": "Adagrad",
                "epochs": 1000,
                "regularizer_weight": 0,
                "regularizer_name": "N3",
                "lr": best["lr"],
                "decay1": 0.9,
                "decay2": 0.999,
            },
        },
        "ConvE": {
            "model": {
                "dimension": 200,
                "input_do_rate": best["input_do_rate"],
                "fmap_do_rate": best["fmap_do_rate"],
                "hid_do_rate": best["hid_do_rate"],
                "hidden_layer_size": 9728,
            },
            "optimizer": {
                "batch_size": 8192,
                "label_smoothing": best["label_smoothing"],
                "epochs": 1000,
                "lr": best["lr"],
                "decay": 0.995,
            },
        },
        "TransE": {
            "model": {
                "norm": best["norm"],
                "dimension": 2 ** best["dimension"],
            },
            "optimizer": {
                "batch_size": 8192,
                "epochs": 1000,
                "lr": best["lr"],
                "margin": best["margin"],
                "negative_triples_ratio": best["negative_triples_ratio"],
                "regularizer_weight": 0,
            },
        },
    }

    return map[model]

def objective(trial, model, dataset):
    hp = map_model_trial_hp(model, trial)
    model_hp = hp["model"]
    optimizer_hp = hp["optimizer"]

    print(f"Initializing model {model}...")
    model_class = MODEL_REGISTRY[model]["class"]
    optimizer_class = MODEL_REGISTRY[model]["optimizer"]
    hyperparams_class = model_class.get_hyperparams_class()
    model_hp = hyperparams_class(**model_hp)
    model = model_class(dataset=dataset, hp=model_hp, init_random=True)
    model.to("cuda")

    optimizer_params = optimizer_class.get_hyperparams_class()(**optimizer_hp)
    optimizer = optimizer_class(model=model, hp=optimizer_params)

    training_triples = dataset.training_triples
    valid_triples = dataset.validation_triples

    optimizer.train(training_triples, None, 5, valid_triples, trial)

    evaluator = Evaluator(model=model)
    ranks = evaluator.evaluate(triples=valid_triples)
    metrics = evaluator.get_metrics(ranks)

    return metrics["h1"]


@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
@click.option("--model", type=click.Choice(MODELS))
def main(dataset, model):
    set_seeds(42)

    print(f"Loading dataset {dataset}...")
    dataset_name = dataset
    dataset = Dataset(dataset=dataset_name)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, model, dataset),
        timeout=4 * 60 * 60,
    )

    config = {}
    config["model"] = model
    best_hp = map_model_best_hp(model, study.best_params)
    config["model_params"] = best_hp["model"]
    config["training"] = best_hp["optimizer"]

    output_path = CONFIGS_PATH / f"{model}_{dataset_name}.json"
    with open(output_path, "w") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    main()
