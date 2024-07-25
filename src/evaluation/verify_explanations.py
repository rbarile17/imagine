import copy
import click
import json

import numpy
import torch

from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from .. import DATASETS, METHODS, MODELS, MODES
from .. import CRIAGE, DATA_POISONING, IMAGINE, I_KELPIE, I_KELPIEPP, KELPIE, KELPIEPP, W_IMAGINE
from .. import NECESSARY, SUFFICIENT
from .. import FIRST, NOT_FIRST

from .. import CONFIGS_PATH, MODELS_PATH, RESULTS_PATH

from ..dataset import MANY_TO_ONE, ONE_TO_ONE

from ..explanation_builders.summarization import NO_SUMMARIZATION, SUMMARIZATIONS

from ..dataset import Dataset
from ..link_prediction import MODEL_REGISTRY
from ..link_prediction.models import TransE
from ..utils import set_seeds

@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
@click.option("--model", type=click.Choice(MODELS))
@click.option("--method", type=click.Choice(METHODS), default=KELPIE)
@click.option("--summarization", type=click.Choice(SUMMARIZATIONS), default=NO_SUMMARIZATION)
def main(
    dataset,
    model,
    method,
    summarization,
):
    set_seeds(42)

    model_config_file = CONFIGS_PATH / f"{model}_{dataset}.json"
    with open(model_config_file, "r") as f:
        model_config = json.load(f)
    model = model_config["model"]
    model_path = model_config.get("model_path", MODELS_PATH / f"{model}_{dataset}.pt")

    if method in [CRIAGE, DATA_POISONING, KELPIE, KELPIEPP, W_IMAGINE]:
        pred_rank = FIRST
    elif method in [IMAGINE, I_KELPIE, I_KELPIEPP]:
        pred_rank = NOT_FIRST
    explanations_path = f"{method}_{model}_{dataset}_{summarization}_{pred_rank}"
    explanations_path = Path(explanations_path)
    explanations_path = RESULTS_PATH / explanations_path
    with open(explanations_path / "output.json", "r") as f:
        explanations = json.load(f)

    print(f"Loading dataset {dataset}...")
    dataset = Dataset(dataset=dataset)

    print(f"Loading model {model}...")
    model_class = MODEL_REGISTRY[model]["class"]
    optimizer_class = MODEL_REGISTRY[model]["optimizer"]
    hyperparams_class = model_class.get_hyperparams_class()
    model_hp = hyperparams_class(**model_config["model_params"])
    model = model_class(dataset=dataset, hp=model_hp, init_random=True)
    model.to("cuda")

    model.load_state_dict(torch.load(model_path))
    model.eval()

    preds = []
    triple_to_best_rule = {}

    if method in [KELPIE, KELPIEPP, I_KELPIE, I_KELPIEPP]:
        triple_to_best_rule = defaultdict(list)
        for explanation in explanations:
            pred = dataset.ids_triple(explanation["triple"])
            preds.append(pred)
            tmp = explanation["rule_to_relevance"][0]
            if len(tmp) == 3:
                _, best_rule, _ = tmp
            else:
                best_rule, _ = tmp
            best_rule = [dataset.ids_triple(triple) for triple in best_rule]

            triple_to_best_rule[pred] = best_rule

        triples_to_remove = []

        for pred in preds:
            triples_to_remove += triple_to_best_rule[pred]

        new_dataset = copy.deepcopy(dataset)

        new_dataset.remove_training_triples(triples_to_remove)

        results = model.predict_triples(numpy.array(preds))
        results = {triple: result for triple, result in zip(preds, results)}
        new_model = model_class(dataset=new_dataset, hp=model_hp, init_random=True)

        hp = model_config["evaluation"]
        optimizer_params = optimizer_class.get_hyperparams_class()(**hp)
        optimizer = optimizer_class(model=new_model, hp=optimizer_params)
        optimizer.train(training_triples=new_dataset.training_triples)
        new_model.eval()

        new_results = new_model.predict_triples(numpy.array(preds))
        new_results = {triple: result for triple, result in zip(preds, new_results)}

        evaluations = []
        for pred in preds:
            result = results[pred]
            new_result = new_results[pred]

            score = result["score"]["tail"]
            rank = result["rank"]["tail"]
            new_score = new_result["score"]["tail"]
            new_rank = new_result["rank"]["tail"]

            evaluation = {
                "triple_to_explain": dataset.labels_triple(pred),
                "rule": [
                    dataset.labels_triple(triple)
                    for triple in triple_to_best_rule[pred]
                ],
                "score": str(score),
                "rank": str(rank),
                "new_score": str(new_score),
                "new_rank": str(new_rank),
            }

            evaluations.append(evaluation)
    elif method == IMAGINE or method == W_IMAGINE:
        triple_to_best_rule = defaultdict(list)
        for explanation in explanations:
            pred = dataset.ids_triple(explanation["triple"])
            preds.append(pred)
            tmp = explanation["rule_to_relevance"][0]
            if len(tmp) == 3:
                _, best_rule, _ = tmp
            else:
                best_rule, _ = tmp
            best_rule = [dataset.ids_triple(triple) for triple in best_rule]

            triple_to_best_rule[pred] = best_rule

        triples_to_add = []

        for pred in preds:
            triples_to_add += triple_to_best_rule[pred]

        new_dataset = copy.deepcopy(dataset)

        new_dataset.add_training_triples(triples_to_add)

        results = model.predict_triples(numpy.array(preds))
        results = {triple: result for triple, result in zip(preds, results)}
        new_model = model_class(dataset=new_dataset, hp=model_hp, init_random=True)

        hp = model_config["evaluation"]
        optimizer_params = optimizer_class.get_hyperparams_class()(**hp)
        optimizer = optimizer_class(model=new_model, hp=optimizer_params)
        optimizer.train(training_triples=new_dataset.training_triples)
        new_model.eval()

        new_results = new_model.predict_triples(numpy.array(preds))
        new_results = {triple: result for triple, result in zip(preds, new_results)}

        evaluations = []
        for pred in preds:
            result = results[pred]
            new_result = new_results[pred]

            score = result["score"]["tail"]
            rank = result["rank"]["tail"]
            new_score = new_result["score"]["tail"]
            new_rank = new_result["rank"]["tail"]

            evaluation = {
                "triple_to_explain": dataset.labels_triple(pred),
                "rule": [
                    dataset.labels_triple(triple)
                    for triple in triple_to_best_rule[pred]
                ],
                "score": str(score),
                "rank": str(rank),
                "new_score": str(new_score),
                "new_rank": str(new_rank),
            }

            evaluations.append(evaluation)


    with open(explanations_path / "output_end_to_end.json", "w") as outfile:
        json.dump(evaluations, outfile, indent=4)


if __name__ == "__main__":
    main()
