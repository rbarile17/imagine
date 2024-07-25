import click
import json

import torch

from pathlib import Path

from . import DATASETS, METHODS, MODELS
from . import CONFIGS_PATH, MODELS_PATH, RESULTS_PATH, SELECTED_PREDICTIONS_PATH
from . import CRIAGE, DATA_POISONING, IMAGINE, I_KELPIE, I_KELPIEPP, KELPIE, KELPIEPP, W_IMAGINE
from . import FIRST, NOT_FIRST
from .link_prediction import MODEL_REGISTRY

from .dataset import Dataset
from .explanation_builders import CriageBuilder, DataPoisoningBuilder, StochasticBuilder
from .explanation_builders.summarization import NO_SUMMARIZATION, SUMMARIZATIONS
from .pipeline import ImaginePipeline, NecessaryPipeline
from .prefilters import TopologyPreFilter, CriagePreFilter
from .relevance_engines import (
    AddWorsenPostTrainingEngine,
    NecessaryCriageEngine,
    NecessaryDPEngine,
    ImaginePostTrainingEngine,
    NecessaryPostTrainingEngine,
    RemoveImprovePostTrainingEngine,
)
from .utils import set_seeds

def build_pipeline(model, dataset, hp, method, xsi, summarization):
    if method in [CRIAGE, DATA_POISONING, KELPIE, KELPIEPP]:
        if method == CRIAGE:
            prefilter = CriagePreFilter(dataset)
            engine = NecessaryCriageEngine(model, dataset)
            builder = CriageBuilder(engine)
        elif method == DATA_POISONING:
            prefilter = TopologyPreFilter(dataset)
            engine = NecessaryDPEngine(model, dataset, hp["lr"])
            builder = DataPoisoningBuilder(engine)
        elif method == KELPIE:
            DEFAULT_XSI_THRESHOLD = 5
            xsi = xsi if xsi is not None else DEFAULT_XSI_THRESHOLD
            prefilter = TopologyPreFilter(dataset)
            engine = NecessaryPostTrainingEngine(model, dataset, hp)
            builder = StochasticBuilder(xsi, engine, summarization=summarization)
        pipeline = NecessaryPipeline(dataset, prefilter, builder)
    elif method == IMAGINE:
        DEFAULT_XSI_THRESHOLD = 0.9
        xsi = xsi if xsi is not None else DEFAULT_XSI_THRESHOLD
        prefilter = TopologyPreFilter(dataset)
        engine = ImaginePostTrainingEngine(model, dataset, hp)
        builder = StochasticBuilder(xsi, engine, summarization=summarization)
        pipeline = ImaginePipeline(dataset, prefilter, builder)
    elif method == I_KELPIE or method == I_KELPIEPP:
        DEFAULT_XSI_THRESHOLD = 0.9
        xsi = xsi if xsi is not None else DEFAULT_XSI_THRESHOLD
        prefilter = TopologyPreFilter(dataset)
        engine = RemoveImprovePostTrainingEngine(model, dataset, hp)
        builder = StochasticBuilder(xsi, engine, summarization=summarization)
        pipeline = NecessaryPipeline(dataset, prefilter, builder)
    elif method == W_IMAGINE:
        DEFAULT_XSI_THRESHOLD = 5
        xsi = xsi if xsi is not None else DEFAULT_XSI_THRESHOLD
        prefilter = TopologyPreFilter(dataset)
        engine = AddWorsenPostTrainingEngine(model, dataset, hp)
        builder = StochasticBuilder(xsi, engine, summarization=summarization)
        pipeline = ImaginePipeline(dataset, prefilter, builder)

    return pipeline


@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
@click.option("--model", type=click.Choice(MODELS))
@click.option(
    "--preds",
    type=click.Path(exists=True),
    help="Path of the predictions to explain.",
)
@click.option(
    "--coverage",
    type=int,
    default=10,
    help="Number of entities to convert (sufficient mode only).",
)
@click.option(
    "--skip",
    type=int,
    default=-1,
    help="Number of predictions to skip.",
)
@click.option("--method", type=click.Choice(METHODS), default=IMAGINE)
@click.option(
    "--relevance_threshold",
    type=float,
    help="The relevance acceptance threshold.",
)
@click.option("--summarization", type=click.Choice(SUMMARIZATIONS), default=NO_SUMMARIZATION)
@click.option(
    "--prefilter_threshold",
    type=int,
    default=20,
    help=f"The number of triples to select in pre-filtering.",
)
def main(
    dataset,
    model,
    preds,
    coverage,
    method,
    relevance_threshold,
    prefilter_threshold,
    summarization,
    skip,
):
    set_seeds(42)

    model_config_file = CONFIGS_PATH / f"{model}_{dataset}.json"
    with open(model_config_file, "r") as f:
        model_config = json.load(f)
    model_name = model_config["model"]
    model_path = model_config.get("model_path", MODELS_PATH / f"{model_name}_{dataset}.pt")

    print("Reading preds...")
    if preds is None:
        if method in [CRIAGE, DATA_POISONING, KELPIE, KELPIEPP, W_IMAGINE]:
            pred_rank = FIRST
        elif method in [IMAGINE, I_KELPIE, I_KELPIEPP]:
            pred_rank = NOT_FIRST
        preds = SELECTED_PREDICTIONS_PATH / f"{model}_{dataset}_{pred_rank}.csv"
    with open(preds, "r") as preds:
        preds = [x.strip().split("\t") for x in preds.readlines()]

    output_dir = f"{method}_{model}_{dataset}_{summarization}_{pred_rank}"

    print(f"Loading dataset {dataset}...")
    dataset = Dataset(dataset=dataset)

    print(f"Loading model {model}...")
    model_class = MODEL_REGISTRY[model_name]["class"]
    hyperparams_class = model_class.get_hyperparams_class()
    model_hp = hyperparams_class(**model_config["model_params"])
    model = model_class(dataset=dataset, hp=model_hp, init_random=True)
    model.to("cuda")

    model.load_state_dict(torch.load(model_path))
    model.eval()

    pipeline_hps = model_config["kelpie"]
    if method == DATA_POISONING and model_name == "TransE":
        pipeline_hps = model_config["data_poisoning"]
    pipeline = build_pipeline(
        model,
        dataset,
        pipeline_hps,
        method,
        relevance_threshold,
        summarization,
    )

    Path(RESULTS_PATH / output_dir).mkdir(exist_ok=True)

    explanations = []
    for i, pred in enumerate(preds):
        if i <= skip:
            continue
        s, p, o = pred
        print(f"\nExplaining pred {i}: <{s}, {p}, {o}>")
        pred = dataset.ids_triple(pred)
        explanation = pipeline.explain(pred=pred, prefilter_k=prefilter_threshold)

        explanations.append(explanation)

        output_path = RESULTS_PATH / output_dir / "output.json"
        with open(output_path, "w") as f:
            json.dump(explanations, f, indent=4)

if __name__ == "__main__":
    main()
