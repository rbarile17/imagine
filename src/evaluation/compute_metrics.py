import click
import json

from pathlib import Path

from .. import DATASETS, METHODS, MODELS
from .. import CRIAGE, DATA_POISONING, IMAGINE, I_KELPIE, I_KELPIEPP, KELPIE, KELPIEPP, W_IMAGINE
from .. import FIRST, NOT_FIRST

from .. import RESULTS_PATH

from ..explanation_builders.summarization import NO_SUMMARIZATION, SUMMARIZATIONS

def hits_at_k(ranks, k):
    count = 0.0
    for rank in ranks:
        if rank <= k:
            count += 1.0
    return round(count / float(len(ranks)), 3)


def mrr(ranks):
    reciprocal_rank_sum = 0.0
    for rank in ranks:
        reciprocal_rank_sum += 1.0 / float(rank)
    return round(reciprocal_rank_sum / float(len(ranks)), 3)


def mr(ranks):
    rank_sum = 0.0
    for rank in ranks:
        rank_sum += float(rank)
    return round(rank_sum / float(len(ranks)), 3)

@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
@click.option("--model", type=click.Choice(MODELS))
@click.option("--method", type=click.Choice(METHODS), default=KELPIE)
@click.option("--summarization", type=click.Choice(SUMMARIZATIONS), default=NO_SUMMARIZATION)
def main(dataset, model, method, summarization):
    if method in [CRIAGE, DATA_POISONING, KELPIE, KELPIEPP, W_IMAGINE]:
        pred_rank = FIRST
    elif method in [IMAGINE, I_KELPIE, I_KELPIEPP]:
        pred_rank = NOT_FIRST
    explanations_path = f"{method}_{model}_{dataset}_{summarization}_{pred_rank}"
    explanations_path = Path(explanations_path)
    explanations_path = RESULTS_PATH / explanations_path

    with open(explanations_path / "output_end_to_end.json", "r") as input_file:
        triple_to_details = json.load(input_file)

    ranks = [float(details["rank"]) for details in triple_to_details]
    new_ranks = [float(details["new_rank"]) for details in triple_to_details]

    original_mrr, original_h1 = mrr(ranks), hits_at_k(ranks, 1)
    new_mrr, new_h1 = mrr(new_ranks), hits_at_k(new_ranks, 1)
    mrr_delta = round(new_mrr - original_mrr, 3)
    h1_delta = round(new_h1 - original_h1, 3)

    explanations_filepath = explanations_path / "output.json"
    with open(explanations_filepath, "r") as input_file:
        explanations = json.load(input_file)
    rels = [x["#relevances"] for x in explanations]
    times = [x["execution_time"] for x in explanations]
    rels = sum(rels)
    time = sum(times)

    metrics = {}
    metrics["delta_h1"] = h1_delta
    metrics["delta_mrr"] = mrr_delta
    metrics["rels"] = rels
    metrics["time"] = time

    metrics_file = explanations_path / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
