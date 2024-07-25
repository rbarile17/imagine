import optuna
import torch

import numpy as np

from pydantic import BaseModel

from tqdm import tqdm

from torch import nn
from torch import optim

from .optimizer import Optimizer

from ..regularizers import L2
from ..models import Model, KelpieModel


class PairwiseRankingOptimizerHyperParams(BaseModel):
    batch_size: int
    epochs: int
    lr: float
    margin: float
    negative_triples_ratio: int
    regularizer_weight: float


class PairwiseRankingOptimizer(Optimizer):
    def __init__(
        self,
        model: Model,
        hp: PairwiseRankingOptimizerHyperParams,
        verbose: bool = True,
    ):
        super().__init__(model=model, hp=hp, verbose=verbose)

        self.batch_size = hp.batch_size
        self.lr = hp.lr
        self.epochs = hp.epochs
        self.margin = hp.margin
        self.negative_triples_ratio = hp.negative_triples_ratio
        self.regularizer_weight = hp.regularizer_weight

        self.loss = nn.MarginRankingLoss(margin=self.margin, reduction="mean").cuda()

        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)
        self.regularizer = L2(self.regularizer_weight)

    def get_hyperparams_class():
        return PairwiseRankingOptimizerHyperParams

    def get_kelpie_class():
        return KelpiePairwiseRankingOptimizer

    def train(
        self,
        training_triples,
        save_path: str = None,
        eval_every: int = -1,
        valid_triples=None,
        trial=None,
        patience=5,
    ):
        inverse_triples = self.dataset.invert_triples(training_triples)
        training_triples = np.vstack((training_triples, inverse_triples))

        self.model.cuda()

        best_valid_metric = None
        epochs_without_improvement = 0

        for e in tqdm(range(1, self.epochs + 1), disable=not self.verbose):
            self.epoch(training_triples=training_triples, batch_size=self.batch_size)

            is_eval_epoch = eval_every > 0 and e % eval_every == 0
            if valid_triples is not None and is_eval_epoch:
                self.model.eval()
                with torch.no_grad():
                    ranks = self.evaluator.evaluate(valid_triples)
                    metrics = self.evaluator.get_metrics(ranks)

                if trial:
                    trial.report(metrics["h1"], e)

                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

                if best_valid_metric is None or metrics["h1"] > best_valid_metric:
                    best_valid_metric = metrics["h1"]
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    break

        if save_path is not None:
            print("Saving model...")
            torch.save(self.model.state_dict(), save_path)

        return e

    def epoch(self, batch_size: int, training_triples):
        np.random.shuffle(training_triples)
        repeated = np.repeat(training_triples, self.negative_triples_ratio, axis=0)

        positive_triples = torch.from_numpy(repeated)
        size = torch.Size([len(positive_triples)])
        head_or_tail = torch.randint(high=2, size=size)
        random_entities = torch.randint(high=self.dataset.num_entities, size=size)

        mask_head = head_or_tail == 1
        positive_heads = positive_triples[:, 0].type(torch.LongTensor)
        corrupted_heads = torch.where(mask_head, random_entities, positive_heads)
        mask_tail = ~mask_head
        positive_tails = positive_triples[:, 2].type(torch.LongTensor)
        corrupted_tails = torch.where(mask_tail, random_entities, positive_tails)
        negative_triples = (corrupted_heads, positive_triples[:, 1], corrupted_tails)
        negative_triples = torch.stack(negative_triples, dim=1)

        positive_triples.cuda()
        negative_triples.cuda()
        self.model.train()

        num_triples = len(training_triples)
        with tqdm(
            total=num_triples, unit="ex", disable=not self.verbose, leave=False
        ) as p:
            p.set_description("Train loss")
            batch_start = 0

            while batch_start < num_triples:
                batch_end = min(batch_start + batch_size, len(training_triples))
                positive_batch = positive_triples[batch_start:batch_end]
                negative_batch = negative_triples[batch_start:batch_end]

                l = self.step_on_batch(positive_batch, negative_batch)
                batch_start += batch_size
                p.update(batch_size)
                p.set_postfix(loss=f"{l.item():.2f}")

    def step_on_batch(self, positive_batch, negative_batch):
        self.optimizer.zero_grad()

        positive_scores, positive_factors = self.model.forward(positive_batch)
        negative_scores, negative_factors = self.model.forward(negative_batch)
        target = torch.tensor([-1], dtype=torch.float).cuda()

        l_fit = self.loss(positive_scores, negative_scores, target)

        positive_reg = self.regularizer.forward(positive_factors)
        negative_reg = self.regularizer.forward(negative_factors)
        l_reg = (positive_reg + negative_reg) / 2

        loss = l_fit + l_reg

        loss.backward()
        self.optimizer.step()

        return loss


class KelpiePairwiseRankingOptimizer(PairwiseRankingOptimizer):
    def __init__(self, model: KelpieModel, hp: dict, verbose: bool = True):
        super().__init__(model=model, hp=hp, verbose=verbose)
        self.kelpie_entity = model.kelpie_entity

    def epoch(self, batch_size: int, training_triples):
        np.random.shuffle(training_triples)
        repeated = np.repeat(training_triples, self.negative_triples_ratio, axis=0)

        positive_triples = torch.from_numpy(repeated)
        size = torch.Size([len(positive_triples)])
        random_entities = torch.randint(high=self.dataset.num_entities, size=size)
        head_or_tail = torch.randint(high=2, size=size)

        mask_head = head_or_tail == 1
        positive_heads = positive_triples[:, 0].type(torch.LongTensor)
        corrupted_heads = torch.where(mask_head, random_entities, positive_heads)
        mask_tail = ~mask_head
        positive_tails = positive_triples[:, 2].type(torch.LongTensor)
        corrupted_tails = torch.where(mask_tail, random_entities, positive_tails)
        negative_triples = (corrupted_heads, positive_triples[:, 1], corrupted_tails)
        negative_triples = torch.stack(negative_triples, dim=1)

        positive_triples.cuda()
        negative_triples.cuda()
        self.model.train()

        num_triples = len(training_triples)
        with tqdm(total=num_triples, unit="ex", disable=not self.verbose) as p:
            p.set_description("Train loss")
            batch_start = 0

            while batch_start < num_triples:
                batch_end = min(batch_start + batch_size, len(training_triples))
                positive_batch = positive_triples[batch_start:batch_end]
                negative_batch = negative_triples[batch_start:batch_end]

                l = self.step_on_batch(positive_batch, negative_batch)

                self.model.update_embeddings()

                batch_start += batch_size
                p.update(batch_size)
                p.set_postfix(loss=str(round(l.item(), 6)))
