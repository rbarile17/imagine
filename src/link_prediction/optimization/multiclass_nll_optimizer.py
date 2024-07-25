import optuna
import torch
import numpy as np

from pydantic import BaseModel
from tqdm import tqdm

from torch import optim, nn

from .optimizer import Optimizer

from ..models import Model, KelpieModel
from ..regularizers import N3, N2


class MultiClassNLLOptimizerHyperParams(BaseModel):
    optimizer_name: str
    batch_size: int
    epochs: int
    lr: float
    decay1: float
    decay2: float
    regularizer_name: str
    regularizer_weight: float


class MultiClassNLLOptimizer(Optimizer):
    def __init__(
        self, model: Model, hp: MultiClassNLLOptimizerHyperParams, verbose: bool = True
    ):
        Optimizer.__init__(self, model=model, hp=hp, verbose=verbose)

        self.optimizer_name = hp.optimizer_name
        self.batch_size = hp.batch_size
        self.epochs = hp.epochs
        self.lr = hp.lr
        self.decay1, self.decay2 = hp.decay1, hp.decay2
        self.regularizer_name = hp.regularizer_name
        self.regularizer_weight = hp.regularizer_weight

        optimizers = {"Adagrad": optim.Adagrad, "Adam": optim.Adam, "SGD": optim.SGD}
        optimizer_args = {"params": model.parameters(), "lr": self.lr}
        if self.optimizer_name == "Adam":
            optimizer_args["betas"] = (self.decay1, self.decay2)

        regularizers = {"N3": N3, "N2": N2}

        self.optimizer = optimizers[self.optimizer_name](**optimizer_args)
        self.regularizer = regularizers[self.regularizer_name](self.regularizer_weight)

    def get_kelpie_class():
        return KelpieMultiClassNLLOptimizer

    def get_hyperparams_class():
        return MultiClassNLLOptimizerHyperParams

    def train(
        self,
        training_triples,
        save_path=None,
        eval_every=-1,
        valid_triples=None,
        trial=None,
        patience=5,
    ):
        inverse_triples = self.model.dataset.invert_triples(training_triples)
        training_triples = np.vstack((training_triples, inverse_triples))
        training_triples = torch.from_numpy(training_triples).cuda()

        batch_size = min(self.batch_size, len(training_triples))

        best_valid_metric = None
        epochs_without_improvement = 0

        for e in tqdm(range(1, self.epochs + 1), disable=not self.verbose):
            self.epoch(batch_size, training_triples)

            is_eval_epoch = eval_every > 0 and (e + 1) % eval_every == 0
            if valid_triples is not None and is_eval_epoch:
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
            print("\tSaving model...")
            torch.save(self.model.state_dict(), save_path)

        return e

    def epoch(self, batch_size: int, training_triples):
        perm = torch.randperm(training_triples.shape[0])
        perm_triples = training_triples[perm, :]
        loss = nn.CrossEntropyLoss(reduction="mean")

        num_triples = perm_triples.shape[0]
        with tqdm(
            total=num_triples, unit="ex", disable=not self.verbose, leave=False
        ) as p:
            p.set_description("Train loss")

            batch_start = 0
            while batch_start < num_triples:
                batch_end = min(batch_start + batch_size, num_triples)
                batch = perm_triples[batch_start:batch_end]
                l = self.step_on_batch(loss, batch)

                batch_start += self.batch_size

                p.update(batch.shape[0])
                p.set_postfix(loss=f"{l.item():.2f}")

    def step_on_batch(self, loss, batch):
        predictions, factors = self.model.forward(batch)
        truth = batch[:, 2]

        l_fit = loss(predictions, truth)
        l_reg = self.regularizer.forward(factors)
        l = l_fit + l_reg

        self.optimizer.zero_grad()
        l.backward()
        self.optimizer.step()

        return l


class KelpieMultiClassNLLOptimizer(MultiClassNLLOptimizer):
    def __init__(
        self,
        model: KelpieModel,
        hp: MultiClassNLLOptimizerHyperParams,
        verbose: bool = True,
    ):
        super().__init__(model=model, hp=hp, verbose=verbose)

    def epoch(self, batch_size: int, triples: np.array):
        perm_triples = triples[torch.randperm(triples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction="mean")

        num_triples = perm_triples.shape[0]
        with tqdm(total=num_triples, unit="ex", disable=not self.verbose) as p:
            p.set_description("Train loss")

            batch_start = 0
            while batch_start < triples.shape[0]:
                batch = perm_triples[batch_start : batch_start + batch_size]
                l = self.step_on_batch(loss, batch)

                self.model.update_embeddings()

                batch_start += self.batch_size
                p.update(batch.shape[0])
                p.set_postfix(loss=f"{l.item():.2f}")
