import torch
import numpy as np

from torch import nn
from ...dataset import Dataset


class Model(nn.Module):
    def __init__(self, dataset: Dataset):
        super().__init__()
        self.dataset = dataset

    def is_minimizer(self):
        pass

    def score(self, triples):
        pass

    def all_scores(self, triples):
        pass

    def forward(self, triples):
        pass

    def predict_triples(self, triples):
        direct_triples = triples
        assert np.all(direct_triples[:, 1] < self.dataset.num_relations)
        direct_scores, tail_ranks = self.predict_tails(direct_triples)

        inverse_triples = self.dataset.invert_triples(direct_triples)
        inverse_scores, head_ranks = self.predict_tails(inverse_triples)

        results = []
        for i in range(direct_triples.shape[0]):
            score = {"tail": direct_scores[i], "head": inverse_scores[i]}
            rank = {"tail": int(tail_ranks[i]), "head": int(head_ranks[i])}

            results.append({"score": score, "rank": rank})

        return results

    def predict_tails(self, triples):
        with torch.no_grad():
            all_scores = self.all_scores(triples)

            targets = torch.zeros(size=(len(triples), 1)).cuda()
            for i, (_, _, tail) in enumerate(triples):
                targets[i, 0] = all_scores[i, tail].item()

            for i, (head, rel, tail) in enumerate(triples):
                filter_out = set(self.dataset.to_filter[(head, rel)])
                default = 1e6 if self.is_minimizer() else -1e6
                all_scores[i, torch.LongTensor(list(filter_out))] = default
                all_scores[i, tail] = targets[i, 0]

            ranks = torch.zeros(len(triples))
            if self.is_minimizer():
                ranks += torch.sum((all_scores <= targets).float(), dim=1).cpu()
            else:
                ranks += torch.sum((all_scores >= targets).float(), dim=1).cpu()
            ranks = ranks.cpu().numpy().tolist()

            all_scores = all_scores.cpu().numpy()
            targets = targets.cpu().numpy()

            scores = [targets[i, 0] for i in range(len(triples))]

        return scores, ranks

    def predict_triple(self, triple):
        assert triple[1] < self.dataset.num_relations

        [result] = self.predict_triples(np.array([triple]))
        return result["score"], result["rank"], result["prediction"]

    def kelpie_model_class(self):
        pass


class KelpieModel:
    @property
    def entity_embeddings(self):
        return self.model.entity_embeddings

    @property
    def dataset(self):
        return self.model.dataset

    def to(self, device):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def eval(self):
        self.model.eval()

    def is_minimizer(self):
        return self.model.is_minimizer()

    def parameters(self):
        return [self.kelpie_entity_emb]

    def forward(self, triples):
        return self.model.forward(triples)

    def all_scores(self, triples):
        return self.model.all_scores(triples)

    def update_embeddings(self):
        with torch.no_grad():
            self.entity_embeddings[self.kelpie_entity] = self.kelpie_entity_emb

    def train(self, mode=True):
        self.training = mode
        for module in self.model.children():
            if not (
                isinstance(module, Model)
                or isinstance(module, torch.nn.BatchNorm1d)
                or isinstance(module, torch.nn.BatchNorm2d)
                or isinstance(module, torch.nn.Linear)
                or isinstance(module, torch.nn.Conv2d)
            ):
                module.train(mode)
        return self

    def kelpie_model_class(self):
        raise Exception(self.__class__.name + " is a KelpieModel.")
