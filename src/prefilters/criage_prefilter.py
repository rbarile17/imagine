from collections import defaultdict

from .prefilter import PreFilter
from ..dataset import Dataset


class CriagePreFilter(PreFilter):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

        self.o_to_training_triples = defaultdict(list)
        for h, r, t in dataset.training_triples:
            self.o_to_training_triples[t].append((h, r, t))

    def select_triples(self, pred, k=50):
        pred_s, _, pred_o = pred

        o_as_o_triples = self.o_to_training_triples.get(pred_o, [])
        o_as_o_triples = sorted(o_as_o_triples)
        s_as_o_triples = self.o_to_training_triples.get(pred_s, [])
        s_as_o_triples = sorted(s_as_o_triples)

        if k == -1:
            return o_as_o_triples + s_as_o_triples
        return o_as_o_triples[:k] + s_as_o_triples[:k]
