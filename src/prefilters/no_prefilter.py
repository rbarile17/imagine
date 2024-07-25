from .prefilter import PreFilter
from ..dataset import Dataset


class NoPreFilter(PreFilter):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

        self.entity_to_training_triples = self.dataset.entity_to_training_triples

    def select_triples(self, pred, k=-1):
        s, _, _ = pred

        return self.entity_to_training_triples[s]
