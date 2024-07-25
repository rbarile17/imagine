from ..dataset import Dataset


class ExplanationBuilder:
    def __init__(self, dataset: Dataset, max_explanation_length: int):
        self.dataset = dataset

        self.length_cap = max_explanation_length

    def build_explanations(self, candidate_triples: list, k: int = 10):
        pass
