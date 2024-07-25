import time

from .explanation_builder import ExplanationBuilder
from .. import key


class DataPoisoningBuilder(ExplanationBuilder):
    def __init__(self, engine):
        super().__init__(engine.dataset, 1)
        self.engine = engine

    def build_explanations(self, pred, candidate_triples: list, k: int = 10):
        start = time.time()

        rule_to_rel = {}

        for triple in candidate_triples:
            printable_triple = self.dataset.printable_triple(triple)
            print(f"\tComputing relevance for rule: {printable_triple}")

            relevance = self.engine.compute_relevance(pred, triple)
            print(f"\tRelevance = {relevance:.3f}")

            rule_to_rel[triple] = relevance

        rule_to_rel = sorted(rule_to_rel.items(), key=key, reverse=True)[:k]
        mapped_rule_to_rel = rule_to_rel
        mapped_rule_to_rel = [
            ([self.dataset.labels_triple(rule)], rel) for rule, rel in mapped_rule_to_rel
        ]

        end = time.time()
        return  {
            "triple": self.dataset.labels_triple(pred),
            "rule_to_relevance": mapped_rule_to_rel,
            "#relevances": len(candidate_triples),
            "execution_time": end - start,
        }
