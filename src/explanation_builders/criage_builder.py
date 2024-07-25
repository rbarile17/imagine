import time

from .explanation_builder import ExplanationBuilder
from .. import key


class CriageBuilder(ExplanationBuilder):
    def __init__(self, engine, reverse=False):
        super().__init__(engine.dataset, 1)

        self.reverse = reverse
        self.engine = engine

    def build_explanations(self, pred, candidate_triples: list, k: int = 10):
        start = time.time()
        pred_s, _, _ = pred
        rule_to_rel = {}

        for triple in candidate_triples:
            _, _, o = triple
            printable_triple = self.dataset.printable_triple(triple)
            print(f"\tComputing relevance for rule: {printable_triple}")

            perspective = "head" if o == pred_s else "tail"
            relevance = self.engine.compute_relevance(pred, triple, perspective)
            print(f"\tRelevance = {relevance:.3f}")

            rule_to_rel[triple] = relevance

        rule_to_rel = sorted(rule_to_rel.items(), key=key, reverse=True)[:k]
        mapped_rule_to_rel = rule_to_rel
        f = self.dataset.labels_triple
        mapped_rule_to_rel = [([f(rule)], rel) for rule, rel in mapped_rule_to_rel]
        end = time.time()

        return  {
            "triple": self.dataset.labels_triple(pred),
            "rule_to_relevance": mapped_rule_to_rel,
            "#relevances": len(candidate_triples),
            "execution_time": end - start,
        }
