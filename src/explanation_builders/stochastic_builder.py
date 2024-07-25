import itertools
import random
import time

import numpy as np

from .explanation_builder import ExplanationBuilder
from .summarization import Simulation, Bisimulation

from .. import key


class StochasticBuilder(ExplanationBuilder):
    def __init__(
        self, xsi, engine, summarization: str = None, max_explanation_length: int = 4
    ):
        dataset = engine.dataset
        super().__init__(dataset=dataset, max_explanation_length=max_explanation_length)

        self.window_size = 10

        self.xsi = xsi
        self.engine = engine

        self.summarization = None
        if summarization == "simulation":
            self.summarization = Simulation(dataset)
        elif summarization == "bisimulation":
            self.summarization = Bisimulation(dataset)
        elif summarization == "bisimulation_d1":
            self.summarization = Bisimulation(dataset, depth=1)

    def build_explanations(self, pred, candidate_triples: list, k: int = 10):
        start = time.time()
        if self.summarization is not None:
            summary_triples = self.summarization.summarize(pred[0], candidate_triples)
            if len(summary_triples) > 0:
                candidate_triples = summary_triples
            else:
                self.summarization = None

        triple_to_rel = self.explore_singleton_rules(pred, candidate_triples)

        triple_to_rel_sorted = triple_to_rel.items()
        triple_to_rel_sorted = sorted(triple_to_rel_sorted, key=key, reverse=True)
        rule_to_rel = [((t,), rel) for (t, rel) in triple_to_rel_sorted]

        triples_number = len(triple_to_rel)
        rels_num = triples_number
        _, best = rule_to_rel[0]
        explore_compound_rules = True
        if best > self.xsi:
            explore_compound_rules = False

        if explore_compound_rules:
            for rule_length in range(2, min(triples_number, self.length_cap) + 1):
                (cur_rule_to_rel, cur_rels_num) = self.explore_compound_rules(
                    pred, candidate_triples, rule_length, triple_to_rel
                )
                rels_num += cur_rels_num
                cur_rule_to_rel = cur_rule_to_rel.items()
                cur_rule_to_rel = sorted(cur_rule_to_rel, key=key, reverse=True)
                rule_to_rel += cur_rule_to_rel

                _, current_best = cur_rule_to_rel[0]
                if current_best > best:
                    best = current_best
                if best > self.xsi:
                    break

                rule_length += 1

        rule_to_rel = sorted(
            rule_to_rel, key=lambda x: (x[1], 1 / len(x[0])), reverse=True
        )
        rule_to_rel = rule_to_rel[:k]

        mapped_rule_to_rel = []
        if self.summarization:
            for rule, rel in rule_to_rel:
                mapped_rule = self.summarization.map_rule(rule)
                mapped_rule = self.dataset.labels_triples(mapped_rule)

                labels_rule = []
                for q_triple in rule:
                    s_part, p, o_part = q_triple
                    s_part = [self.dataset.id_to_entity[e] for e in s_part]
                    o_part = [self.dataset.id_to_entity[e] for e in o_part]
                    p = self.dataset.id_to_relation[p]
                    labels_rule.append((s_part, p, o_part))
                
                mapped_rule_to_rel.append((labels_rule, mapped_rule, rel))
        else:
            mapped_rule_to_rel = rule_to_rel

            mapped_rule_to_rel = [
                (self.dataset.labels_triples(rule), rel) for rule, rel in mapped_rule_to_rel
            ]

        end = time.time()
        return {
            "triple": self.dataset.labels_triple(pred),
            "rule_to_relevance": mapped_rule_to_rel,
            "#relevances": rels_num,
            "execution_time": end - start,
        }


    def explore_singleton_rules(self, pred, triples: list):
        triple_to_relevance = {}

        for triple in triples:
            mapped_triple = [triple]
            if self.summarization:
                mapped_triple = self.summarization.map_rule(mapped_triple)

            printable_triple = self.dataset.printable_nple(mapped_triple)
            print(f"\tComputing relevance for rule: {printable_triple}")

            relevance = self.engine.compute_relevance(pred, mapped_triple)
            triple_to_relevance[triple] = relevance
            print(f"\tRelevance = {relevance:.3f}")
        return triple_to_relevance

    def explore_compound_rules(
        self, pred, triples: list, length: int, triple_to_relevance: dict
    ):
        rules = itertools.combinations(triples, length)
        rules = [(r, self.compute_rule_prescore(r, triple_to_relevance)) for r in rules]
        rules = sorted(rules, key=lambda x: x[1], reverse=True)

        terminate = False
        best = -1e6
        sliding_window = [None for _ in range(self.window_size)]

        rule_to_relevance = {}
        computed_relevances = 0
        for i, (rule, _) in enumerate(rules):
            if terminate:
                break
            mapped_rule = rule
            if self.summarization:
                mapped_rule = self.summarization.map_rule(rule)

            print(
                "\tComputing relevance for rule: \n\t\t"
                f"{self.dataset.printable_nple(mapped_rule)}"
            )
            relevance = self.engine.compute_relevance(pred, mapped_rule)
            print(f"\tRelevance = {relevance:.3f}")
            rule_to_relevance[rule] = relevance
            computed_relevances += 1

            sliding_window[i % self.window_size] = relevance

            if relevance > self.xsi:
                return rule_to_relevance, computed_relevances
            elif relevance >= best:
                best = relevance
            elif i >= self.window_size:
                avg_window_relevance = sum(sliding_window) / self.window_size
                terminate_threshold = avg_window_relevance / best
                random_value = random.random()
                terminate = random_value > terminate_threshold

        return rule_to_relevance, computed_relevances

    def compute_rule_prescore(self, rule, triple_to_relevance):
        return sum([triple_to_relevance[triple] for triple in rule])
