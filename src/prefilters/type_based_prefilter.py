import numpy as np

from multiprocessing.pool import ThreadPool as Pool
from collections import defaultdict

from .prefilter import PreFilter, key
from .. import MAX_PROCESSES
from ..dataset import Dataset


class TypeBasedPreFilter(PreFilter):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

        self.entity_to_training_triples = self.dataset.entity_to_training_triples
        shape = (self.dataset.num_entities, self.dataset.num_relations * 2)
        self.entity_to_relation_vector = defaultdict(lambda: np.zeros(shape))

        self.thread_pool = Pool(processes=MAX_PROCESSES)

        for s, p, o in dataset.training_triples:
            self.entity_to_relation_vector[s][p] += 1
            self.entity_to_relation_vector[o][p + self.dataset.num_relations] += 1

    def select_triples(self, pred, k=50):
        self.counter = 0

        pred_s, _, _ = pred

        triples = self.entity_to_training_triples[pred_s]

        inputs = [(pred, triple) for triple in triples]

        results = self.thread_pool.map(self.analyze_triple, inputs)
        results = {t: results[i] for i, t in enumerate(triples)}
        results = sorted(results.items(), reverse=True, key=key)
        results = [x[0] for x in results]

        return results[:k]

    def analyze_triple(self, input_data):
        pred, triple = input_data

        pred_s, _, pred_o = pred
        s, _, o = triple

        entity_to_analyze = o if pred_s == s else s

        return self._cosine_similarity(pred_o, entity_to_analyze)

    def _cosine_similarity(self, entity1, entity2):
        entity1_vector = self.entity_to_relation_vector[entity1]
        entity2_vector = self.entity_to_relation_vector[entity2]
        dot_product = np.inner(entity1_vector, entity2_vector)
        magnitude = np.linalg.norm(entity1_vector) * np.linalg.norm(entity2_vector)
        return dot_product / magnitude
