import numpy as np

from .quotient_graph import QuotientGraph


class Simulation(QuotientGraph):
    def __init__(self, dataset):
        self.dataset = dataset

    def summarize(self, entity, triples):
        partition = self.dataset.get_partition_sub(triples)
        q_triples = self.build_quotient_graph(triples, partition)
        self.set_quotient_triple_to_triples(q_triples, triples)
        return q_triples

    def build_quotient_graph(self, triples, partition):
        triples = np.array(triples)
        ss = triples[:, 0]
        os = triples[:, 2]

        s_masks = [np.isin(ss, cl) for cl in partition]
        o_masks = [np.isin(os, cl) for cl in partition]
        part_map = {i: part for i, part in enumerate(partition)}
        pairs = [(i, j) for i, _ in part_map.items() for j, _ in part_map.items()]

        q_triples = []
        for pair in pairs:
            i, j = pair

            ps = set(triples[np.logical_and(s_masks[i], o_masks[j])][:, 1])
            qs, qo = tuple(part_map[i]), tuple(part_map[j])
            q_triples.extend([(qs, p.item(), qo) for p in ps])

        return q_triples
