import networkx as nx
import numpy as np

from bispy import compute_maximum_bisimulation

from .quotient_graph import QuotientGraph

is_tuple = lambda node: isinstance(node, tuple)
contain_tuples = lambda cl: any(is_tuple(node) for node in cl)


class Bisimulation(QuotientGraph):
    def __init__(self, dataset, depth=0):
        self.dataset = dataset
        self.depth = depth

    def preprocess(self, multigraph):
        digraph = nx.DiGraph()

        for s, o, data in multigraph.edges(data=True):
            p = data["label"]

            if not digraph.has_node(s):
                digraph.add_node(s, **{"label": self.dataset.id_to_entity[s]})
            if not digraph.has_node(o):
                digraph.add_node(o, **{"label": self.dataset.id_to_entity[o]})
            po_node = (p, o)
            if not digraph.has_node(po_node):
                po_label = f"{p}_{self.dataset.id_to_entity[o]}"
                digraph.add_node(po_node, **{"label": po_label})

            digraph.add_edge(s, po_node)

        return digraph

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
            ps_filtered = []
            for p in ps:
                if all(any((s, p, o) in triples for o in qo) for s in qs):
                    ps_filtered.append(p)
            q_triples.extend([(qs, p.item(), qo) for p in ps])

        return q_triples

    def summarize(self, entity, triples):
        subgraph = self.dataset.get_subgraph(entity, triples=triples, depth=self.depth)
        digraph = self.preprocess(subgraph)
        partition = self.dataset.get_partition_sub(subgraph)

        for node in digraph.nodes():
            if is_tuple(node):
                partition.append(frozenset({node}))

        bisimulation = compute_maximum_bisimulation(digraph, partition)
        bisimulation = [frozenset(cl) for cl in bisimulation]
        bisimulation = [cl for cl in bisimulation if not contain_tuples(cl)]

        q_triples = self.build_quotient_graph(triples, bisimulation)

        filtered_q_triples = []
        entities = [s for s, _, _ in triples]
        entities.extend([o for _, _, o in triples])

        entity_q_triples = [
            (qs, relation, qo)
            for (qs, relation, qo) in q_triples
            if entity in qs or entity in qo
        ]

        for q_triple in entity_q_triples:
            qs, p, qo = q_triple
            if any([s in entities for s in qs]) and any([o in entities for o in qo]):
                qs = tuple([s for s in qs if s in entities])
                qo = tuple([o for o in qo if o in entities])

                filtered_q_triples.append((qs, p, qo))

        self.set_quotient_triple_to_triples(filtered_q_triples, triples)
        
        return list(self.quotient_triple_to_triples.keys())
