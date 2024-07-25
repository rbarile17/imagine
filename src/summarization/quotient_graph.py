import numpy as np

from tqdm import tqdm

class QuotientGraph:
    def build_quotient_graph(self, partition):
        triples = self.dataset.training_triples
        ss = triples[:, 0]
        os = triples[:, 2]

        s_masks = [np.isin(ss, cl) for cl in partition]
        o_masks = [np.isin(os, cl) for cl in partition]
        part_map = {i: part for i, part in enumerate(partition)}
        pairs = [(i, j) for i, _ in part_map.items() for j, _ in part_map.items()]

        q_triples = []
        for pair in tqdm(pairs):
            i, j = pair

            ps = set(triples[np.logical_and(s_masks[i], o_masks[j])][:, 1])
            q_triples.extend([(i, p, j) for p in ps])

        return q_triples, part_map

    # def plot(self, graph, filename="graph", format="svg", quotient=False):
    #     agraph = to_agraph(graph)
    #     agraph.graph_attr["rankdir"] = "LR"
    #     agraph.graph_attr["pad"] = 0.01

    #     for node in agraph.nodes():
    #         node.attr["shape"] = "rectangle"
    #         node.attr["style"] = "rounded"
    #     for edge in agraph.edges():
    #         edge.attr["arrowsize"] = 0.3
    #         edge.attr["color"] = "red"

    #     agraph.draw(f"pictures/{filename}.{format}", prog="dot", format=format)

    def set_quotient_triple_to_triples(self, q_triples, filter):
        self.quotient_triple_to_triples = {}

        for s_part, p, o_part in q_triples:
            triples = [(s, p, o) for s in s_part for o in o_part]
            triples = [triple for triple in triples if triple in filter]

            if len(triples) > 0:
                self.quotient_triple_to_triples[(s_part, p, o_part)] = triples

    def map_rule(self, rule):
        triples = []
        for q_triple in rule:
            triples += self.quotient_triple_to_triples[q_triple]
        return triples
