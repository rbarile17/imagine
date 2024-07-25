import networkx as nx

from .prefilter import PreFilter, key
from ..dataset import Dataset
from ..utils import jaccard_similarity


class WeightedTopologyPreFilter(PreFilter):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

        self.graph = nx.MultiGraph()
        self.graph.add_nodes_from(list(dataset.id_to_entity.keys()))
        self.graph.add_edges_from([(h, t) for h, _, t in dataset.training_triples])

        entities_semantic = self.dataset.entities_semantic_impl
        entities_semantic = entities_semantic.set_index("entity")
        entities_semantic = entities_semantic[["classes"]].to_dict(orient="index")
        self.entities_semantic = entities_semantic

        self.entity_to_training_triples = self.dataset.entity_to_training_triples

    def select_triples(self, pred, k=50):
        self.pred_s, _, self.pred_o = pred

        triples = self.entity_to_training_triples[self.pred_s]
        triples = sorted(triples)

        results = {t: self.analyze_triple(t) for t in triples}
        results = sorted(results.items(), key=key)
        results = [x[0] for x in results]

        return results[:k]

    def semantic_score(self, entity1, entity2, edge_data):
        entity1_classes = self.entities_semantic[entity1]["classes"]
        entity2_classes = self.entities_semantic[entity2]["classes"]

        return 1 - jaccard_similarity(entity1_classes, entity2_classes)

    def analyze_triple(self, triple):
        s, _, o = triple

        entity = o if s == self.pred_s else s

        try:
            return nx.shortest_path_length(
                self.graph, entity, self.pred_o, weight=self.semantic_score
            )
        except nx.NetworkXNoPath:
            return 1e6
