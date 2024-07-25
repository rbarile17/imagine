import networkx as nx

from .prefilter import PreFilter
from .. import key
from ..dataset import Dataset

from tqdm import tqdm

class TopologyPreFilter(PreFilter):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

        self.graph = nx.MultiGraph()
        self.graph.add_nodes_from(list(dataset.id_to_entity.keys()))
        self.graph.add_edges_from([(h, t) for h, _, t in dataset.training_triples])

        self.entity_to_training_triples = self.dataset.entity_to_training_triples

    def select_triples(self, pred, k=20, new_triples=[]):
        self.pred_s, _, self.pred_o = pred

        if new_triples != []:
            triples = sorted(new_triples)
        else:
            triples = self.entity_to_training_triples[self.pred_s]
            triples = sorted(triples)

        results = {t: self.analyze_triple(t) for t in tqdm(triples)}
        results = sorted(results.items(), key=key)
        results = [x[0] for x in results]
        return results[:k]

    def analyze_triple(self, triple):
        s, _, o = triple

        entity = o if s == self.pred_s else s

        try:
            return nx.shortest_path_length(self.graph, entity, self.pred_o)
        except nx.NetworkXNoPath:
            return 1e6
