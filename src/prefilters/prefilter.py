from ..dataset import Dataset

TOPOLOGY_PREFILTER = "topology_based"
WEIGHTED_TOPOLOGY_PREFILTER = "weighted_topology_based"
TYPE_PREFILTER = "type_based"
NO_PREFILTER = "none"

key = lambda x: x[1]


class PreFilter:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
