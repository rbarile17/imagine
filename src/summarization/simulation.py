from .quotient_graph import QuotientGraph


class Simulation(QuotientGraph):
    def __init__(self, dataset):
        self.dataset = dataset

    def summarize(self):
        partition = self.dataset.get_partition()
        q_triples = self.build_quotient_graph(partition)

        return q_triples