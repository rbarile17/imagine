class FactBuilder:
    def __init__(self, dataset):
        self.dataset = dataset
        self.dataset.load_summary()

    def build_triples(self, entity, pred):
        qe = self.dataset.get_quotient_entity(entity)
        qtriples = self.dataset.quotient_entity_to_triples[qe]

        qtriples = [
            (self.dataset.quotient_entities[qs], p, self.dataset.quotient_entities[qo])
            for qs, p, qo in qtriples
        ]

        triples = []
        for qs, p, qo in qtriples:
            if entity in qs:
                triples.extend([(entity, p, o) for o in qo])
            elif entity in qo:
                triples.extend([(s, p, entity) for s in qs])

        triples = [t for t in triples if t != pred]
        existing_triples = self.dataset.entity_to_training_triples[entity]
        triples = [t for t in triples if t not in existing_triples]

        return triples
