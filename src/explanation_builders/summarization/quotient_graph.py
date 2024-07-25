class QuotientGraph:
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
