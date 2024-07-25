from ..utils import jaccard_similarity, pairs


def compute_semantic_similarity_entities(entities_semantic, entity1, entity2):
    return jaccard_similarity(
        entities_semantic.loc[entity1]["classes"],
        entities_semantic.loc[entity2]["classes"],
    )


def compute_semantic_similarity_relations(relations_semantic, relation1, relation2):
    rel1_domains = relations_semantic.loc[relation1]["domains"]
    rel2_domains = relations_semantic.loc[relation2]["domains"]
    rel1_ranges = relations_semantic.loc[relation1]["ranges"]
    rel2_ranges = relations_semantic.loc[relation2]["ranges"]

    domains_similarity = jaccard_similarity(set(rel1_domains), set(rel2_domains))
    ranges_similarity = jaccard_similarity(set(rel1_ranges), set(rel2_ranges))
    return (domains_similarity + ranges_similarity) / 2


def compute_semantic_similarity_triples(dataset, rule, known_common_entity):
    rule_relations = set([p for _, p, _ in rule])
    entities_s_adj = set([s if s != known_common_entity else o for (s, _, o) in rule])
    e_pairs = [(ent1, ent2) for (ent1, ent2) in pairs(entities_s_adj)]
    r_pairs = [(rel1, rel2) for (rel1, rel2) in pairs(rule_relations)]

    semantic_similarity_rels = 0
    if len(r_pairs) > 0:
        semantic_similarity_rels = [
            compute_semantic_similarity_relations(dataset.relations_semantic, r1, r2)
            for (r1, r2) in r_pairs
        ]
        semantic_similarity_rels = sum(semantic_similarity_rels) / len(r_pairs)

    semantic_similarity_entities = 0
    if len(e_pairs) > 0:
        semantic_similarity_entities = [
            compute_semantic_similarity_entities(dataset.entities_semantic, e1, e2)
            for (e1, e2) in e_pairs
        ]
        semantic_similarity_entities = sum(semantic_similarity_entities) / len(e_pairs)

    semantic_similarity = (semantic_similarity_entities + semantic_similarity_rels) / 2

    return semantic_similarity
