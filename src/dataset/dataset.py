import torch
import numpy as np
import pandas as pd
import networkx as nx

from ast import literal_eval
from collections import defaultdict

from pykeen.datasets import get_dataset

from .names import ONE_TO_ONE, ONE_TO_MANY, MANY_TO_ONE, MANY_TO_MANY

from .. import DATA_PATH
from .. import DB100K_PATH, DB100K_REASONED_PATH
from .. import DB50K_PATH, DB50K_REASONED_PATH
from .. import YAGO4_20_PATH, YAGO4_20_REASONED_PATH


class Dataset:
    def __init__(self, dataset: str):
        self.name = dataset
        if dataset == "YAGO4-20":
            self.dataset = get_dataset(
                training=YAGO4_20_PATH / "train.txt",
                testing=YAGO4_20_PATH / "test.txt",
                validation=YAGO4_20_PATH / "valid.txt",
            )

            e_sem = pd.read_csv(
                YAGO4_20_PATH / "entities.csv",
                converters={"classes": literal_eval},
            )
            e_sem["entity"] = e_sem["entity"].map(self.entity_to_id.get)
            e_sem["classes_str"] = e_sem["classes"].map(", ".join)

            e_sem_impl = pd.read_csv(
                YAGO4_20_REASONED_PATH / "entities.csv",
                converters={"classes": literal_eval},
            )
            e_sem_impl["entity"] = e_sem_impl["entity"].map(self.entity_to_id.get)
            e_sem_impl["classes_str"] = e_sem_impl["classes"].map(", ".join)

            self.entities_semantic = e_sem
            self.entities_semantic_impl = e_sem_impl
        elif dataset == "DB100K":
            self.dataset = get_dataset(
                training=DB100K_PATH / "train.txt",
                testing=DB100K_PATH / "test.txt",
                validation=DB100K_PATH / "valid.txt",
            )

            e_sem = pd.read_csv(
                DB100K_PATH / "entities.csv",
                converters={"classes": literal_eval},
            )
            e_sem["entity"] = e_sem["entity"].map(self.entity_to_id.get)
            e_sem["classes_str"] = e_sem["classes"].map(", ".join)

            e_sem_impl = pd.read_csv(
                DB100K_REASONED_PATH / "entities.csv",
                converters={"classes": literal_eval},
            )
            e_sem_impl["entity"] = e_sem_impl["entity"].map(self.entity_to_id.get)
            e_sem_impl["classes"] = e_sem_impl["classes"].map(sorted)
            e_sem_impl["classes_str"] = e_sem_impl["classes"].map(", ".join)

            self.entities_semantic = e_sem
            self.entities_semantic_impl = e_sem_impl
        elif dataset == "DB50K":
            self.dataset = get_dataset(
                training=DB50K_PATH / "train.txt",
                testing=DB50K_PATH / "test.txt",
                validation=DB50K_PATH / "valid.txt",
            )

            e_sem = pd.read_csv(
                DB50K_PATH / "entities.csv",
                converters={"classes": literal_eval},
            )
            e_sem["entity"] = e_sem["entity"].map(self.entity_to_id.get)
            e_sem["classes_str"] = e_sem["classes"].map(", ".join)

            e_sem_impl = pd.read_csv(
                DB50K_REASONED_PATH / "entities.csv",
                converters={"classes": literal_eval},
            )
            e_sem_impl["entity"] = e_sem_impl["entity"].map(self.entity_to_id.get)
            e_sem_impl["classes_str"] = e_sem_impl["classes"].map(", ".join)

            self.entities_semantic = e_sem
            self.entities_semantic_impl = e_sem_impl

        self.id_to_entity = {v: k for k, v in self.dataset.entity_to_id.items()}
        self.id_to_relation = {v: k for k, v in self.dataset.relation_to_id.items()}

        self.entity_to_training_triples = defaultdict(list)
        self.entity_to_validation_triples = defaultdict(list)
        self.entity_to_testing_triples = defaultdict(list)
        for s, p, o in self.training_triples:
            self.entity_to_training_triples[s].append((s, p, o))
            self.entity_to_training_triples[o].append((s, p, o))
        for s, p, o in self.validation_triples:
            self.entity_to_validation_triples[s].append((s, p, o))
            self.entity_to_validation_triples[o].append((s, p, o))
        for s, p, o in self.testing_triples:
            self.entity_to_testing_triples[s].append((s, p, o))
            self.entity_to_testing_triples[o].append((s, p, o))

        for entity in self.entity_to_training_triples:
            self.entity_to_training_triples[entity] = list(
                set(self.entity_to_training_triples[entity])
            )
        for entity in self.entity_to_validation_triples:
            self.entity_to_training_triples[entity] = list(
                set(self.entity_to_training_triples[entity])
            )
        for entity in self.entity_to_testing_triples:
            self.entity_to_training_triples[entity] = list(
                set(self.entity_to_training_triples[entity])
            )

        self.entity_to_degree = {
            e: len(triples) for e, triples in self.entity_to_training_triples.items()
        }

        self.train_to_filter = defaultdict(list)
        for s, p, o in self.training_triples:
            self.train_to_filter[(s, p)].append(o)
            self.train_to_filter[(o, p + self.num_relations)].append(s)

        self.to_filter = defaultdict(list)
        for s, p, o in self.all_triples:
            self.to_filter[(s, p)].append(o)
            self.to_filter[(o, p + self.num_relations)].append(s)

        self._compute_relation_to_type()

        self.entities = list(self.entity_to_id.keys())
        self.entity_ids = list(self.entity_to_id.values())

    @property
    def training(self):
        return self.dataset.training

    @property
    def training_triples(self):
        return self.dataset.training.mapped_triples.numpy()

    @property
    def validation(self):
        return self.dataset.validation

    @property
    def validation_triples(self):
        return self.dataset.validation.mapped_triples.numpy()

    @property
    def testing(self):
        return self.dataset.testing

    @property
    def testing_triples(self):
        return self.dataset.testing.mapped_triples.numpy()

    @property
    def all_triples(self):
        return np.vstack(
            [self.training_triples, self.validation_triples, self.testing_triples]
        )

    @property
    def num_entities(self):
        return self.dataset.num_entities

    @property
    def num_relations(self):
        return self.dataset.num_relations

    @property
    def entity_to_id(self):
        return self.dataset.entity_to_id

    @property
    def relation_to_id(self):
        return self.dataset.relation_to_id

    def __iter__(self):
        return self.dataset.__iter__()

    def labels_triple(self, ids_triple):
        s, p, o = ids_triple
        return (self.id_to_entity[s], self.id_to_relation[p], self.id_to_entity[o])

    def labels_triples(self, ids_triples):
        return [self.labels_triple(ids_triple) for ids_triple in ids_triples]

    def ids_triple(self, labels_triple):
        s, p, o = labels_triple
        return (self.entity_to_id[s], self.relation_to_id[p], self.entity_to_id[o])

    def ids_triples(self, labels_triples):
        return [self.ids_triple(labels_triple) for labels_triple in labels_triples]

    def printable_triple(self, triple):
        s, p, o = self.labels_triple(triple)
        return f"<{s}, {p}, {o}>"

    def get_related_triples(self, node, triples=None, depth=1):
        if not triples:
            triples = self.entity_to_training_triples[node]
        triples = set(triples)
        neighbors = [head if head != node else tail for head, _, tail in triples]

        if depth > 0:
            for neighbor in neighbors:
                neighbor_triples = self.get_related_triples(neighbor, depth=depth - 1)
                triples = triples.union(neighbor_triples)

        return triples

    def get_subgraph(self, node, triples=None, depth=1):
        triples = self.get_related_triples(node, triples=triples, depth=depth)
        edges = [(h, t, {"label": self.id_to_relation[r]}) for h, r, t in triples]
        graph = nx.MultiDiGraph(edges)

        labels = {node: {"label": self.id_to_entity[node]} for node in graph.nodes}
        nx.set_node_attributes(graph, labels)
        return graph

    def get_partition_sub(self, triples):
        if type(triples) == list:
            nodes = set([s for s, _, _ in triples] + [o for _, _, o in triples])
        else:
            nodes = triples.nodes
        e_sem = self.entities_semantic_impl
        e_sem = e_sem[e_sem.entity.isin(nodes)]
        node_types = e_sem.groupby("classes_str")["entity"].apply(list)
        node_types = node_types.to_dict()
        node_types = [part for part in node_types.values()]

        return node_types

    def get_partition(self):
        e_sem = self.entities_semantic_impl
        e_sem = e_sem.loc[e_sem.entity.isin(self.entity_ids)]
        node_types = e_sem.groupby("classes_str")["entity"].apply(list)
        node_types = node_types.to_dict()
        node_types = [part for part in node_types.values()]

        return node_types

    def add_training_triple(self, triple):
        s, p, o = triple

        self.dataset.training.mapped_triples = torch.cat(
            [self.dataset.training.mapped_triples, torch.tensor([[s, p, o]])],
            dim=0,
        )
        self.entity_to_training_triples[s].append(triple)
        self.entity_to_training_triples[o].append(triple)
        self.entity_to_degree[s] += 1
        self.entity_to_degree[o] += 1
        self.to_filter[(s, p)].append(o)
        self.train_to_filter[(s, p)].append(o)

    def add_training_triples(self, triples):
        for triple in triples:
            self.add_training_triple(triple)

    def remove_training_triple(self, triple):
        s, p, o = triple
        self.dataset.training.mapped_triples = self.dataset.training.mapped_triples[
            ~(
                (self.dataset.training.mapped_triples[:, 0] == s)
                & (self.dataset.training.mapped_triples[:, 1] == p)
                & (self.dataset.training.mapped_triples[:, 2] == o)
            )
        ]
        self.entity_to_training_triples[s].remove(triple)
        if s != o:
            self.entity_to_training_triples[o].remove(triple)
        self.entity_to_degree[s] -= 1
        if s != o:
            self.entity_to_degree[o] -= 1
        self.to_filter[(s, p)].remove(o)
        self.train_to_filter[(s, p)].remove(o)

    def remove_training_triples(self, triples):
        for triple in set(triples):
            self.remove_training_triple(triple)

    def _compute_relation_to_type(self):
        """
        This method computes the type of each relation in the dataset based on the self.train_to_filter structure
        (that must have been already computed and populated).
        The mappings relation - relation type are written in the self.relation_to_type dict.
        :return: None
        """
        if len(self.train_to_filter) == 0:
            raise Exception(
                "The dataset has not been loaded yet, so it is not possible to compute relation types yet."
            )

        relation_to_s_num = defaultdict(list)
        relation_to_o_num = defaultdict(list)

        for entity, relation in self.train_to_filter:
            length = len(self.to_filter[(entity, relation)])
            if relation >= self.num_relations:
                relation_to_s_num[relation - self.num_relations].append(length)
            else:
                relation_to_o_num[relation].append(length)

        self.relation_to_type = {}

        for relation in relation_to_s_num:
            average_s_per_o = np.average(relation_to_s_num[relation])
            average_o_per_s = np.average(relation_to_o_num[relation])

            if average_s_per_o > 1.2 and average_o_per_s > 1.2:
                self.relation_to_type[relation] = MANY_TO_MANY
            elif average_s_per_o > 1.2 and average_o_per_s <= 1.2:
                self.relation_to_type[relation] = MANY_TO_ONE
            elif average_s_per_o <= 1.2 and average_o_per_s > 1.2:
                self.relation_to_type[relation] = ONE_TO_MANY
            else:
                self.relation_to_type[relation] = ONE_TO_ONE

    def invert_triples(self, triples: np.array):
        """
        This method computes and returns the inverted version of the passed triples.
        :param triples: the direct triples to invert, in the form of a numpy array
        :return: the corresponding inverse triples, in the form of a numpy array
        """
        output = np.copy(triples)

        output[:, 0] = output[:, 2]
        output[:, 2] = triples[:, 0]
        output[:, 1] += self.num_relations

        return output

    @staticmethod
    def replace_entity_in_triple(triple, old_entity: int, new_entity: int):
        s, p, o = triple
        if s == old_entity:
            s = new_entity
        if o == old_entity:
            o = new_entity
        return (s, p, o)

    @staticmethod
    def replace_entity_in_triples(triples, old_entity: int, new_entity: int):
        results = []
        for s, p, o in triples:
            if s == old_entity:
                s = new_entity
            if o == old_entity:
                o = new_entity
            results.append((s, p, o))

        return results

    def printable_nple(self, nple: list):
        return " +\n\t\t".join([self.printable_triple(sample) for sample in nple])

    def load_summary(self):
        summ_path = DATA_PATH / self.name
        summary = pd.read_csv(
            summ_path / "train_summarization.txt",
            sep="\t",
            header=None,
            names=["s", "p", "o"],
            converters={"s": literal_eval, "o": literal_eval},
        )

        part_map = pd.read_csv(
            summ_path / "part_map.csv",
            sep="\t",
            header=None,
            names=["id", "qe"],
            converters={"qe": literal_eval},
        )

        self.quotient_entities = part_map.set_index("id")["qe"].to_dict()

        quotient_entity_to_triples = defaultdict(list)
        for _, row in summary.iterrows():
            quotient_entity_to_triples[row["s"]].append((row["s"], row["p"], row["o"]))
            quotient_entity_to_triples[row["o"]].append((row["s"], row["p"], row["o"]))
        self.quotient_entity_to_triples = quotient_entity_to_triples
    
    def get_quotient_entity(self, entity):
        for qe_id, qe in self.quotient_entities.items():
            if entity in qe:
                return qe_id
