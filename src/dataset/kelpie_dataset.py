import copy

import numpy as np

from collections import defaultdict

from .dataset import Dataset


class KelpieDataset:
    def __init__(self, dataset: Dataset, entity):
        self.dataset = dataset
        self.to_filter = copy.deepcopy(dataset.to_filter)
        self.train_to_filter = copy.deepcopy(dataset.train_to_filter)
        self.entity_to_id = copy.deepcopy(dataset.entity_to_id)
        self.id_to_entity = copy.deepcopy(dataset.id_to_entity)
        self.relation_to_id = copy.deepcopy(dataset.relation_to_id)
        self.id_to_relation = copy.deepcopy(dataset.id_to_relation)

        self.num_entities = dataset.num_entities + 1
        self.num_relations = dataset.num_relations

        self.original_entity = entity
        self.original_entity_label = self.id_to_entity[entity]
        self.kelpie_entity = self.num_entities - 1
        self.kelpie_entity_label = "kelpie_" + self.original_entity_label
        self.entity_to_id[self.kelpie_entity_label] = self.kelpie_entity
        self.id_to_entity[self.kelpie_entity] = self.kelpie_entity_label

        self.kelpie_training_triples = Dataset.replace_entity_in_triples(
            dataset.entity_to_training_triples[self.original_entity],
            self.original_entity,
            self.kelpie_entity,
        )
        self.kelpie_validation_triples = Dataset.replace_entity_in_triples(
            dataset.entity_to_validation_triples[self.original_entity],
            self.original_entity,
            self.kelpie_entity,
        )
        self.kelpie_testing_triples = Dataset.replace_entity_in_triples(
            dataset.entity_to_testing_triples[self.original_entity],
            self.original_entity,
            self.kelpie_entity,
        )
        self.kelpie_training_triples_copy = copy.deepcopy(self.kelpie_training_triples)
        self.kelpie_validation_triples_copy = copy.deepcopy(
            self.kelpie_validation_triples
        )
        self.kelpie_testing_triples_copy = copy.deepcopy(self.kelpie_testing_triples)

        triples_to_stack = [self.kelpie_training_triples]
        if len(self.kelpie_validation_triples) > 0:
            triples_to_stack.append(self.kelpie_validation_triples)
        if len(self.kelpie_testing_triples) > 0:
            triples_to_stack.append(self.kelpie_testing_triples)
        all_kelpie_triples = np.vstack(triples_to_stack)
        for s, p, o in self.kelpie_training_triples:
            self.train_to_filter[(s, p)].append(o)
            self.train_to_filter[(o, p + self.num_relations)].append(s)
        for s, p, o in all_kelpie_triples:
            self.to_filter[(s, p)].append(o)
            self.to_filter[(o, p + self.num_relations)].append(s)

        self.kelpie_triple_to_index = {}
        for i in range(len(self.kelpie_training_triples)):
            s, p, o = self.kelpie_training_triples[i]
            self.kelpie_triple_to_index[(s, p, o)] = i

        self.last_added_triples = []
        self.last_added_triples_number = 0
        self.last_filter_additions = defaultdict(list)
        self.last_added_kelpie_triples = []

        self.last_removed_triples = []
        self.last_removed_triples_number = 0
        self.last_filter_removals = defaultdict(list)
        self.last_removed_kelpie_triples = []

    def as_kelpie_triple(self, original_triple):
        if not self.original_entity in original_triple:
            raise Exception(
                f"Could not find the original entity {str(self.original_entity)} "
                f"in the passed triple {str(original_triple)}"
            )

        return Dataset.replace_entity_in_triple(
            triple=original_triple,
            old_entity=self.original_entity,
            new_entity=self.kelpie_entity,
        )

    def add_training_triples(self, triples_to_add):
        for s, _, o in triples_to_add:
            assert self.original_entity == s or self.original_entity == o

        self.last_added_triples = triples_to_add
        self.last_added_triples_number = len(triples_to_add)
        self.last_filter_additions = defaultdict(list)
        self.last_added_kelpie_triples = []

        kelpie_triples_to_add = Dataset.replace_entity_in_triples(
            triples_to_add,
            self.original_entity,
            self.kelpie_entity
        )
        for s, rel, o in kelpie_triples_to_add:
            self.to_filter[(s, rel)].append(o)
            self.to_filter[(o, rel + self.num_relations)].append(s)
            self.train_to_filter[(s, rel)].append(o)
            self.train_to_filter[(o, rel + self.num_relations)].append(s)

            self.last_added_kelpie_triples.append((s, rel, o))
            self.last_filter_additions[(s, rel)].append(o)
            self.last_filter_additions[(o, rel + self.num_relations)].append(s)

        self.kelpie_training_triples = np.vstack(
            (self.kelpie_training_triples, np.array(kelpie_triples_to_add))
        )

    def remove_training_triples(self, triples: np.array):
        for s, _, o in triples:
            assert self.original_entity == s or self.original_entity == o

        self.last_removed_triples = triples
        self.last_removed_triples_number = len(triples)
        self.last_filter_removals = defaultdict(list)
        self.last_removed_kelpie_triples = []

        kelpie_triples_to_remove = Dataset.replace_entity_in_triples(
            triples=triples,
            old_entity=self.original_entity,
            new_entity=self.kelpie_entity,
        )

        for s, rel, o in kelpie_triples_to_remove:
            self.to_filter[(s, rel)].remove(o)
            self.to_filter[(o, rel + self.num_relations)].remove(s)
            self.train_to_filter[(s, rel)].remove(o)
            self.train_to_filter[(o, rel + self.num_relations)].remove(s)

            self.last_removed_kelpie_triples.append((s, rel, o))
            self.last_filter_removals[(s, rel)].append(o)
            self.last_filter_removals[(o, rel + self.num_relations)].append(s)

        idxs = [self.kelpie_triple_to_index[x] for x in kelpie_triples_to_remove]
        self.kelpie_training_triples = np.delete(
            self.kelpie_training_triples, idxs, axis=0
        )

    def undo_removal(self):
        """
        This method undoes the last removal performed on this KelpieDataset
        calling its add_training_triples method.

        The purpose of undoing the removals performed on a pre-existing KelpieDataset,
        instead of creating a new KelpieDataset from scratch, is to improve efficiency.
        """
        if self.last_removed_triples_number <= 0:
            raise Exception("No removal to undo.")

        self.kelpie_training_triples = copy.deepcopy(self.kelpie_training_triples_copy)
        for key in self.last_filter_removals:
            for x in self.last_filter_removals[key]:
                self.to_filter[key].append(x)
                self.train_to_filter[key].append(x)

        self.last_removed_triples = []
        self.last_removed_triples_number = 0
        self.last_filter_removals = defaultdict(list)
        self.last_removed_kelpie_triples = []

    def undo_addition(self):
        """
        This method undoes the last addition performed on this KelpieDataset
        calling its add_training_triples method.

        The purpose of undoing the additions performed on a pre-existing KelpieDataset,
        instead of creating a new KelpieDataset from scratch, is to improve efficiency.
        """

        if self.last_added_triples_number <= 0:
            raise Exception("No addition to undo.")

        self.kelpie_training_triples = copy.deepcopy(self.kelpie_training_triples_copy)
        for key in self.last_filter_additions:
            for x in self.last_filter_additions[key]:
                self.to_filter[key].remove(x)
                self.train_to_filter[key].remove(x)

        self.last_added_triples = []
        self.last_added_triples_number = 0
        self.last_filter_additions = defaultdict(lambda: [])
        self.last_added_kelpie_triples = []

    def as_original_triple(self, kelpie_triple):
        if not self.kelpie_entity in kelpie_triple:
            raise Exception(
                f"Could not find the original entity {str(self.kelpie_entity)} "
                f"in the passed triple {str(kelpie_triple)}"
            )
        return Dataset.replace_entity_in_triple(
            triple=kelpie_triple,
            old_entity=self.kelpie_entity,
            new_entity=self.original_entity,
        )

    def invert_triples(self, triples):
        return self.dataset.invert_triples(triples)

    def printable_triple(self, triple):
        return self.dataset.printable_triple(triple)
