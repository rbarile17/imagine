import random
import numpy
import torch

from collections import defaultdict

from tqdm import tqdm

from ..dataset import Dataset, ONE_TO_ONE, MANY_TO_ONE
from ..link_prediction.models import Model


class RelevanceEngine:
    def __init__(self, model: Model, dataset: Dataset):
        self.model = model
        self.dataset = dataset

        self.o_to_training_triples = defaultdict(list)
        for h, r, t in dataset.training_triples:
            self.o_to_training_triples[t].append((h, r, t))

    def select_entities_to_convert(self, pred, k: int, degree_cap=None, criage=False):
        """
        Extract k entities to replace the head in the passed triple.

        The purpose of such entities is to allow the engine to identify sufficient rules to explain the triple.
        To do so, the engine replaces the head in the triple with the extracted entities,
        and the engine analyzes the effect of adding/removing fact featuring those entities.

        The whole system works assuming that the extracted entities, when put in the passed triple,
        result in *wrong* facts, that are not predicted as true by the model;
        the purpose of the engine is identify the minimal combination of facts to added to those entities
        in order to "fool" the model and make it predict those "wrong" facts as true.

        As a consequence each extracted entities will adhere to the following criteria:
            - must be different from the head of the prediciton
            - must be seen in training
            - must form a "true" fact when put in the triple
              E.g., Michelle can not be a good replacement for Barack in <Barack, parent, Natasha>
              if <Michelle, parent, Natasha> is already present in the dataset (in train, valid or test set)
            - if the relation in the triple has *_TO_ONE type, the extracted entities must not already have
              a known tail for the relation under analysis in the training set
              e.g., when explaining <Barack, nationality, USA> we can not use entity "Vladimir" to replace Barack
              if <Vladimir, nationality, Russia> is in the dataset
            - must not form a "true" fact when put in the triple.
              e.g., Michelle can not be a good replacement for Barack in <Barack, parent, Natasha>
              if <Michelle, parent, Natasha> is in the dataset
            - must not form a fact that is predicted by the model.
              (we want current_other_entities that, without additions, do not predict the target entity with rank 1)
              (e.g. if we are explaining <Barack, nationality, USA>, George is an acceptable "convertible entity"
              only if <George, nationality, ?> does not already rank USA in 1st position!)

        :param model: the model the prediction of which must be explained
        :param dataset: the dataset on which the passed model has been trained
        :param triple: the triple that the engine is trying to explain
        :param k: the number of entities to extract
        :param degree_cap:
        :return:
        """

        # important in models with dropout and/or batch normalization
        self.model.eval()

        s, p, o = pred
        perspective_entity = s

        overall_entities = []

        entities = []
        for entity in range(self.dataset.num_entities):
            if entity == perspective_entity:
                continue
            if self.dataset.entity_to_degree[entity] < 1:
                continue
            if degree_cap and self.dataset.entity_to_degree[entity] > degree_cap:
                continue

            if criage and entity not in self.o_to_training_triples:
                continue

            if (entity, p) in self.dataset.to_filter:
                if self.dataset.relation_to_type[p] in [ONE_TO_ONE, MANY_TO_ONE]:
                    continue

                if o in self.dataset.to_filter[(entity, p)]:
                    continue

            entities.append(entity)

        if len(entities) == 0:
            return []
        triples = [(entity, p, o) for entity in entities]

        batch_size = 256
        batch_start = 0
        with tqdm(total=len(triples), unit="ex", leave=False) as bar:
            while batch_start < len(triples):
                batch_end = min(len(triples), batch_start + batch_size)
                batch = triples[batch_start:batch_end]
                batch_scores = self.model.all_scores(triples=numpy.array(batch))
                batch_scores = batch_scores.detach().cpu().numpy()

                j = 0
                for i in range(batch_start, batch_end):
                    entity = entities[i]
                    s, p, o = triples[i]
                    triple_scores = batch_scores[j]

                    filter_out = self.dataset.to_filter.get((s, p), [])
                    filter_out = torch.LongTensor(filter_out)

                    target_score = triple_scores[o]
                    if self.model.is_minimizer():
                        triple_scores[filter_out] = 1e6
                        if 1e6 > target_score > numpy.min(triple_scores):
                            overall_entities.append(entity)
                    else:
                        triple_scores[filter_out] = -1e6
                        if -1e6 < target_score < numpy.max(triple_scores):
                            overall_entities.append(entity)
                    j += 1

                batch_start += batch_size
                bar.update(batch_size)
        entities = random.sample(overall_entities, k=min(k, len(overall_entities)))
        self.entities_to_convert = entities
