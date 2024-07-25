import copy
import torch

import numpy as np

from pydantic import BaseModel

from torch import nn
from torch.nn import Parameter
from torch.nn.init import xavier_normal_

from .model import Model, KelpieModel
from ...dataset import Dataset, KelpieDataset


class TuckERHyperParams(BaseModel):
    entity_dimension: int
    relation_dimension: int
    input_dropout_rate: float
    hidden_dropout_1_rate: float
    hidden_dropout_2_rate: float


class TuckER(Model):
    def __init__(self, dataset: Dataset, hp: TuckERHyperParams, init_random=True):
        Model.__init__(self, dataset)
        nn.Module.__init__(self)

        self.name = "TuckER"
        self.dataset = dataset
        self.num_entities = dataset.num_entities
        self.num_relations = 2 * dataset.num_relations
        self.entity_dimension = hp.entity_dimension
        self.relation_dimension = hp.relation_dimension
        self.input_dropout_rate = hp.input_dropout_rate
        self.hidden_dropout_1_rate = hp.hidden_dropout_1_rate
        self.hidden_dropout_2_rate = hp.hidden_dropout_2_rate

        self.input_dropout = torch.nn.Dropout(self.input_dropout_rate)
        self.hidden_dropout1 = torch.nn.Dropout(self.hidden_dropout_1_rate)
        self.hidden_dropout2 = torch.nn.Dropout(self.hidden_dropout_2_rate)
        self.batch_norm_1 = torch.nn.BatchNorm1d(self.entity_dimension)
        self.batch_norm_2 = torch.nn.BatchNorm1d(self.entity_dimension)

        if init_random:
            entity_embs = torch.rand(self.num_entities, self.entity_dimension).cuda()
            self.entity_embeddings = Parameter(entity_embs, requires_grad=True)
            rel_embs = torch.empty(self.num_relations, self.entity_dimension).cuda()
            self.relation_embeddings = Parameter(rel_embs, requires_grad=True)
            size = (hp.relation_dimension, hp.entity_dimension, hp.entity_dimension)
            core_tensor = np.random.uniform(-1, 1, size)
            core_tensor = torch.tensor(core_tensor, dtype=torch.float).cuda()
            self.core_tensor = Parameter(core_tensor, requires_grad=True)

            xavier_normal_(self.entity_embeddings)
            xavier_normal_(self.relation_embeddings)

    def is_minimizer(self):
        return False

    def score(self, triples):
        all_scores = self.all_scores(triples)

        triples_scores = []
        for i, (_, _, tail) in enumerate(triples):
            triples_scores.append(all_scores[i][tail])

        return np.array(triples_scores)

    def score_embeddings(self, lhs, rel, rhs):
        lhs = self.batch_norm_1(lhs)
        lhs = self.input_dropout(lhs)
        head_embeddings_reshaped = lhs.view(-1, 1, self.entity_dimension)

        first_mul = torch.mm(rel, self.core_tensor.view(self.relation_dimension, -1))
        first_mul = first_mul.view(-1, self.entity_dimension, self.entity_dimension)
        first_mul = self.hidden_dropout1(first_mul)

        second_mul = torch.bmm(head_embeddings_reshaped, first_mul)
        second_mul = second_mul.view(-1, self.entity_dimension)
        second_mul = self.batch_norm_2(second_mul)
        second_mul = self.hidden_dropout2(second_mul)

        result = torch.mm(second_mul, rhs.transpose(1, 0))
        scores = torch.sigmoid(result)

        output_scores = torch.diagonal(scores)
        return output_scores

    def forward(self, triples):
        return self.all_scores(triples)

    def all_scores(self, triples):
        head_indexes, relation_indexes = triples[:, 0], triples[:, 1]
        lhs = self.entity_embeddings[head_indexes]
        rel = self.relation_embeddings[relation_indexes]
        rhs = self.entity_embeddings
        lhs = self.batch_norm_1(lhs)
        lhs = self.input_dropout(lhs)
        lhs = lhs.view(-1, 1, self.entity_dimension)

        first_mul = torch.mm(rel, self.core_tensor.view(self.relation_dimension, -1))
        first_mul = first_mul.view(-1, self.entity_dimension, self.entity_dimension)
        first_mul = self.hidden_dropout1(first_mul)

        second_mul = torch.bmm(lhs, first_mul)
        second_mul = second_mul.view(-1, self.entity_dimension)
        second_mul = self.batch_norm_2(second_mul)
        second_mul = self.hidden_dropout2(second_mul)

        result = torch.mm(second_mul, rhs.transpose(1, 0))

        scores = torch.sigmoid(result)

        return scores

    def predict_tails(self, triples):
        scores, ranks = [], []

        batch_size = 128
        for i in range(0, triples.shape[0], batch_size):
            batch = triples[i : min(i + batch_size, len(triples))]
            all_scores = self.all_scores(batch)
            tails = torch.tensor(batch[:, 2]).cuda()

            for j, (head, relation, tail) in enumerate(batch):
                tails_to_filter = self.dataset.to_filter[(head, relation)]
                target_tail_score = all_scores[j, tail].item()
                scores.append(target_tail_score)

                all_scores[j, tails_to_filter] = 0.0
                all_scores[j, tail] = target_tail_score

            _, sorted_indexes = torch.sort(all_scores, dim=1, descending=True)
            sorted_indexes = sorted_indexes.cpu().numpy()

            pred_out = [row for row in sorted_indexes]
            for row in range(batch.shape[0]):
                rank = np.where(sorted_indexes[row] == tails[row].item())[0][0]
                ranks.append(rank + 1)

        return scores, ranks, pred_out

    def kelpie_model_class(self):
        return KelpieTuckER


class KelpieTuckER(KelpieModel):
    def __init__(self, dataset: KelpieDataset, model: TuckER, init_tensor):
        hp = TuckERHyperParams(
            entity_dimension=model.entity_dimension,
            relation_dimension=model.relation_dimension,
            input_dropout_rate=model.input_dropout_rate,
            hidden_dropout_1_rate=model.hidden_dropout_1_rate,
            hidden_dropout_2_rate=model.hidden_dropout_2_rate,
        )
        self.model = TuckER(dataset, hp, init_random=False)

        self.model = model
        self.original_entity = dataset.original_entity
        self.kelpie_entity = dataset.kelpie_entity

        frozen_entity_embeddings = model.entity_embeddings.clone().detach()
        frozen_relation_embeddings = model.relation_embeddings.clone().detach()
        frozen_core = model.core_tensor.clone().detach()
        self.kelpie_entity_emb = Parameter(init_tensor.cuda(), requires_grad=True)
        entity_embs = torch.cat([frozen_entity_embeddings, self.kelpie_entity_emb], 0)
        self.model.entity_embeddings = entity_embs
        self.model.relation_embeddings = frozen_relation_embeddings
        self.model.core_tensor = frozen_core

        self.batch_norm_1 = copy.deepcopy(self.model.batch_norm_1)
        self.batch_norm_2 = copy.deepcopy(self.model.batch_norm_2)
        self.batch_norm_1.eval()
        self.batch_norm_2.eval()
