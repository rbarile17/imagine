import numpy as np
import torch

from pydantic import BaseModel

from torch.nn import Parameter

from .model import Model, KelpieModel
from ...dataset import Dataset, KelpieDataset


class DistMultHyperParams(BaseModel):
    dimension: int
    init_scale: float


class DistMult(Model):
    def __init__(self, dataset: Dataset, hp: DistMultHyperParams, init_random=True):
        Model.__init__(self, dataset)

        self.name = "DistMult"
        self.dataset = dataset
        self.num_entities = dataset.num_entities
        self.num_relations = 2 * dataset.num_relations
        self.dimension = hp.dimension
        self.init_scale = hp.init_scale

        if init_random:
            entity_embeddings = torch.rand(self.num_entities, self.dimension).cuda()
            self.entity_embeddings = Parameter(entity_embeddings, requires_grad=True)
            rel_embeddings = torch.rand(self.num_relations, self.dimension).cuda()
            self.relation_embeddings = Parameter(rel_embeddings, requires_grad=True)

            with torch.no_grad():
                self.entity_embeddings *= self.init_scale
                self.relation_embeddings *= self.init_scale

    def is_minimizer(self):
        return False

    def score(self, triples):
        lhs = self.entity_embeddings[triples[:, 0]]
        rel = self.relation_embeddings[triples[:, 1]]
        rhs = self.entity_embeddings[triples[:, 2]]

        return self.score_embeddings(lhs, rel, rhs).detach().cpu().numpy()

    def score_embeddings(self, lhs, rel, rhs):
        return torch.sum((lhs * rel * rhs), 1, keepdim=True)

    def all_scores(self, triples):
        lhs = self.entity_embeddings[triples[:, 0]]
        rel = self.relation_embeddings[triples[:, 1]]
        to_score = self.entity_embeddings

        return (lhs * rel) @ to_score.transpose(0, 1)

    def forward(self, triples):
        lhs = self.entity_embeddings[triples[:, 0]]
        rel = self.relation_embeddings[triples[:, 1]]
        rhs = self.entity_embeddings[triples[:, 2]]

        to_score_embeddings = self.entity_embeddings

        score = (lhs * rel) @ to_score_embeddings.transpose(0, 1)
        reg = (torch.sqrt(lhs**2), torch.sqrt(rel**2), torch.sqrt(rhs**2))

        return score, reg

    def kelpie_model_class(self):
        return KelpieDistMult


class KelpieDistMult(KelpieModel):
    def __init__(self, dataset: KelpieDataset, model: DistMult, init_tensor=None):
        hp = DistMultHyperParams(dimension=model.dimension, init_scale=model.init_scale)
        self.model = DistMult(self, dataset=dataset, hp=hp, init_random=False)

        self.original_entity = dataset.original_entity
        self.kelpie_entity = dataset.kelpie_entity

        frozen_entity_embs = model.entity_embeddings.clone().detach()
        frozen_relation_embs = model.relation_embeddings.clone().detach()

        if init_tensor is None:
            init_tensor = torch.rand(1, self.dimension)

        self.kelpie_entity_embedding = Parameter(init_tensor.cuda(), requires_grad=True)
        with torch.no_grad():
            self.kelpie_entity_embedding *= self.init_scale

        entity_embs = torch.cat([frozen_entity_embs, self.kelpie_entity_embedding], 0)
        self.model.entity_embeddings = entity_embs
        self.model.relation_embeddings = frozen_relation_embs
