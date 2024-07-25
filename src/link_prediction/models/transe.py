import torch

from pydantic import BaseModel

from torch.nn import Parameter
from torch.nn.init import xavier_normal_

from .model import Model, KelpieModel
from ...dataset import Dataset, KelpieDataset


class TransEHyperParams(BaseModel):
    dimension: int
    norm: int


class TransE(Model):
    def __init__(self, dataset: Dataset, hp: TransEHyperParams, init_random=True):
        super().__init__(dataset)

        self.name = "TransE"
        self.num_entities = dataset.num_entities
        self.num_relations = 2 * dataset.num_relations
        self.dimension = hp.dimension
        self.norm = hp.norm

        if init_random:
            entity_embs = torch.rand(self.num_entities, self.dimension).cuda()
            self.entity_embeddings = Parameter(entity_embs, requires_grad=True)
            relation_embs = torch.rand(self.num_relations, self.dimension).cuda()
            self.relation_embeddings = Parameter(relation_embs, requires_grad=True)
            xavier_normal_(self.entity_embeddings)
            xavier_normal_(self.relation_embeddings)

    def is_minimizer(self):
        return True

    def score(self, triples):
        lhs = self.entity_embeddings[triples[:, 0]]
        rel = self.relation_embeddings[triples[:, 1]]
        rhs = self.entity_embeddings[triples[:, 2]]

        return self.score_embs(lhs, rel, rhs).detach().cpu().numpy()

    def score_embs(self, lhs, rel, rhs):
        return (lhs + rel - rhs).norm(p=self.norm, dim=1)

    def all_scores(self, triples):
        lhs = self.entity_embeddings[triples[:, 0]]
        rel = self.relation_embeddings[triples[:, 1]]
        all_rhs = self.entity_embeddings

        all_rhs_batches = torch.split(all_rhs, 2048, dim=0)

        all_scores_batches = []

        for batch in all_rhs_batches:
            translation = lhs + rel
            batch_scores = translation.unsqueeze(0) - batch.unsqueeze(1)
            batch_scores = batch_scores.norm(p=self.norm, dim=2)
            all_scores_batches.append(batch_scores)

        all_scores = torch.cat(all_scores_batches, dim=0)

        return all_scores.transpose(0, 1)

    def forward(self, triples):
        lhs = self.entity_embeddings[triples[:, 0]]
        rel = self.relation_embeddings[triples[:, 1]]
        rhs = self.entity_embeddings[triples[:, 2]]

        score = (lhs + rel - rhs).norm(p=self.norm, dim=1)
        reg_matrices = (lhs, rel, rhs)

        return score, reg_matrices

    def kelpie_model_class(self):
        return KelpieTransE

    def get_hyperparams_class():
        return TransEHyperParams


class KelpieTransE(KelpieModel):
    def __init__(self, dataset: KelpieDataset, model: TransE, init_tensor):
        frozen_entity_embs = model.entity_embeddings.clone().detach()
        frozen_relation_embs = model.relation_embeddings.clone().detach()
        hp = TransEHyperParams(dimension=model.dimension, norm=model.norm)
        self.model = TransE(dataset, hp, init_random=False)
        self.original_entity = dataset.original_entity
        self.kelpie_entity = dataset.kelpie_entity

        self.kelpie_entity_emb = Parameter(init_tensor.cuda(), requires_grad=True)
        with torch.no_grad():
            xavier_normal_(self.kelpie_entity_emb)

        entity_embs = torch.cat([frozen_entity_embs, self.kelpie_entity_emb], 0)
        self.model.entity_embeddings = entity_embs
        self.model.relation_embeddings = frozen_relation_embs
