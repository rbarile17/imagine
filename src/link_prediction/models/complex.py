import torch

from pydantic import BaseModel

from torch.nn import Parameter

from .model import Model, KelpieModel

from ...dataset import KelpieDataset


class ComplExHyperParams(BaseModel):
    dimension: int
    init_scale: float


class ComplEx(Model):
    def __init__(self, dataset, hp: ComplExHyperParams, init_random=True):
        super().__init__(dataset)

        self.name = "ComplEx"
        self.num_entities = dataset.num_entities
        self.num_relations = 2 * dataset.num_relations
        self.dimension = 2 * hp.dimension
        self.real_dimension = hp.dimension
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

        return self.score_embs(lhs, rel, rhs).detach().cpu().numpy()

    def score_embs(self, lhs, rel, rhs):
        lhs = (lhs[:, : self.real_dimension], lhs[:, self.real_dimension :])
        rel = (rel[:, : self.real_dimension], rel[:, self.real_dimension :])
        rhs = (rhs[:, : self.real_dimension], rhs[:, self.real_dimension :])

        real = (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0]
        im = (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1]
        score = real + im

        return torch.sum(score, 1, keepdim=True)

    def forward(self, triples):
        lhs = self.entity_embeddings[triples[:, 0]]
        rel = self.relation_embeddings[triples[:, 1]]
        rhs = self.entity_embeddings[triples[:, 2]]

        lhs = lhs[:, : self.real_dimension], lhs[:, self.real_dimension :]
        rel = rel[:, : self.real_dimension], rel[:, self.real_dimension :]
        rhs = rhs[:, : self.real_dimension], rhs[:, self.real_dimension :]

        to_score = self.entity_embeddings
        to_score = (
            to_score[:, : self.real_dimension],
            to_score[:, self.real_dimension :],
        )

        real_product = lhs[0] * rel[0] - lhs[1] * rel[1]
        imaginary_product = lhs[0] * rel[1] + lhs[1] * rel[0]
        score_real = real_product @ to_score[0].transpose(0, 1)
        score_imaginary = imaginary_product @ to_score[1].transpose(0, 1)
        score = score_real + score_imaginary

        reg_matrices = (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2),
        )

        return (score, reg_matrices)

    def all_scores(self, triples):

        lhs = self.entity_embeddings[triples[:, 0]]
        rel = self.relation_embeddings[triples[:, 1]]

        lhs = (lhs[:, : self.real_dimension], lhs[:, self.real_dimension :])
        rel = (rel[:, : self.real_dimension], rel[:, self.real_dimension :])

        real = lhs[0] * rel[0] - lhs[1] * rel[1]
        im = lhs[0] * rel[1] + lhs[1] * rel[0]

        q = torch.cat([real, im], 1)
        all_rhs = self.entity_embeddings

        all_rhs_batches = torch.split(all_rhs, 512, dim=0)

        all_scores_batches = []

        for batch in all_rhs_batches:
            batch = batch.transpose(0, 1)
            batch_scores = q @ batch
            all_scores_batches.append(batch_scores)

        out1 = torch.cat(all_scores_batches, dim=1)

        return out1

    def _get_rhs(self):
        rhs = self.entity_embeddings.transpose(0, 1)
        return rhs.detach()

    def _get_queries(self, triples):
        lhs = self.entity_embeddings[triples[:, 0]]
        rel = self.relation_embeddings[triples[:, 1]]

        lhs = (lhs[:, : self.real_dimension], lhs[:, self.real_dimension :])
        rel = (rel[:, : self.real_dimension], rel[:, self.real_dimension :])

        real = lhs[0] * rel[0] - lhs[1] * rel[1]
        im = lhs[0] * rel[1] + lhs[1] * rel[0]

        return torch.cat([real, im], 1)

    def criage_first_step(self, triples):
        return self._get_queries(triples)

    def criage_last_step(self, x: torch.Tensor, rhs: torch.Tensor):
        return x @ rhs

    def kelpie_model_class(self):
        return KelpieComplEx

    def get_hyperparams_class():
        return ComplExHyperParams


class KelpieComplEx(KelpieModel):
    def __init__(self, dataset: KelpieDataset, model: ComplEx, init_tensor):
        frozen_entity_embs = model.entity_embeddings.clone().detach()
        frozen_relation_embs = model.relation_embeddings.clone().detach()
        dimension = model.real_dimension
        init_scale = model.init_scale
        hp = ComplExHyperParams(dimension=dimension, init_scale=init_scale)
        self.model = ComplEx(dataset, hp, init_random=False)
        self.original_entity = dataset.original_entity
        self.kelpie_entity = dataset.kelpie_entity

        self.kelpie_entity_emb = Parameter(init_tensor.cuda(), requires_grad=True)
        with torch.no_grad():
            self.kelpie_entity_emb *= self.model.init_scale
        entity_embs = torch.cat([frozen_entity_embs, self.kelpie_entity_emb], 0)
        self.model.entity_embeddings = entity_embs
        self.model.relation_embeddings = frozen_relation_embs
