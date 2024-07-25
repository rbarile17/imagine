import numpy

from collections import defaultdict

from .engine import RelevanceEngine

from ..dataset import Dataset
from ..link_prediction.models import Model, ComplEx, ConvE, DistMult


class CriageEngine(RelevanceEngine):
    def __init__(self, model: Model, dataset: Dataset):
        RelevanceEngine.__init__(self, model=model, dataset=dataset)

        if (
            not isinstance(model, ComplEx)
            and not isinstance(model, ConvE)
            and not isinstance(model, DistMult)
        ):
            raise Exception("Criage does not support this model.")

        self.entity_dimension = self.model.dimension
        self.tail_to_training_triples = defaultdict(list)
        for h, r, t in dataset.training_triples:
            self.tail_to_training_triples[t].append((h, r, t))

        # caches
        self.hr_2_z = {}
        self.entity_2_hessian = {}

    def compute_relevance(self, pred, triple, perspective: str):
        self.model.eval()

        pred_s, pred_p, pred_o = pred
        perspective_entity = pred_o if perspective == "tail" else pred_s
        if perspective == "head":
            pred = (pred_o, pred_p, pred_s)
        z_pred = self.get_z(pred)
        z_triple = self.get_z(triple)
        hessian_matrix = self.get_hessian(
            entity=perspective_entity,
            triples=self.tail_to_training_triples.get(perspective_entity, []),
        )

        score_variation = self.estimate_score_variation(
            z_pred=z_pred,
            z_triple=z_triple,
            entity_id=perspective_entity,
            entity_hessian_matrix=hessian_matrix,
        )

        return score_variation

    def estimate_score_variation(
        self, z_pred, z_triple, entity_id, entity_hessian_matrix
    ):
        raise NotImplementedError

    def get_z(self, triple):
        s, p, _ = triple
        hr = (s, p)
        if hr not in self.hr_2_z:
            self.hr_2_z[hr] = self.model.criage_first_step(numpy.array([triple]))
        return self.hr_2_z[hr]

    def get_hessian(self, entity, triples):
        if entity not in self.entity_2_hessian:
            self.entity_2_hessian[entity] = self.compute_hessian(entity, triples)
        return self.entity_2_hessian[entity]

    def sigmoid(self, x):
        return 1 / (1 + numpy.exp(-x))

    def compute_hessian(self, entity, triples):
        """
        This method computes the Hessian matrix for an entity, based on the triples that feature that entity as tail

        :param entity: the id of the entity to compute the hessian for
        :param triples_featuring_entity_as_tail: the list of triples featuring the passed entity as tail
        :return: the computed Hessian matrix, that is, an entity_dimension x entity_dimension matrix
        """

        entity_embeddings = self.model.entity_embeddings
        relation_embeddings = self.model.relation_embeddings

        entity_embedding = entity_embeddings[entity].detach().cpu().numpy()

        hessian_matrix = numpy.zeros((self.entity_dimension, self.entity_dimension))

        for triple in triples:
            (s, p, _) = triple
            lhs = entity_embeddings[s].detach().cpu().numpy()
            rel = relation_embeddings[p].detach().cpu().numpy()

            lhs = numpy.reshape(lhs, (1, -1))
            rel = numpy.reshape(rel, (1, -1))

            x = numpy.multiply(lhs, rel)
            x_2 = numpy.dot(entity_embedding, numpy.transpose(x))

            sig = self.sigmoid(x_2)
            sig = sig * (1 - sig)
            hessian_matrix += sig * numpy.dot(numpy.transpose(x), x)
        return hessian_matrix


class NecessaryCriageEngine(CriageEngine):
    def estimate_score_variation(
        self, z_pred, z_triple, entity_id, entity_hessian_matrix
    ):
        entity_embedding = self.model.entity_embeddings[entity_id]
        entity_embedding = entity_embedding.detach().cpu().numpy()

        z_triple = z_triple.detach().cpu().numpy()
        z_pred = z_pred.detach().cpu().numpy()

        x_2 = numpy.dot(entity_embedding, numpy.transpose(z_triple))
        sig_tri = self.sigmoid(x_2)

        m = numpy.linalg.inv(
            entity_hessian_matrix
            + sig_tri
            * (1 - sig_tri)
            * numpy.dot(numpy.transpose(z_triple), z_triple)
        )
        relevance = numpy.dot(
            z_pred,
            numpy.transpose((1 - sig_tri) * numpy.dot(z_triple, m)),
        )

        return -relevance[0][0]


class SufficientCriageEngine(CriageEngine):
    def compute_relevance(self, pred, triple, perspective: str):
        pred_s, pred_p, pred_o = pred
        s, p, _ = triple
        relevances = []

        for entity in self.entities_to_convert:
            triple = (s, p, entity)

            if perspective == "head":
                pred = (entity, pred_p, pred_o)
            else:
                pred = (pred_s, pred_p, entity)

            relevance = super().compute_relevance(pred, triple, perspective)
            relevances.append(relevance)

        return sum(relevances) / len(relevances)

    def estimate_score_variation(
        self, z_pred, z_triple, entity_id, entity_hessian_matrix
    ):
        entity_embedding = self.model.entity_embeddings[entity_id]
        entity_embedding = entity_embedding.detach().cpu().numpy()

        z_triple = z_triple.detach().cpu().numpy()
        z_pred = z_pred.detach().cpu().numpy()
        x_2 = numpy.dot(entity_embedding, numpy.transpose(z_triple))
        sig_tri = self.sigmoid(x_2)

        m = numpy.linalg.inv(
            entity_hessian_matrix
            + sig_tri * (1 - sig_tri) * numpy.dot(numpy.transpose(z_triple), z_triple)
        )
        relevance = numpy.dot(
            z_pred,
            numpy.transpose((1 - sig_tri) * numpy.dot(z_triple, m)),
        )

        return relevance[0][0]
