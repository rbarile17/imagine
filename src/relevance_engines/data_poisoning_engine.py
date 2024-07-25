import numpy

from .engine import RelevanceEngine

from ..dataset import Dataset
from ..link_prediction.models import Model, TuckER


class DPEngine(RelevanceEngine):
    def __init__(self, model: Model, dataset: Dataset, epsilon: float):
        RelevanceEngine.__init__(self, model=model, dataset=dataset)
        self.epsilon = epsilon
        self.lambd = 1

        # cache (triple, entity) -> gradient
        self.gradients_cache = {}

    def compute_relevance(self, pred, triple):
        raise NotImplementedError

    def get_gradient(self, triple, entity: int):
        s, p, o = triple
        assert entity in ([s, o])

        if (triple, entity) in self.gradients_cache:
            return self.gradients_cache[(triple, entity)].cuda()

        entity_dimension = self.model.dimension
        relation_dimension = self.model.dimension
        if isinstance(self.model, TuckER):
            entity_dimension = self.model.entity_dimension
            relation_dimension = self.model.relation_dimension

        lhs = self.model.entity_embeddings[s]
        lhs = lhs.detach().reshape(1, entity_dimension)
        rel = self.model.relation_embeddings[p]
        rel = rel.detach().reshape(1, relation_dimension)
        rhs = self.model.entity_embeddings[o]
        rhs = rhs.detach().reshape(1, entity_dimension)

        entity_embedding = lhs if entity == s else rhs
        entity_embedding.requires_grad = True

        score = self.model.score_embs(lhs, rel, rhs)
        score.backward()
        gradient = entity_embedding.grad[0]
        entity_embedding.grad = None

        self.gradients_cache[(triple, entity)] = gradient.cpu()
        return gradient.cuda()


class NecessaryDPEngine(DPEngine):
    def compute_relevance(self, pred, triple):
        pred_s, _, _ = pred
        s, _, _ = triple
        entity = pred_s

        # important in models with dropout and/or batch normalization
        self.model.eval()

        # compute the gradient of the triple to explain with respect to the embedding of the perspective entity
        # the gradient points towards the direction that increases the score
        gradient = self.get_gradient(triple=pred, entity=entity)
        gradient = gradient.detach()

        entity_embedding = self.model.entity_embeddings[entity].detach()
        # move the embedding of entity in the direction that worsens the score
        if self.model.is_minimizer():
            perturbed_entity_embedding = entity_embedding + self.epsilon * gradient
        else:
            perturbed_entity_embedding = entity_embedding - self.epsilon * gradient

        # get the original and perturbed embeddings
        triples_numpy = numpy.array([triple, triple])
        lhs = self.model.entity_embeddings[triples_numpy[:, 0]]
        rel = self.model.relation_embeddings[triples_numpy[:, 1]]
        rhs = self.model.entity_embeddings[triples_numpy[:, 2]]
        if s == entity:
            lhs[1] = perturbed_entity_embedding
        else:
            rhs[1] = perturbed_entity_embedding

        # compute the original score and the perturbed score
        scores = self.model.score_embs(lhs, rel, rhs)
        scores = scores.detach().cpu().numpy()
        original_triple_score, perturbed_triple_score = (scores[0], scores[1])

        if self.model.is_minimizer():
            relevance = -original_triple_score + self.lambd * perturbed_triple_score
        else:
            relevance = original_triple_score - self.lambd * perturbed_triple_score

        return relevance.item()


class SufficientDPEngine(DPEngine):
    def compute_individual_relevance(self, pred, triple):
        pred_s, _, _ = pred
        s, _, _ = triple
        entity = pred_s

        # important in models with dropout and/or batch normalization
        self.model.eval()

        # compute the gradient of the triple to explain with respect to the embedding of the perspective entity
        # the gradient points towards the direction that increases the score
        gradient = self.get_gradient(triple=pred, entity=entity)
        gradient = gradient.detach()

        entity_embedding = self.model.entity_embeddings[entity].detach()
        # move the embedding of entity in the direction that worsens the score
        if self.model.is_minimizer():
            perturbed_entity_embedding = entity_embedding - self.epsilon * gradient
        else:
            perturbed_entity_embedding = entity_embedding + self.epsilon * gradient

        # get the original and perturbed embeddings
        triples_numpy = numpy.array([triple, triple])
        lhs = self.model.entity_embeddings[triples_numpy[:, 0]]
        rel = self.model.relation_embeddings[triples_numpy[:, 1]]
        rhs = self.model.entity_embeddings[triples_numpy[:, 2]]
        if s == entity:
            lhs[1] = perturbed_entity_embedding
        else:
            rhs[1] = perturbed_entity_embedding

        # compute the original score and the perturbed score
        scores = self.model.score_embs(lhs, rel, rhs)
        scores = scores.detach().cpu().numpy()
        original_triple_score, perturbed_triple_score = (scores[0], scores[1])

        if self.model.is_minimizer():
            relevance = original_triple_score - self.lambd * perturbed_triple_score
        else:
            relevance = -original_triple_score + self.lambd * perturbed_triple_score

        return relevance.item()

    def compute_relevance(self, pred, triple):
        pred_s, _, _ = pred
        relevances = []

        for entity in self.entities_to_convert:
            converted_triple = Dataset.replace_entity_in_triple(triple, pred_s, entity)
            converted_pred = Dataset.replace_entity_in_triple(pred, pred_s, entity)

            relevance = self.compute_individual_relevance(converted_pred, converted_triple)

            relevances.append(relevance)

        return sum(relevances) / len(relevances)
