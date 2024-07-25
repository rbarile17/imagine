import math
import time
import numpy as np
import torch

from collections import OrderedDict

from torch.profiler import profile, record_function, ProfilerActivity

from .engine import RelevanceEngine

from ..dataset import Dataset, KelpieDataset
from ..link_prediction import MODEL_REGISTRY
from ..link_prediction.models import Model, KelpieModel


class PostTrainingEngine(RelevanceEngine):
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def __init__(self, model: Model, dataset: Dataset, hp: dict):
        RelevanceEngine.__init__(self, model=model, dataset=dataset)

        self.hp = hp

        if isinstance(model, KelpieModel):
            raise Exception("Already a post-trainable KelpieModel.")

    def set_cache(self):
        self.base_pt_results = {}
        self.kelpie_dataset_cache_size = 20
        self.kelpie_dataset_cache = OrderedDict()

    def _get_kelpie_dataset(self, original_entity: int) -> KelpieDataset:
        if original_entity not in self.kelpie_dataset_cache:
            kelpie_dataset = KelpieDataset(dataset=self.dataset, entity=original_entity)
            self.kelpie_dataset_cache[original_entity] = kelpie_dataset
            self.kelpie_dataset_cache.move_to_end(original_entity)

            if len(self.kelpie_dataset_cache) > self.kelpie_dataset_cache_size:
                self.kelpie_dataset_cache.popitem(last=False)

        return self.kelpie_dataset_cache[original_entity]

    def compute_relevance(self, pred, triples: list):
        s, _, _ = pred

        dataset = self._get_kelpie_dataset(original_entity=s)

        kelpie_model_class = self.model.kelpie_model_class()
        init_tensor = torch.rand(1, self.model.dimension)

        # run base post-training on homologous mimic of the pred subject
        base_model = kelpie_model_class(dataset, self.model, init_tensor)
        base_pt_results = self.get_base_post_train_results(base_model, dataset, pred)

        # run actual post-training on non homologous mimic of the pred subject
        pt_model = kelpie_model_class(dataset, self.model, init_tensor)
        pt_results = self.get_post_train_results(pt_model, dataset, pred, triples)

        return pt_results, base_pt_results

    def post_train(
        self,
        model: KelpieModel,
        triples: np.array,
    ):
        model.to("cuda")

        optimizer_class = MODEL_REGISTRY[self.model.name]["optimizer"]
        optimizer_params = optimizer_class.get_hyperparams_class()(**self.hp)
        optimizer_class = optimizer_class.get_kelpie_class()
        optimizer = optimizer_class(model=model, hp=optimizer_params, verbose=False)
        optimizer.train(training_triples=np.array(triples))
        return model

    def get_base_post_train_results(
        self, model: KelpieModel, dataset: KelpieDataset, pred
    ):
        kelpie_pred = dataset.as_kelpie_triple(pred)

        if not pred in self.base_pt_results:
            triples = dataset.kelpie_training_triples
            base_pt_model = self.post_train(model=model, triples=triples)

            results = self.get_triple_results(base_pt_model, kelpie_pred)
            self.base_pt_results[pred] = results

        return self.base_pt_results[pred]

    def get_post_train_results(
        self,
        model: KelpieModel,
        dataset: KelpieDataset,
        pred: np.array,
        triples: np.array,
    ):
        raise NotImplementedError

    def get_triple_results(self, model: Model, triple):
        model.eval()
        s, p, o = triple

        with torch.no_grad():
            all_scores = model.all_scores(np.array([triple]))[0].detach()

        target_score = all_scores[o].item()
        filter_out = model.dataset.to_filter.get((s, p), [])
        if model.is_minimizer():
            all_scores[filter_out] = 1e6
            all_scores[o] = target_score
            best_score = torch.min(all_scores)
            target_rank = torch.sum(all_scores <= target_score)
        else:
            all_scores[filter_out] = -1e6
            best_score = torch.max(all_scores)
            target_rank = torch.sum(all_scores >= target_score)
            all_scores[o] = target_score

        return {
            "target_score": target_score,
            "best_score": best_score,
            "target_rank": target_rank,
        }


class NecessaryPostTrainingEngine(PostTrainingEngine):
    def __init__(self, model: Model, dataset: Dataset, hp: dict):
        super().__init__(model=model, dataset=dataset, hp=hp)

    def compute_relevance(
        self,
        pred,
        triples
    ):
        pt_results, base_pt_results = super().compute_relevance(pred, triples)
        rank_worsening = pt_results["target_rank"] - base_pt_results["target_rank"]
        score_worsening = (
            pt_results["target_score"] - base_pt_results["target_score"]
            if self.model.is_minimizer()
            else base_pt_results["target_score"] - pt_results["target_score"]
        )

        return float(rank_worsening + self.sigmoid(score_worsening))

    def get_post_train_results(
        self, model: KelpieModel, dataset: KelpieDataset, pred, triples
    ):
        kelpie_pred = dataset.as_kelpie_triple(pred)
        dataset.remove_training_triples(triples)

        model = self.post_train(model=model, triples=dataset.kelpie_training_triples)
        results = self.get_triple_results(model, kelpie_pred)
        dataset.undo_removal()

        return results


class RemoveImprovePostTrainingEngine(PostTrainingEngine):
    def __init__(self, model: Model, dataset: Dataset, hp: dict):
        super().__init__(model=model, dataset=dataset, hp=hp)

    def compute_relevance(
        self,
        pred,
        triples
    ):
        pt_results, base_pt_results = super().compute_relevance(pred, triples)
        rank_improvement = base_pt_results["target_rank"] - pt_results["target_rank"]
        score_improvement = (
            base_pt_results["target_score"] - pt_results["target_score"]
            if self.model.is_minimizer()
            else pt_results["target_score"] - base_pt_results["target_score"]
        )

        relevance = float(rank_improvement + self.sigmoid(score_improvement))
        if base_pt_results["target_rank"] > 0:
            relevance /= float(base_pt_results["target_rank"])

        return relevance

    def get_post_train_results(
        self, model: KelpieModel, dataset: KelpieDataset, pred, triples
    ):
        kelpie_pred = dataset.as_kelpie_triple(pred)
        dataset.remove_training_triples(triples)

        model = self.post_train(model=model, triples=dataset.kelpie_training_triples)
        results = self.get_triple_results(model, kelpie_pred)
        dataset.undo_removal()

        return results
    

class ImaginePostTrainingEngine(PostTrainingEngine):
    def __init__(self, model: Model, dataset: Dataset, hp: dict):
        super().__init__(model=model, dataset=dataset, hp=hp)

    def compute_relevance(
        self,
        pred,
        triples: list,
    ):
        pt_results, base_pt_results = super().compute_relevance(pred, triples)
        rank_improvement = base_pt_results["target_rank"] - pt_results["target_rank"]
        score_improvement = (
            base_pt_results["target_score"] - pt_results["target_score"]
            if self.model.is_minimizer()
            else pt_results["target_score"] - base_pt_results["target_score"]
        )

        relevance = float(rank_improvement + self.sigmoid(score_improvement))
        if base_pt_results["target_rank"] > 0:
            relevance /= float(base_pt_results["target_rank"])

        return relevance

    def get_post_train_results(
        self, model: KelpieModel, dataset: KelpieDataset, pred, triples
    ):
        kelpie_pred = dataset.as_kelpie_triple(pred)
        dataset.add_training_triples(triples)

        model = self.post_train(model=model, triples=dataset.kelpie_training_triples)
        results = self.get_triple_results(model, kelpie_pred)
        dataset.undo_addition()

        return results
    

class AddWorsenPostTrainingEngine(PostTrainingEngine):
    def __init__(self, model: Model, dataset: Dataset, hp: dict):
        super().__init__(model=model, dataset=dataset, hp=hp)

    def compute_relevance(
        self,
        pred,
        triples: list,
    ):
        pt_results, base_pt_results = super().compute_relevance(pred, triples)
        rank_worsening = pt_results["target_rank"] - base_pt_results["target_rank"]
        score_worsening = (
            pt_results["target_score"] - base_pt_results["target_score"]
            if self.model.is_minimizer()
            else base_pt_results["target_score"] - pt_results["target_score"]
        )

        return float(rank_worsening + self.sigmoid(score_worsening))

    def get_post_train_results(
        self, model: KelpieModel, dataset: KelpieDataset, pred, triples
    ):
        kelpie_pred = dataset.as_kelpie_triple(pred)
        dataset.add_training_triples(triples)

        model = self.post_train(model=model, triples=dataset.kelpie_training_triples)
        results = self.get_triple_results(model, kelpie_pred)
        dataset.undo_addition()

        return results


class SufficientPostTrainingEngine(PostTrainingEngine):
    def __init__(self, model: Model, dataset: Dataset, hp: dict):
        super().__init__(model=model, dataset=dataset, hp=hp)

    def compute_individual_relevance(self, pred, triples: list):
        pt_results, base_pt_results = super().compute_relevance(pred, triples)
        rank_improvement = base_pt_results["target_rank"] - pt_results["target_rank"]
        score_improvement = (
            base_pt_results["target_score"] - pt_results["target_score"]
            if self.model.is_minimizer()
            else pt_results["target_score"] - base_pt_results["target_score"]
        )

        relevance = float(rank_improvement + self.sigmoid(score_improvement))
        relevance /= float(base_pt_results["target_rank"])

        return relevance

    def compute_relevance(self, pred, rule):
        # start = time.time()
        pred_s, _, _ = pred
        relevances = []
        for entity in self.entities_to_convert:
            converted_rule = Dataset.replace_entity_in_triples(rule, pred_s, entity)
            converted_pred = Dataset.replace_entity_in_triple(pred, pred_s, entity)
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            individual_relevance = self.compute_individual_relevance(converted_pred, converted_rule)

            relevances.append(individual_relevance)

        # print(f"Time: {time.time() - start}")
        return sum(relevances) / len(relevances)

    def get_post_train_results(
        self,
        model: KelpieModel,
        dataset: KelpieDataset,
        pred,
        triples,
    ):
        kelpie_pred = dataset.as_kelpie_triple(pred)
        dataset.add_training_triples(triples)

        model = self.post_train(model=model, triples=dataset.kelpie_training_triples)
        results = self.get_triple_results(model, kelpie_pred)
        dataset.undo_addition()

        return results
