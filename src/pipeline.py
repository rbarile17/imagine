from .relevance_engines import PostTrainingEngine
from .fact_builder import FactBuilder


class Pipeline:
    def __init__(self, dataset, prefilter, builder):
        self.dataset = dataset
        self.prefilter = prefilter
        self.builder = builder

        self.engine = self.builder.engine
        self.model = self.engine.model

    def set(self):
        if isinstance(self.engine, PostTrainingEngine):
            self.engine.set_cache()


class ImaginePipeline(Pipeline):
    def __init__(self, dataset, prefilter, builder):
        super().__init__(dataset, prefilter, builder)
        self.fact_builder = FactBuilder(dataset)

    def explain(self, pred, prefilter_k=-1):
        super().set()

        s, _, _ = pred
        new_triples = self.fact_builder.build_triples(s, pred)

        print("\tRunning prefilter...")
        triples = self.prefilter.select_triples(
            pred=pred, k=prefilter_k, new_triples=new_triples
        )
        result = self.builder.build_explanations(pred, triples)

        return result


class NecessaryPipeline(Pipeline):
    def __init__(self, dataset, prefilter, builder):
        super().__init__(dataset, prefilter, builder)

    def explain(self, pred, prefilter_k=-1):
        super().set()
        print("\tRunning prefilter...")
        filtered_triples = self.prefilter.select_triples(pred=pred, k=prefilter_k)

        result = self.builder.build_explanations(pred, filtered_triples)

        return result


class SufficientPipeline(Pipeline):
    def __init__(self, dataset, prefilter, builder, criage):
        super().__init__(dataset, prefilter, builder)
        self.criage = criage

    def fail(self, pred):
        return {
            "triple": self.dataset.labels_triple(pred),
            "rule_to_relevance": [],
            "#relevances": 0,
            "execution_time": 0,
            "entities_to_convert": [],
        }

    def explain(self, pred, prefilter_k=50, to_convert_k=10):
        self.engine.select_entities_to_convert(
            pred, to_convert_k, 200, criage=self.criage
        )
        if self.engine.entities_to_convert == []:
            return self.fail(pred)

        super().set()
        print("\tRunning prefilter...")
        filtered_triples = self.prefilter.select_triples(pred=pred, k=prefilter_k)

        result = self.builder.build_explanations(pred, filtered_triples)
        entities_to_conv = self.engine.entities_to_convert
        entities_to_conv = [self.dataset.id_to_entity[x] for x in entities_to_conv]

        result["entities_to_convert"] = entities_to_conv

        return result
