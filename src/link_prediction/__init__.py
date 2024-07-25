from .optimization import BCEOptimizer, MultiClassNLLOptimizer, PairwiseRankingOptimizer

from .models import ConvE, ComplEx, TransE

MODEL_REGISTRY = {
    "ComplEx": {"class": ComplEx, "optimizer": MultiClassNLLOptimizer},
    "TransE": {"class": TransE, "optimizer": PairwiseRankingOptimizer},
    "ConvE": {"class": ConvE, "optimizer": BCEOptimizer},
}
