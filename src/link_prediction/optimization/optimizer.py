import numpy
from ..evaluation import Evaluator
from ..models import Model


class Optimizer:
    """
    The Optimizer class provides the interface that any LP Optimizer should implement.
    """

    def __init__(self, model: Model, hp: dict, verbose: bool = True):
        self.model = model
        self.dataset = self.model.dataset
        self.verbose = verbose

        # create the evaluator to use between epochs
        self.evaluator = Evaluator(self.model)

    def train(
        self,
        training_triples: numpy.array,
        save_path: str = None,
        evaluate_every: int = -1,
        valid_triples: numpy.array = None,
    ):
        pass
