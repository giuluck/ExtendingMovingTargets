"""Moving Targets Manager."""

from typing import Any, Union, List

import numpy as np

from experimental.datasets.managers import TestManager, Fold
from experimental.models.managers.model_manager import ModelManager
from src.models import MT, MTLearner


# noinspection PyMissingOrEmptyDocstring
class MTManager(ModelManager):
    def __init__(self, test_manager: TestManager, iterations: int = 10, verbose: Union[int, str] = False, **kwargs):
        super(MTManager, self).__init__(test_manager=test_manager, **kwargs)
        self.iterations = iterations
        self.verbose = verbose

    def get_folds(self, num_folds: int, extrapolation: bool, compute_monotonicities: bool = False) -> List[Fold]:
        return super(MTManager, self).get_folds(num_folds=num_folds,
                                                extrapolation=extrapolation,
                                                compute_monotonicities=compute_monotonicities)

    def fit(self, fold: Fold) -> Any:
        label = self.test_manager.dataset.y_column
        model = MT(
            learner=MTLearner(scalers=fold.scalers, **self.test_manager.learner_args),
            master=self.test_manager.master_class(monotonicities=fold.monotonicities,
                                                  augmented_mask=np.isnan(fold.y[label]),
                                                  **self.test_manager.master_args),
            init_step=self.test_manager.mt_init_step
        )
        model.fit(
            x=fold.x,
            y=fold.y[label],
            iterations=self.iterations,
            val_data=fold.validation,
            callbacks=[],
            verbose=self.verbose
        )
        return model
