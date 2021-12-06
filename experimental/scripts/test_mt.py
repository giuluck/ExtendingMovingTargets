"""Moving Targets Test Script."""
import warnings

from sklearn.exceptions import ConvergenceWarning

from experimental.utils import ExperimentHandler
from moving_targets.learners import *
from src.datasets import *
from src.masters import *

if __name__ == '__main__':
    warnings.simplefilter("ignore", category=ConvergenceWarning)

    iterations = 10
    callbacks = []
    manager = RedwineManager(filepath='../../res/redwine.csv')

    ExperimentHandler(
        manager=manager,
        learner=LogisticRegression(),
        master=BalancedCounts(loss_fn='mae'),
        init_step='pretraining'
    ).experiment(
        iterations=iterations,
        num_folds=None,
        callbacks=callbacks,
        model_verbosity=True,
        fold_verbosity=False,
        plot_args={'num_columns': 3, 'figsize': (30, 20), 'tight_layout': True},
        summary_args=None
    )
