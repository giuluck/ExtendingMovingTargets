"""Moving Targets Test Script."""
import warnings

from sklearn.exceptions import ConvergenceWarning

from experimental.utils import ExperimentHandler
from moving_targets.learners import *
from src.datasets import *
from src.masters import *

if __name__ == '__main__':
    warnings.simplefilter("ignore", category=ConvergenceWarning)

    tasks = {
        'redwine': dict(
            manager=RedwineManager(filepath='../../res/redwine.csv'),
            learner=LogisticRegression(),
            master=BalancedCounts(loss_fn='mse')
        ),
        'adult': dict(
            manager=AdultManager(filepath='../../res/adult.csv'),  # test_size=0.999),
            learner=LogisticRegression(),
            master=FairClassification(protected='race', loss_fn='ce')
        ),
        'communities': dict(
            manager=CommunitiesManager(filepath='../../res/communities.csv'),
            learner=LinearRegression(),
            master=FairRegression(protected='race', loss_fn='mse')
        )
    }

    iterations = 5
    callbacks = []

    ExperimentHandler(init_step='pretraining', **tasks['adult']).experiment(
        iterations=iterations,
        num_folds=None,
        callbacks=callbacks,
        model_verbosity=False,
        fold_verbosity=False,
        plot_args={'num_columns': 3, 'figsize': (15, 10), 'tight_layout': True},
        summary_args=None
    )
