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
        'iris': dict(
            manager=IrisManager(filepath='../../res/iris.csv'),
            learner=LogisticRegression(),
            master=BalancedCounts(loss_fn='ce')
        ),
        'redwine': dict(
            manager=RedwineManager(filepath='../../res/redwine.csv'),
            learner=LogisticRegression(),
            master=BalancedCounts(loss_fn='ce')
        ),
        'whitewine': dict(
            manager=WhitewineManager(filepath='../../res/whitewine.csv'),
            learner=LogisticRegression(),
            master=BalancedCounts(loss_fn='ce')
        ),
        'shuttle': dict(
            manager=ShuttleManager(filepath='../../res/shuttle.trn'),
            learner=LogisticRegression(),
            master=BalancedCounts(loss_fn='ce')
        ),
        'dota': dict(
            manager=DotaManager(filepath='../../res/dota2.csv'),
            learner=LogisticRegression(),
            master=BalancedCounts(loss_fn='ce')
        ),
        'adult': dict(
            manager=AdultManager(filepath='../../res/adult.csv'),
            learner=LogisticRegression(),
            master=FairClassification(protected='race', loss_fn='ce')
        ),
        'communities': dict(
            manager=CommunitiesManager(filepath='../../res/communities.csv', test_size=0.99),
            learner=LinearRegression(),
            master=FairRegression(protected='race', loss_fn='mse')
        )
    }

    iterations = 15
    callbacks = []

    ExperimentHandler(init_step='pretraining', **tasks['adult']).experiment(
        iterations=iterations,
        num_folds=None,
        callbacks=callbacks,
        model_verbosity=True,
        fold_verbosity=False,
        plot_args={'num_columns': 3, 'figsize': (15, 10), 'tight_layout': True},
        summary_args=None
    )
