"""Moving Targets Test Script."""
import warnings

from moving_targets.learners import *
from moving_targets.masters.backends import GurobiBackend
from sklearn.exceptions import ConvergenceWarning

from experimental.utils import ExperimentHandler
from src.datasets import *
from src.masters import *

if __name__ == '__main__':
    warnings.simplefilter("ignore", category=ConvergenceWarning)

    tasks = {
        'iris': dict(
            manager=IrisManager(filepath='../../res/iris.csv'),
            learner=LogisticRegression(),
            master=BalancedCounts(backend=GurobiBackend(time_limit=30), loss='mse')
        ),
        'redwine': dict(
            manager=RedwineManager(filepath='../../res/redwine.csv'),
            learner=LogisticRegression(),
            master=BalancedCounts(backend=GurobiBackend(time_limit=30), loss='mse')
        ),
        'whitewine': dict(
            manager=WhitewineManager(filepath='../../res/whitewine.csv'),
            learner=LogisticRegression(),
            master=BalancedCounts(backend=GurobiBackend(time_limit=30), loss='mse')
        ),
        'shuttle': dict(
            manager=ShuttleManager(filepath='../../res/shuttle.trn'),
            learner=LogisticRegression(),
            master=BalancedCounts(backend=GurobiBackend(time_limit=30), loss='mse')
        ),
        'dota': dict(
            manager=DotaManager(filepath='../../res/dota2.csv'),
            learner=LogisticRegression(),
            master=BalancedCounts(backend=GurobiBackend(time_limit=30), loss='mse')
        ),
        'adult': dict(
            manager=AdultManager(filepath='../../res/adult.csv'),
            learner=LogisticRegression(),
            master=FairClassification(backend=GurobiBackend(time_limit=30), protected='race', loss='mse')
        ),
        'communities': dict(
            manager=CommunitiesManager(filepath='../../res/communities.csv'),
            learner=LinearRegression(),
            master=FairRegression(backend=GurobiBackend(time_limit=30), protected='race', loss='mse')
        )
    }

    iterations = 5
    callbacks = []

    ExperimentHandler(init_step='pretraining', **tasks['adult']).experiment(
        iterations=iterations,
        num_folds=None,
        callbacks=callbacks,
        model_verbosity=1,
        fold_verbosity=False,
        plot_args=dict(figsize=(20, 10)),
        summary_args=None
    )
