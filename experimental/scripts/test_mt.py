"""Moving Targets Test Script."""
import warnings

from sklearn.exceptions import ConvergenceWarning

from experimental.utils import ExperimentHandler
from moving_targets.learners import LogisticRegression
from src.datasets import AdultManager
from src.masters import FairClassification

if __name__ == '__main__':
    warnings.simplefilter("ignore", category=ConvergenceWarning)

    iterations = 1
    callbacks = []
    manager = AdultManager(filepath='../../res/adult.csv', test_size=0.99)

    ExperimentHandler(
        manager=manager,
        learner=LogisticRegression(),
        master=FairClassification(protected=manager.PROTECTED, loss_fn='mse', time_limit=None),
        init_step='projection'
    ).experiment(
        iterations=iterations,
        num_folds=None,
        callbacks=callbacks,
        model_verbosity=True,
        fold_verbosity=False,
        plot_args={'num_columns': 3, 'figsize': (30, 20), 'tight_layout': True},
        summary_args=None
    )
