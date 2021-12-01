"""Moving Targets Test Script."""
import warnings

from sklearn.exceptions import ConvergenceWarning

from experimental.utils import ExperimentHandler
from experimental.utils.configuration import get_manager

if __name__ == '__main__':
    warnings.simplefilter("ignore", category=ConvergenceWarning)

    iterations = 15
    callbacks = []
    manager = get_manager(dataset='redwine')

    ExperimentHandler(
        manager=manager,
        alpha=1.0,
        beta=None,
        loss_fn='mse'
    ).experiment(
        iterations=iterations,
        num_folds=None,
        callbacks=callbacks,
        model_verbosity=True,
        fold_verbosity=False,
        plot_args={},
        summary_args=None
    )
