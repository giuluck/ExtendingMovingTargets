"""Moving Targets Test Script."""
import warnings

from sklearn.exceptions import ConvergenceWarning

from src.managers import get_manager

if __name__ == '__main__':
    warnings.simplefilter("ignore", category=ConvergenceWarning)

    manager = get_manager(
        dataset='communities',
        init_step='pretraining',
        loss='mse',
        alpha=1,
        beta=1,
        adaptive=True
    ).experiment(
        iterations=5,
        num_folds=None,
        callbacks=[],
        model_verbosity=1,
        fold_verbosity=False,
        plot_args=dict(figsize=(20, 10)),
        summary_args=None
    )
