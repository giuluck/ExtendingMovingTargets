import warnings

from sklearn.exceptions import ConvergenceWarning

from src.experiments import Handler

if __name__ == '__main__':
    warnings.simplefilter("ignore", category=ConvergenceWarning)

    Handler(dataset='restaurants').experiment(
        iterations=0,
        num_folds=None,
        callbacks=None,
        model_verbosity=True,
        fold_verbosity=False,
        plot_history=True,
        plot_summary=True
    )
