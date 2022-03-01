import warnings

from sklearn.exceptions import ConvergenceWarning

from src.experiments import Handler

if __name__ == '__main__':
    warnings.simplefilter("ignore", category=ConvergenceWarning)

    Handler(dataset='cars', loss='mse', degree=2).experiment(
        iterations=10,
        num_folds=None,
        callbacks=[],
        model_verbosity=1,
        fold_verbosity=False,
        plot_history=False,
        plot_summary=True
    )
