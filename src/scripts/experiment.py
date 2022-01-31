import warnings

from sklearn.exceptions import ConvergenceWarning

from src.experiments import Handler
from src.util.callbacks import AnalysisCallback

if __name__ == '__main__':
    warnings.simplefilter("ignore", category=ConvergenceWarning)

    Handler(dataset='synthetic').experiment(
        iterations=2,
        num_folds=None,
        callbacks=[AnalysisCallback(file_signature='../../temp/analysis', num_columns=1)],
        model_verbosity=True,
        fold_verbosity=False,
        plot_history=True
    )
