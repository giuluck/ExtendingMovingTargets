"""Moving Targets Test Script."""
import warnings

from sklearn.exceptions import ConvergenceWarning

from experimental.utils.experiment_factory import ExperimentFactory

if __name__ == '__main__':
    warnings.simplefilter("ignore", category=ConvergenceWarning)

    iterations: int = 1
    factory, callbacks = ExperimentFactory().dota(callbacks=['logger'])
    manager = factory.experiment(iterations=iterations, callbacks=callbacks, model_verbosity=True, plot_args={})
