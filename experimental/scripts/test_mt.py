"""Moving Targets Test Script."""

from experimental.utils.experiment_factory import ExperimentFactory

if __name__ == '__main__':
    iterations: int = 10
    factory, callbacks = ExperimentFactory().whitewine()
    manager = factory.experiment(iterations=iterations, callbacks=callbacks, plot_args={})
