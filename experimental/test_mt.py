"""Moving Target's Test Script."""
import numpy as np

from experimental.utils import DatasetFactory

if __name__ == '__main__':
    iterations: int = 1
    factory, callbacks = DatasetFactory().cars(
        data_args=dict(full_features=False, full_grid=False),
        num_col=int(np.ceil(np.sqrt(iterations + 1))),
        callbacks=['logger', 'distance', 'adjustments', 'response'],
    )
    manager = factory.get_mt(
        wandb_name=None,
        lrn_loss='mse',
        lrn_epochs=200,
        lrn_warm_start=False,
        lrn_verbose=False,
        mst_master_kind='regression',
        mst_backend='gurobi',
        mst_loss_fn='mse',
        mst_alpha=1.0,
        mst_learner_omega=1.0,
        mst_master_omega=1.0,
        mst_learner_weights='infeasible'
    )
    plot_args = dict(columns=[
        'learner/loss',
        'metrics/train loss',
        'metrics/train metric',
        'metrics/is feasible',
        'learner/epochs',
        'metrics/validation loss',
        'metrics/validation metric',
        'metrics/pct. violation',
        'master/adj. mse',
        'metrics/test loss',
        'metrics/test metric',
        'metrics/avg. violation'
    ])
    manager.experiment(iterations=iterations, callbacks=callbacks, plot_args=None,
                       summary_args=None, num_folds=None, fold_verbosity=False, model_verbosity=1)
