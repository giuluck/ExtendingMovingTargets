"""Moving Target's Test Script."""
import numpy as np

from experimental.utils import DatasetFactory

if __name__ == '__main__':
    iterations: int = 1
    factory, callbacks = DatasetFactory().get_dataset(
        name='cars',
        data_args=dict(full_features=False, full_grid=True),
        num_col=int(np.ceil(np.sqrt(iterations + 1))),
        callbacks=['logger', 'adjustments', 'response']
    )
    manager = factory.get_mt(
        wandb_name=None,
        mst_master_kind='regression',
        lrn_loss='mse',
        lrn_epochs=0,
        lrn_warm_start=False,
        lrn_verbose=True,
        mst_backend='cvxpy',
        mst_loss_fn='mae',
        mst_alpha=1.0,
        mst_master_omega=1.0,
        mst_learner_omega=1.0,
        mst_learner_weights='all',
        mst_time_limit=None,
        mst_custom_args=dict(verbose=True, solver='GUROBI')
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
    manager.experiment(iterations=iterations, callbacks=callbacks, plot_args=None, summary_args={},
                       num_folds=1, fold_verbosity=False, model_verbosity=1)
