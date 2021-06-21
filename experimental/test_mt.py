"""Moving Target's Test Script."""
import numpy as np

from experimental.utils import DatasetFactory

if __name__ == '__main__':
    iterations: int = 20
    factory, callbacks = DatasetFactory().cars_univariate(
        data_args=dict(full_features=True, full_grid=False),
        num_col=int(np.ceil(np.sqrt(iterations + 1))),
        callbacks=['logger', 'adjustments', 'response']
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
        mst_alpha=0.01,
        mst_master_omega=1.0,
        mst_learner_omega=1.0,
        mst_learner_weights='all',
        mst_time_limit=None,
        mst_custom_args=dict(verbose=False)
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
    manager.experiment(iterations=iterations, callbacks=None, plot_args=plot_args, summary_args=dict(do_plot=False),
                       num_folds=10, fold_verbosity=False, model_verbosity=1)
