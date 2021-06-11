"""Moving Target's Test Script."""
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping

from experimental.utils import DatasetFactory

if __name__ == '__main__':
    iterations: int = 0
    factory, callbacks = DatasetFactory().get_dataset(
        name='default',
        num_col=int(np.ceil(np.sqrt(iterations + 1))),
        callbacks=['logger', 'adjustments', 'response']
    )
    manager = factory.get_mt(
        wandb_name=None,
        mst_master_kind='regression',
        lrn_loss='mse',
        lrn_warm_start=False,
        lrn_verbose=False,
        mst_backend='gurobi',
        mst_loss_fn='mse',
        mst_alpha=1.0,
        mst_master_omega=1.0,
        mst_learner_omega=1.0,
        mst_learner_weights='all',
        mst_time_limit=None,
        mst_custom_args={'verbose': False}
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
    manager.experiment(iterations=iterations, callbacks=None, plot_args=None, summary_args={},
                       num_folds=1, fold_verbosity=False, model_verbosity=1)
