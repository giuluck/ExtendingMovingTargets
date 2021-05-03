import os

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping
from src import restaurants
from src.models import MT, MTLearner, MLP, MTClassificationMaster
from src.restaurants import compute_monotonicities, get_augmented_data, import_extension_methods
from src.util.augmentation import get_monotonicities_list
from src.util.preprocessing import Scaler
# noinspection PyUnresolvedReferences
from moving_targets.callbacks import FileLogger, ConsoleLogger
from moving_targets.metrics import AUC, MonotonicViolation, CrossEntropy
# noinspection PyUnresolvedReferences
from tests.util.callbacks import DistanceAnalysis, BoundsAnalysis
from tests.util.experiments import setup


def neural_model(h_units, scaler):
    model = MLP(output_act='sigmoid', h_units=h_units, scaler=scaler)
    model.compile(optimizer='adam', loss='mse')
    return model


if __name__ == '__main__':
    # set random seeds
    setup()
    import_extension_methods()

    # load and prepare data
    val_data = restaurants.load_data()
    x_aug, y_aug, full_aug, aug_scaler = get_augmented_data(val_data['train'][0], val_data['train'][1])
    scalers = (aug_scaler, Scaler.get_default(1))
    aug_mask = np.isnan(y_aug['clicked'])
    mono = get_monotonicities_list(full_aug, compute_monotonicities, 'clicked', 'group')

    # similar to the default behaviour of the scikit MLP (tol = 1e-4, n_iter_no_change = 10, max_iter = 200)
    es = EarlyStopping(monitor='loss', patience=10, min_delta=1e-4)
    learner = MTLearner(lambda: neural_model([16, 8, 8], aug_scaler), epochs=200, callbacks=[es], verbose=False)
    master = MTClassificationMaster(monotonicities=mono, augmented_mask=aug_mask, use_prob=False, alpha=0.01,
                                    learner_y='original', learner_weights='all', learner_omega=1, master_omega=1)
    iterations = 5

    num_col = int(np.ceil(np.sqrt(iterations + 1)))
    callbacks = [
        # FileLogger('temp/log.txt', routines=['on_iteration_end']),
        # DistanceAnalysis(scalers, do_plot=False, sorting_attribute=None, file_signature='temp/distance_analysis'),
        # DistanceAnalysis(scalers, ground_only=True, num_columns=num_col, sorting_attribute=None)
    ]

    # moving targets
    mt = MT(
        learner=learner,
        master=master,
        init_step='pretraining',
        metrics=[CrossEntropy(), AUC(),
                 MonotonicViolation(monotonicities=mono, aggregation='average', name='avg. violation'),
                 MonotonicViolation(monotonicities=mono, aggregation='percentage', name='pct. violation'),
                 MonotonicViolation(monotonicities=mono, aggregation='feasible', name='is feasible')]
    )
    history = mt.fit(
        x=x_aug,
        y=y_aug['clicked'],
        iterations=iterations,
        val_data=val_data,
        callbacks=callbacks,
        verbose=1
    )

    # exit()
    history.plot(figsize=(20, 10), n_columns=4, columns=[
        'learner/loss',
        'metrics/train crossentropy',
        'metrics/train auc',
        'metrics/is feasible',
        'learner/epochs',
        'metrics/validation crossentropy',
        'metrics/validation auc',
        'metrics/pct. violation',
        'master/avg. flips',
        'metrics/test crossentropy',
        'metrics/test auc',
        'metrics/avg. violation'
    ])

    # exit()
    print('-------------------------------------------------------')
    # noinspection PyUnresolvedReferences
    mt.evaluation_summary(**val_data)
    plt.show()
