from src import regressions as reg
# noinspection PyUnresolvedReferences
from moving_targets.callbacks import FileLogger
from moving_targets.metrics import R2, MSE, MAE
from src.models import MT
from src.regressions.model import cars_summary, synthetic_summary, puzzles_summary
from src.util.augmentation import get_monotonicities_list
# noinspection PyUnresolvedReferences
from tests.regressions.callbacks import BoundsAnalysis, CarsAdjustments, DistanceAnalysis, SyntheticAdjustments, \
    SyntheticResponse, PuzzlesResponse, ConsoleLogger
from tests.regressions.models import Learner, UnsupervisedMaster
from tests.util.experiments import setup


def retrieve(dataset, kinds, rand=None, aug=None, ground=None):
    # dataset without augmentation
    if dataset == 'cars univariate':
        dt = reg.load_cars('../../res/cars.csv')
        xag, yag = dt['train']
        if ground is not None:
            xag, yag = xag.head(ground), yag.head(ground)
        mn = get_monotonicities_list(xag, lambda s, r: reg.compute_monotonicities(s, r, -1), 'sales', 'all', 'ignore')
        return xag, yag, mn, dt, cars_summary
    # datasets with augmentation
    dt, dirs, nrs, nas, fn = None, None, None, None, None
    if dataset == 'synthetic':
        dt, dirs, nrs, nas, fn = reg.load_synthetic(), [1, 0], 0, 15, synthetic_summary
    elif dataset == 'cars':
        dt, dirs, nrs, nas, fn = reg.load_cars('../../res/cars.csv'), -1, 0, 15, cars_summary
    elif dataset == 'puzzles':
        dt, dirs, nrs, nas, fn = reg.load_puzzles('../../res/puzzles.csv'), [-1, 1, 1], 465, [3, 4, 8], puzzles_summary
    xag, yag, fag = reg.get_augmented_data(
        x=dt['train'][0],
        y=dt['train'][1],
        directions=dirs,
        num_random_samples=nrs if rand is None else rand,
        num_augmented_samples=nas if aug is None else aug,
        num_ground_samples=ground
    )
    mn = get_monotonicities_list(
        data=fag,
        kinds=kinds,
        label=yag.columns[0],
        compute_monotonicities=lambda samples, references: reg.compute_monotonicities(samples, references, dirs)
    )
    return xag, yag[yag.columns[0]], mn, dt, fn


if __name__ == '__main__':
    setup()
    x_aug, y_aug, mono, data, summary = retrieve('cars', 'group', aug=None, ground=None)
    # mono = []

    callbacks = [
        ConsoleLogger(),
        # FileLogger('temp/log.txt', routines=['on_iteration_end']),
        # ------------------------------------------------ SYNTHETIC ------------------------------------------------
        # DistanceAnalysis(data['scalers'], ground_only=True, num_columns=2, sorting_attributes='a'),
        # SyntheticAdjustments(data['scalers'], num_columns=3, sorting_attributes='a'),
        # SyntheticResponse(data['scalers'], num_columns=3, sorting_attributes='a'),
        # ------------------------------------------------    CARS   ------------------------------------------------
        # DistanceAnalysis(data['scalers'], ground_only=True, num_columns=2, sorting_attributes='price'),
        CarsAdjustments(data['scalers'], num_columns=3, sorting_attributes='price', plot_kind='scatter'),
        # ------------------------------------------------  PUZZLES  ------------------------------------------------
        # DistanceAnalysis(data['scalers'], ground_only=True, num_columns=2, sorting_attributes=None),
        # PuzzlesResponse(data['scalers'], feature='word_count', num_columns=3, sorting_attributes='word_count'),
        # PuzzlesResponse(data['scalers'], feature='star_rating', num_columns=3, sorting_attributes='star_rating'),
        # PuzzlesResponse(data['scalers'], feature='num_reviews', num_columns=3, sorting_attributes='num_reviews')
    ]

    # moving targets
    mt = MT(
        learner=Learner(backend='keras', optimizer='adam', warm_start=False, verbose=False),
        master=UnsupervisedMaster(monotonicities=mono, loss_fn='mae', alpha=1.0, beta=1.0, beta_method='none',
                                  gamma=15, min_weight=0.0, weight_method='uniform',
                                  perturbation_method='constraint', perturbation=0.03),
        init_step='pretraining',
        metrics=[MSE(), MAE(), R2()]
    )
    history = mt.fit(
        x=x_aug,
        y=y_aug,
        iterations=5,
        val_data={k: v for k, v in data.items() if k != 'scalers'},
        callbacks=callbacks,
        verbose=0
    )

    # exit()
    history.plot(figsize=(20, 10), n_columns=4, columns=[
        'learner/loss',
        'learner/epochs',
        'metrics/train_r2',
        'master/is feasible',
        'metrics/train_mse',
        'metrics/train_mae',
        'metrics/validation_r2',
        # 'master/pct. violation',
        'master/method',
        'master/adj. mse',
        'master/adj. mae',
        'metrics/test_r2',
        'master/avg. violation'
    ])

    exit()
    summary(mt, **data)
