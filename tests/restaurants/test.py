import os

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping
from src import restaurants
from src.models import MT, MTLearner, MTMaster
from src.restaurants import ctr_estimate, compute_monotonicities, get_augmented_data
from src.util.augmentation import get_monotonicities_list
from moving_targets.callbacks import FileLogger
from moving_targets.metrics import AUC
from tests.restaurants.experiments import get_model
from tests.util.experiments import setup


class TestMTL(MTLearner):
    def predict(self, x):
        return self.model.predict(x).reshape(-1, )


class TestMTM(MTMaster):
    def __init__(self, monotonicities, omega=1.0, **kwargs):
        super(TestMTM, self).__init__(monotonicities=monotonicities, **kwargs)
        self.omega = omega

    def build_model(self, macs, model, x, y, iteration):
        p = None if iteration == 0 and macs.init_step == 'projection' else macs.predict(x)
        variables = np.array(model.continuous_var_list(keys=len(y), lb=0.0, ub=1.0, name='y'))
        v_higher, v_lower = variables[self.higher_indices], variables[self.lower_indices]
        model.add_constraints([h >= l for h, l in zip(self.omega * v_higher, v_lower)])
        return variables, p

    def beta_step(self, macs, model, model_info, x, y, iteration):
        alpha_step = super().beta_step(macs, model, model_info, x, y, iteration)
        macs.cache.update(FEASIBLE=alpha_step)
        return alpha_step

    # noinspection DuplicatedCode
    def return_solutions(self, macs, solution, model_info, x, y, iteration):
        adj_y = super().return_solutions(macs, solution, model_info, x, y, iteration)
        pred = np.array([np.nan] * len(adj_y)) if model_info[1] is None else model_info[1]

        temp = pd.DataFrame(x[:, :2], columns=['average_rating', 'num_reviews'])
        temp['dollar_rating'] = ['D' * (dr + 1) for dr in x[:, 2:6].argmax(axis=1)]
        temp['y'] = y
        temp['ctr'] = ctr_estimate(temp['average_rating'], temp['num_reviews'], temp['dollar_rating'])
        temp['pred'] = pred
        temp['adj'] = adj_y
        temp[f'pred err'] = np.abs(pred - temp['ctr'])
        temp[f'adj err'] = np.abs(adj_y - temp['ctr'])
        temp['lowers'] = temp.index.map(lambda idx: self.lower_indices[self.higher_indices == idx])
        temp['highers'] = temp.index.map(lambda idx: self.higher_indices[self.lower_indices == idx])
        temp['pred lb'] = temp['lowers'].map(lambda l: pred[l]).map(lambda l: 0.0 if len(l) == 0 else l.max())
        temp['pred ub'] = temp['highers'].map(lambda l: pred[l]).map(lambda l: 1.0 if len(l) == 0 else l.min())
        temp['adj lb'] = temp['lowers'].map(lambda l: adj_y[l]).map(lambda l: 0.0 if len(l) == 0 else l.max())
        temp['adj ub'] = temp['highers'].map(lambda l: adj_y[l]).map(lambda l: 1.0 if len(l) == 0 else l.min())
        temp['range'] = [ub - lb for lb, ub in zip(temp['adj lb'], temp['adj ub'])]
        temp['in bet'] = [lb <= ctr <= ub for ctr, lb, ub in zip(temp['ctr'], temp['adj lb'], temp['adj ub'])]
        temp = temp.drop(['average_rating', 'num_reviews', 'dollar_rating', 'highers', 'lowers'], axis=1)
        macs.cache.update(
            ADJ_MAE=temp['adj err'].mean(),
            PRED_MAE=temp['pred err'].mean(),
            RANGE_MEAN=temp['range'].mean(),
            RANGE_MEDN=temp['range'].median(),
            IN_BETWEEN=temp['in bet'].mean(),
            DETAILS=f'\n{solution.solve_details}',
            DATA=f'\n{temp}'
        )

        to_plot = ['pred err', 'adj err', 'range', 'in bet']  # ['ctr', 'pred', 'adj', 'adj ub', 'adj lb']
        _, axes = plt.subplots(2, len(to_plot), sharex='row', sharey='row', figsize=(24, 10), tight_layout=True)
        for i, c in enumerate(to_plot):
            sns.scatterplot(x=temp.index, y=temp[c], ax=axes[0, i]).set_title(c)
            sns.histplot(x=temp[c], ax=axes[1, i]).set_title(c)
        # plt.show()

        # return adjusted labels and sample weights
        return adj_y


class TestMT(MT):
    def log(self, **kwargs):
        pass

    def on_pretraining_end(self, macs, x, y, val_data, **kwargs):
        pass
        # # noinspection PyUnresolvedReferences
        # self.evaluation_summary(**val_data)
        # plt.show()

    def on_training_end(self, macs, x, y, val_data, iteration, **kwargs):
        exit(0)


if __name__ == '__main__':
    # set random seeds
    setup()

    # load and prepare data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = restaurants.load_data()
    x_aug, y_aug, full_aug, aug_scaler = get_augmented_data(x_train, y_train, num_ground_samples=None)
    mono = get_monotonicities_list(full_aug, compute_monotonicities, 'clicked', 'group')

    # moving targets
    mt = TestMT(
        learner=TestMTL(
            build_model=lambda: get_model([16, 8, 8], aug_scaler),
            restart_fit=True,
            validation_data=(x_val, y_val),
            callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
            epochs=200,
            verbose=1
        ),
        master=TestMTM(mono, alpha=0.01, beta=0.01, omega=0.9),
        init_step='pretraining',
        metrics=[AUC(name='auc')]
    )
    mt.fit(x=x_aug.values, y=y_aug['clicked'].values, iterations=1,
           val_data=dict(train=(x_train, y_train), val=(x_val, y_val), test=(x_test, y_test)),
           callbacks=[FileLogger('log.txt', sort_keys=False)])
    # noinspection PyUnresolvedReferences
    mt.evaluation_summary(train=(x_train, y_train), val=(x_val, y_val), test=(x_test, y_test))
    plt.show()
