import time
import warnings
from typing import Optional, Union, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from moving_targets.callbacks import Logger, Callback
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import PolynomialFeatures

from src.datasets import AnalysisCallback
from src.experiments import Handler


class CarsRegressor(AnalysisCallback, Logger):
    """Investigates how the master problem builds the regressor weights in the cars dataset.

    - 'degree' is the polynomial degree to cancel higher-order effects.
    """

    def __init__(self, degree: int, file_signature: Optional[str] = None, num_columns: Union[int, str] = 'auto'):
        super(CarsRegressor, self).__init__(sorting_attribute='x_1',
                                            file_signature=file_signature,
                                            num_columns=num_columns)
        assert degree > 0, f"'degree' should be a positive integer, got {degree}"
        self.degree: int = degree

    def on_process_start(self, macs, x, y, val_data):
        a = PolynomialFeatures(degree=self.degree).fit_transform(x[['price']])
        w, _, _, _ = np.linalg.lstsq(a, y, rcond=None)
        self.data = pd.DataFrame(a, columns=[f'x_{degree}' for degree in range(self.degree + 1)])
        self.data['adj 0'] = y
        self.data['ref 0'] = a @ w

    def on_adjustment_end(self, macs, x, y, z, val_data):
        a = self.data[[c for c in self.data.columns if c.startswith('x')]]
        w_lr = self._cache['w_lr']
        w_mt = self._cache['w_mt']
        self.data[f'adj {macs.iteration}'] = z
        self.data[f'ref {macs.iteration}'] = a @ w_lr
        self.data[f'mt {macs.iteration}'] = a @ w_mt

    def _plot_function(self, iteration: int) -> Optional[str]:
        x = self.data['x_1'].values
        plt.plot(x, self.data[f'adj {iteration}'], color='red', alpha=0.4)
        plt.plot(x, self.data[f'ref {iteration}'], color='blue')
        if iteration == 0:
            plt.legend(['adjusted', 'ref regression'])
        else:
            plt.plot(x, self.data[f'mt {iteration}'], color='black')
            plt.legend(['adjusted', 'ref regression', 'mt regression'])
        return


class ExperimentCallback(Callback):
    def __init__(self, degrees):
        super(ExperimentCallback, self).__init__()
        self.idx: int = 0
        self.time: float = 0.0
        self.data: pd.DataFrame = pd.DataFrame()
        self.degrees: List[int] = list(degrees)

    def on_process_start(self, macs, x, y, val_data):
        # rewrite x and y after each new experiment, it does not change anything since it is always the same input
        print(f'> Degree {self.degrees[self.idx]:2} ({self.idx + 1:02}/{len(self.degrees):02})', end='')
        self.time = time.time()
        self.data['x'] = x['price']
        self.data['y'] = y

    # def on_adjustment_end(self, macs, x, y, z, val_data):
    def on_pretraining_end(self, macs, x, y, p, val_data):
        # rewrite self.data[degree] after each adjustment of the same experiment since we want just the last one
        degree = self.degrees[self.idx]
        self.data[degree] = p

    def on_process_end(self, macs, val_data):
        # increase the idx after each new experiment
        print(f' -- elapsed time: {time.time() - self.time:.2f}s')
        self.idx += 1

    def plot(self):
        self.data = self.data.sort_values('x')
        plt.figure(figsize=(16, 9), tight_layout=True)
        num_columns = max(np.sqrt(16 * len(self.degrees) / 9).round().astype(int), 1)
        num_rows = np.ceil(len(self.degrees) / num_columns).astype(int)
        ax = None
        for idx, deg in enumerate(self.degrees):
            ax = plt.subplot(num_rows, num_columns, idx + 1, sharex=ax, sharey=ax)
            x = self.data['x'].values
            y = self.data['y'].values
            z = self.data[deg].values
            a = PolynomialFeatures(degree=deg).fit_transform(x.reshape((-1, 1)))
            w, _, _, _ = np.linalg.lstsq(a, z, rcond=None)
            plt.scatter(x, y, alpha=0.3, color='red', label='targets')
            plt.plot(x, a @ w, alpha=0.6, color='blue', label='regressor')
            plt.plot(x, z, color='black', label='predictions')
            ax.set(xlabel='', ylabel='')
            ax.set_title(f'Order {deg}')
            ax.legend()
        plt.show()


# TODO: with degree 10 we get an infeasible master problem at iteration 9, is this due to numerical errors?

if __name__ == '__main__':
    warnings.simplefilter("ignore", category=ConvergenceWarning)

    # callback = ExperimentCallback(degrees=[1, 2, 3, 5, 10, 20])
    # for d in callback.degrees:
    #     Handler(dataset='cars', loss='mse', degree=d).experiment(
    #         iterations=10,
    #         num_folds=None,
    #         callbacks=[callback],
    #         model_verbosity=False,
    #         fold_verbosity=False,
    #         plot_history=False,
    #         plot_summary=False
    #     )
    # callback.plot()

    d = 10
    Handler(dataset='cars', loss='mse', degree=d).experiment(
        iterations=10,
        num_folds=None,
        # callbacks=[],
        callbacks=[CarsRegressor(degree=d), 'adjustments_line', 'adjustments_scatter'],
        model_verbosity=1,
        fold_verbosity=False,
        plot_history=False,
        plot_summary=True
    )
