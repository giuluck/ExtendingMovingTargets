import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from moving_targets.callbacks import Logger
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

# TODO: increasing polynomial orders is not numerically stable
#
# > even numpy least squares method makes mistakes (quantified by the sum of the differences between (A.T @ A) @ w and
#   A.T @ z, but they are still some order of magnitudes less than the errors made in the master
# > for orders higher than 4, there is a lot of discrepancy between the master weights and the numpy weights
# > sometimes the master weights don't even satisfy the constraint -- TODO: check why
# > using standardized inputs/outputs to limit numerical errors has only a small effect


if __name__ == '__main__':
    warnings.simplefilter("ignore", category=ConvergenceWarning)

    d = 10
    Handler(dataset='cars', loss='mse', degree=d).experiment(
        iterations=10,
        num_folds=None,
        callbacks=[CarsRegressor(degree=d)],
        # callbacks=[],
        model_verbosity=False,
        fold_verbosity=False,
        plot_history=False,
        plot_summary=False
    )
