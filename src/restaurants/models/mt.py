from typing import Optional
from docplex.mp.model import Model as CPModel

from src.moving_targets.learners import Learner
from src.moving_targets.masters import Master
from src.restaurants.models import MLP
from src.restaurants import compute_monotonicities


class MTLearner(Learner):
    def __init__(self, h_units=None, scaler=None, loss='mse', optimizer='adam', **kwargs):
        super(MTLearner, self).__init__()
        self.h_units = h_units
        self.scaler = scaler
        self.loss = loss
        self.optimizer = optimizer
        self.fit_args = kwargs
        self.model: Optional[MLP] = None

    def fit(self, macs, x, y, iteration):
        self.model = MLP(output_act=None, h_units=self.h_units, scaler=self.scaler)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        self.model.fit(x[y != -1], y[y != -1], **self.fit_args)

    def predict(self, x):
        return (self.predict_proba(x) > 0.5).astype('int')

    def predict_proba(self, x):
        return self.model.predict(x).reshape(-1, )


class MTMaster(Master):
    def __init__(self, monotonicities_matrix, alpha=1., beta=1., time_limit=30):
        super(MTMaster, self).__init__(alpha=alpha, beta=beta)
        self.greater_indices, self.lower_indices = np.where(monotonicities_matrix == 1)
        self.time_limit = time_limit

    def adjust_targets(self, macs, x, y, iteration):
        assert macs.init_step == 'pretraining', "This master supports 'pretraining' initial step only"
        p = macs.predict_proba(x)

        # build model, set time limit, and create decision variables
        model = CPModel()
        model.set_time_limit(self.time_limit)
        cp_vars = model.continuous_var_list(keys=len(y), lb=0.0, ub=1.0, name='y')

        # compute monotonicities for each pair of
        # if there is a positive monotonicity (negative ones are ignored as they are simply dual), we:
        #   - change the feasibility according to the fact that the monotonicity is satisfied or not
        #   - add the constraint in the cplex model
        feasible = True
        for greater, lower in zip(self.greater_indices, self.lower_indices):
            feasible = feasible and (p[greater] >= p[lower])
            model.add_constraint(cp_vars[greater] >= cp_vars[lower])

        # define the total loss w.r.t. the true labels (y) and the loss w.r.t. the predictions (p)
        y_loss = [model.abs(yy - vy) for yy, vy in zip(y, cp_vars) if yy != -1]
        y_loss = model.sum(y_loss) / len(y_loss)
        p_loss = [model.abs(py - vy) for py, vy in zip(p, cp_vars)]
        p_loss = model.sum(p_loss) / len(p_loss)

        # core algorithm (minimize depending on feasibility)
        if feasible:
            model.add(p_loss <= self.beta)
            model.minimize(y_loss)
        else:
            model.minimize(y_loss + (1.0 / self.alpha) * p_loss)

        # solve model and return adjusted solutions
        model.solve()
        print(f'MASTER STEP {iteration}:')
        print(model.solve_details)
        print()
        return np.array([vy.solution_value for vy in cp_vars])


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tensorflow.keras.callbacks import EarlyStopping

    from src import restaurants
    from src.moving_targets import MACS
    from src.util.preprocessing import Scaler

    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 1000)
    (xtr, ytr), (xvl, yvl), (xts, yts) = restaurants.load_data()

    n = len(xtr)
    agd, agi = restaurants.augment_data(xtr.iloc[:n], n=5)
    xag = pd.concat((xtr.iloc[:n], agd)).reset_index(drop=True)
    yag = pd.concat((ytr.iloc[:n], agi)).rename({0: 'clicked'}, axis=1).reset_index(drop=True)
    yag = yag.fillna({'ground_index': pd.Series(yag.index), 'clicked': -1, 'monotonicity': 0}).astype('int')
    scl = Scaler(xag, methods=dict(avg_rating='std', num_reviews='std'))

    cbs = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
    arg = dict(epochs=200, validation_data=(xvl, yvl), callbacks=cbs, verbose=1)
    lrn = MTLearner(h_units=[16, 8, 8], scaler=scl, loss='mse', optimizer='adam', **arg)

    mtx = compute_monotonicities(xag.values, xag.values)
    mst = MTMaster(mtx)

    mt = MACS(lrn, mst, init_step='pretraining')
    mt.fit(xag.values, yag['clicked'].values, iterations=5)

    lrn.model.evaluation_summary(train=(xtr, ytr), val=(xvl, yvl), test=(xts, yts))
    plt.show()
