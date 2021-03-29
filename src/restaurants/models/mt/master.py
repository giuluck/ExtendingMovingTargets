import numpy as np
from docplex.mp.model import Model as CPModel

from src.moving_targets.masters import Master


class MTMaster(Master):
    def __init__(self, monotonicities, alpha=1., beta=1., time_limit=30):
        super(MTMaster, self).__init__(alpha=alpha, beta=beta)
        self.monotonicities = monotonicities
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
        for higher, lower in self.monotonicities:
            feasible = feasible and (p[higher] >= p[lower])
            model.add_constraint(cp_vars[higher] >= cp_vars[lower])

        # define the total loss w.r.t. the true labels (y) and the loss w.r.t. the predictions (p)
        y_loss = [model.abs(yv - cpv) for yv, cpv in zip(y, cp_vars) if yv != -1]
        y_loss = model.sum(y_loss) / len(y_loss)
        p_loss = [model.abs(pv - cpv) for pv, cpv in zip(p, cp_vars)]
        p_loss = model.sum(p_loss) / len(p_loss)

        # core algorithm (minimize depending on feasibility)
        if feasible:
            model.add(p_loss <= self.beta)
            model.minimize(y_loss)
        else:
            model.minimize(y_loss + (1.0 / self.alpha) * p_loss)

        # solve model and return adjusted solutions
        model.solve()
        return np.array([vy.solution_value for vy in cp_vars])
