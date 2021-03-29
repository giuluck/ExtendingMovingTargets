import numpy as np

from src.moving_targets.masters import CplexMaster


class BalancedCounts(CplexMaster):
    def __init__(self, n_classes, alpha=1., beta=1., time_limit=30, use_prob=True):
        super(BalancedCounts, self).__init__(alpha=alpha, beta=beta, time_limit=time_limit)
        self.num_classes = n_classes
        self.use_prob = use_prob

    def define_variables(self, macs, model, x, y, iteration):
        variables = model.binary_var_matrix(keys1=len(y), keys2=self.num_classes, name='y')
        variables = list(variables.values())
        return np.array(variables).reshape(-1, self.num_classes)

    def compute_losses(self, macs, model, variables, x, y, iteration):
        # if this is the first iteration and the initial macs step is 'projection', the learner has not been fitted yet
        # thus we use the original labels, otherwise we use either the predicted classes or the predicted probabilities
        if iteration == 0 and macs.init_step == 'projection':
            prob = None
            pred = y.reshape(-1, )
        elif self.use_prob is False:
            prob = None
            pred = macs.learner.predict(x)
        else:
            assert hasattr(macs.learner, 'predict_proba'), "Learner must have method 'predict_proba(x)' for use_prob"
            # noinspection PyUnresolvedReferences
            prob = np.clip(macs.learner.predict_proba(x), a_min=.01, a_max=.99)
            pred = np.argmax(prob, axis=1)

        num_samples = len(y)
        max_count = np.ceil(1.05 * num_samples / self.num_classes)  # upper bound for number of counts for a class
        _, pred_classes_counts = np.unique(pred, return_counts=True)

        # constrain the class counts to the maximal value
        for c in range(self.num_classes):
            class_count = model.sum([variables[i, c] for i in range(num_samples)])
            model.add_constraint(class_count <= max_count)
        # each sample should be labeled with one class only
        for i in range(num_samples):
            class_label = model.sum(variables[i, c] for c in range(self.num_classes))
            model.add_constraint(class_label == 1)

        # define feasibility and total loss w.r.t. the true labels (y) and the loss w.r.t. the predictions (prob / pred)
        is_feasible = np.all(pred_classes_counts <= max_count)
        y_loss = model.sum([BalancedCounts._indicator_loss(vv, vy) for vv, vy in zip(variables, y)]) / num_samples
        if prob is None:
            p_loss = model.sum([self._indicator_loss(vv, vp) for vv, vp in zip(variables, pred)]) / num_samples
        else:
            p_loss = model.sum([self._crossentropy_loss(vv, vp) for vv, vp in zip(variables, prob)]) / num_samples
        return is_feasible, y_loss, p_loss

    def return_solutions(self, macs, solution, variables, x, y, iteration):
        y_adj = [sum(c * solution.get_value(variables[i, c]) for c in range(self.num_classes)) for i in range(len(y))]
        y_adj = np.array([int(v) for v in y_adj])
        return y_adj

    @staticmethod
    def _crossentropy_loss(variable, probabilities):
        # the loss is the negative log-probability over all the possible classes
        n_classes = len(probabilities)
        return -sum([variable[j] * np.log(probabilities[j]) for j in range(n_classes)])

    @staticmethod
    def _indicator_loss(variable, label):
        # variable[label] = 1 if the variable has the same class as the label, 0 otherwise
        # therefore, the loss for each sample is 0 if the class has been assigned correctly, 1 otherwise
        return 1 - variable[label]
