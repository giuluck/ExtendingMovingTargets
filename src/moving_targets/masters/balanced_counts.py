import numpy as np
from docplex.mp.model import Model as CPModel

from src.moving_targets.masters import Master


class BalancedCounts(Master):
    def __init__(self, n_classes, alpha=1., beta=1., use_prob=True, time_limit=30):
        super(BalancedCounts, self).__init__(alpha=alpha, beta=beta)
        self.n_classes = n_classes
        self.use_prob = use_prob
        self.time_limit = time_limit

    def adjust_targets(self, macs, x, y, iteration):
        # if this is the first iteration and the initial macs step is 'projection', the learner has not been fitted yet
        # thus we use the original labels, otherwise we use either the predicted classes or the predicted probabilities
        if iteration == 0 and macs.init_step == 'projection':
            prob = None
            pred = y.reshape(-1, )
        elif self.use_prob is False:
            prob = None
            pred = macs.learner.predict(x)
        else:
            prob = np.clip(macs.learner.predict_proba(x), a_min=.01, a_max=.99)
            pred = np.argmax(prob, axis=1)

        n_samples = len(y)
        max_count = np.ceil(1.05 * n_samples / self.n_classes)  # upper bound for number of counts for a class
        _, pred_classes_counts = np.unique(pred, return_counts=True)

        # build model and the decision variables
        model = CPModel()
        model.set_time_limit(self.time_limit)
        vy = model.binary_var_matrix(keys1=n_samples, keys2=self.n_classes, name='y')

        # constrain the class counts to the maximal value
        for c in range(self.n_classes):
            class_count = model.sum([vy[i, c] for i in range(n_samples)])
            model.add_constraint(class_count <= max_count)
        # each sample should be labeled with one class only
        for i in range(n_samples):
            class_label = model.sum(vy[i, c] for c in range(self.n_classes))
            model.add_constraint(class_label == 1)

        # define the total loss w.r.t. the true labels (y) and the loss w.r.t. the predictions (prob / pred)
        y_loss = model.sum([BalancedCounts._indicator_loss(vy, y, i) for i in range(n_samples)]) / n_samples
        if prob is None:
            p_loss = model.sum([BalancedCounts._indicator_loss(vy, pred, i) for i in range(n_samples)]) / n_samples
        else:
            p_loss = model.sum([BalancedCounts._crossentropy_loss(vy, prob, i) for i in range(n_samples)]) / n_samples

        # check for feasibility (i.e., if all the classes have a number of samples which is lesser than the maximal)
        # and behave depending on that
        if np.all(pred_classes_counts <= max_count):
            model.add(p_loss <= self.beta)
            model.minimize(y_loss)
        else:
            model.minimize(y_loss + (1.0 / self.alpha) * p_loss)

        # solve the problem and get the adjusted labels
        sol = model.solve()
        y_adj = [sum(c * sol.get_value(vy[i, c]) for c in range(self.n_classes)) for i in range(n_samples)]
        y_adj = np.array([int(v) for v in y_adj])

        return y_adj

    @staticmethod
    def _crossentropy_loss(var_y, probabilities, i):
        # the loss is the negative log-probability over all the possible classes
        n_classes = probabilities.shape[1]
        return -sum([var_y[i, j] * np.log(probabilities[i, j]) for j in range(n_classes)])

    @staticmethod
    def _indicator_loss(var_y, labels, i):
        # var_y[i, label[i]] = 1 if the i-th sample has class label[i], 0 otherwise
        # therefore, the loss for each sample is 0 if the class has been assigned correctly, 1 otherwise
        return 1 - var_y[i, labels[i]]
