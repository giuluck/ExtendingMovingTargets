import numpy as np
from docplex.mp.model import Model as CPModel


def _crossentropy_loss(var_y, probabilities, i):
    # the loss is the negative log-probability over all the possible classes
    num_classes = probabilities.shape[1]
    return -sum([var_y[i, j] * np.log(probabilities[i, j]) for j in range(num_classes)])


def _indicator_loss(var_y, labels, i):
    # var_y[i, label[i]] = 1 if the i-th sample has class label[i], 0 otherwise
    # therefore, the loss for each sample is 0 if the class has been assigned correctly, 1 otherwise
    return 1 - var_y[i, labels[i]]


class Master:
    def __init__(self, **kwargs):
        super(Master, self).__init__()

    def adjust_targets(self, y, pred, alpha, beta, use_prob):
        pass


class BalancedCounts(Master):
    def __init__(self, num_classes, time_limit=30):
        super(BalancedCounts, self).__init__()
        self.num_classes = num_classes
        self.time_limit = time_limit

    def adjust_targets(self, y, pred, alpha, beta, use_prob):
        prob = None
        if use_prob:
            prob = pred.copy()
            prob = np.clip(prob, a_min=.01, a_max=.99)
            pred = np.argmax(prob, axis=1)

        num_samples = len(y)
        max_count = np.ceil(1.05 * num_samples / self.num_classes)  # upper bound for number of counts for a class
        _, pred_classes_counts = np.unique(pred, return_counts=True)

        # build model and the decision variables
        model = CPModel()
        model.timelimit = self.time_limit
        vy = model.binary_var_matrix(keys1=num_samples, keys2=self.num_classes, name='y')

        # constrain the class counts to the maximal value
        for c in range(self.num_classes):
            class_count = model.sum([vy[i, c] for i in range(num_samples)])
            model.add_constraint(class_count <= max_count)
        # each sample should be labeled with one class only
        for i in range(num_samples):
            class_label = model.sum(vy[i, c] for c in range(self.num_classes))
            model.add_constraint(class_label == 1)

        # define the total loss w.r.t. the true labels (y) and the loss w.r.t. the predictions (prob / pred)
        y_loss = (1 / num_samples) * model.sum([_indicator_loss(vy, y, i) for i in range(num_samples)])
        if use_prob:
            p_loss = (1 / num_samples) * model.sum([_crossentropy_loss(vy, prob, i) for i in range(num_samples)])
        else:
            p_loss = (1 / num_samples) * model.sum([_indicator_loss(vy, pred, i) for i in range(num_samples)])

        # check for feasibility (i.e., if all the classes have a number of samples which is lesser than the maximal)
        # and behave depending on that
        if np.all(pred_classes_counts <= max_count):
            model.add(p_loss <= beta)
            model.minimize(y_loss)
        else:
            model.minimize(y_loss + (1.0 / alpha) * p_loss)

        # solve the problem and get the adjusted labels
        sol = model.solve()
        y_adj = [sum(c * sol.get_value(vy[i, c]) for c in range(self.num_classes)) for i in range(num_samples)]
        y_adj = np.array([int(v) for v in y_adj])

        return y_adj
