import numpy as np

from moving_targets.masters.cplex_master import CplexMaster


class BalancedCounts(CplexMaster):
    def __init__(self, n_classes, alpha=1., beta=1., time_limit=30, use_prob=True):
        super(BalancedCounts, self).__init__(alpha=alpha, beta=beta, time_limit=time_limit)
        self.num_classes = n_classes
        self.use_prob = use_prob

    def build_model(self, macs, model, x, y, iteration):
        # if the model has not been fitted yet (i.e., the initial macs step is 'projection') we use the original labels
        # otherwise we use either the predicted classes or the predicted probabilities
        if not macs.fitted:
            prob = None
            pred = y.reshape(-1, )
        elif self.use_prob is False:
            prob = None
            pred = macs.learner.predict(x)
        else:
            assert hasattr(macs.learner, 'predict_proba'), "Learner must have method 'predict_proba(x)' for use_prob"
            prob = np.clip(macs.learner.predict_proba(x), a_min=.01, a_max=.99)
            pred = macs.learner.predict(x)

        # define variables and max_count (i.e., upper bound for number of counts for a class)
        num_samples = len(y)
        max_count = np.ceil(1.05 * num_samples / self.num_classes)
        variables = model.binary_var_matrix(keys1=num_samples, keys2=self.num_classes, name='y').values()
        variables = np.array(list(variables)).reshape(num_samples, self.num_classes)

        # constrain the class counts to the maximal value
        for c in range(self.num_classes):
            class_count = model.sum([variables[i, c] for i in range(num_samples)])
            model.add_constraint(class_count <= max_count)
        # each sample should be labeled with one class only
        for i in range(num_samples):
            class_label = model.sum(variables[i, c] for c in range(self.num_classes))
            model.add_constraint(class_label == 1)

        # return model info
        return variables, pred, prob, max_count

    def beta_step(self, macs, model, model_info, x, y, iteration):
        _, pred, _, max_count = model_info
        _, pred_classes_counts = np.unique(pred, return_counts=True)
        return np.all(pred_classes_counts <= max_count)

    def y_loss(self, macs, model, model_info, x, y, iteration):
        variables, _, _, _ = model_info
        return CplexMaster.categorical_hamming(model=model, numeric_variables=y, model_variables=variables)

    def p_loss(self, macs, model, model_info, x, y, iteration):
        variables, pred, prob, _ = model_info
        if prob is None:
            return CplexMaster.categorical_hamming(model=model, numeric_variables=pred, model_variables=variables)
        else:
            return CplexMaster.categorical_crossentropy(model=model, numeric_variables=prob, model_variables=variables)

    def return_solutions(self, macs, solution, model_info, x, y, iteration):
        variables, _, _, _ = model_info
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
