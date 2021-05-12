class Master:
    def __init__(self, alpha: float, beta: float):
        super(Master, self).__init__()
        assert alpha > 0, "alpha should be a positive number"
        assert beta > 0, "beta should be a positive number"
        self.alpha = alpha
        self.beta = beta

    def build_model(self, macs, model, x, y, iteration):
        raise NotImplementedError("Please implement method 'build_model'")

    def beta_step(self, macs, model, model_info, x, y, iteration):
        return False

    def y_loss(self, macs, model, model_info, x, y, iteration):
        return 0.0

    def p_loss(self, macs, model, model_info, x, y, iteration):
        return 0.0

    def return_solutions(self, macs, solution, model_info, x, y, iteration):
        raise NotImplementedError("Please implement method 'return_solutions'")

    def adjust_targets(self, macs, x, y, iteration):
        pass
