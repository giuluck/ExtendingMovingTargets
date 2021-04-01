class Master:
    def __init__(self, alpha: float, beta: float):
        super(Master, self).__init__()
        assert alpha > 0, "alpha should be a positive number"
        assert beta > 0, "beta should be a positive number"
        self.alpha = alpha
        self.beta = beta

    def adjust_targets(self, macs, x, y, iteration):
        pass
