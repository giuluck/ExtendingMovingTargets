class Metric:
    def __init__(self, name):
        super(Metric, self).__init__()
        self.name = name

    def __call__(self, x, y, p):
        pass
