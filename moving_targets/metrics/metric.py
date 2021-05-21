class Metric:
    def __init__(self, name: str):
        super(Metric, self).__init__()
        self.__name__ = name

    def __call__(self, x, y, p) -> float:
        pass
