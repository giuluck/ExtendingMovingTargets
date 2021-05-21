from typing import Any, Optional


class Master:
    def __init__(self, alpha: float, beta: Optional[float]):
        super(Master, self).__init__()
        assert alpha > 0, "alpha should be a positive number"
        assert beta is None or beta > 0, "beta should be either None or a positive number"
        self.alpha: float = alpha
        self.beta: float = beta

    def build_model(self, macs, model, x, y, iteration: Any):
        raise NotImplementedError("Please implement method 'build_model'")

    def beta_step(self, macs, model, model_info, x, y, iteration: Any) -> bool:
        return False

    def y_loss(self, macs, model, model_info, x, y, iteration: Any) -> float:
        return 0.0

    def p_loss(self, macs, model, model_info, x, y, iteration: Any) -> float:
        return 0.0

    def return_solutions(self, macs, solution, model_info, x, y, iteration: Any) -> Any:
        raise NotImplementedError("Please implement method 'return_solutions'")

    def adjust_targets(self, macs, x, y, iteration: Any) -> Any:
        pass
