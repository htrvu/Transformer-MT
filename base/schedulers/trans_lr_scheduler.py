import torch


class TransLRScheduler:
    """
    Special learning rate scheduler for Transformer from the paper
    """

    def __init__(
        self, optimizer: torch.optim, init_lr: float, d_model: int, n_warmup_steps: int
    ) -> None:
        """
        Args:
            - optimizer (torch.optim): Optimizer
            - init_lr (float): Initial learning rate
            - d_model (int): Dimension of model
            - n_warmup_steps (int): Number of warmup steps
        """
        self._optimizer = optimizer
        self._init_lr = init_lr
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return (self.d_model**-0.5) * min(
            self.n_steps ** (-0.5), self.n_steps * self.n_warmup_steps ** (-1.5)
        )

    def _update_learning_rate(self):
        self.n_steps += 1
        lr = self._init_lr * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

    def state_dict(self):
        return {
            "_optimizer": self._optimizer.state_dict(),
            "init_lr": self._init_lr,
            "d_model": self.d_model,
            "n_warmup_steps": self.n_warmup_steps,
            "n_steps": self.n_steps,
        }

    def load_state_dict(self, state_dict):
        self.init_lr = state_dict["init_lr"]
        self.d_model = state_dict["d_model"]
        self.n_warmup_steps = state_dict["n_warmup_steps"]
        self.n_steps = state_dict["n_steps"]

        self._optimizer.load_state_dict(state_dict["_optimizer"])
