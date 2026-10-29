"""Adam with the exact update rule of Keras 2.2 (keras/optimizers.py).

    p <- p - lr * sqrt(1 - b2^t) / (1 - b1^t) * m / (sqrt(v) + eps)

torch.optim.Adam instead adds eps to the bias-corrected sqrt(v_hat), which gives
noticeably different steps for gradient entries of magnitude ~eps / sqrt(1 - b2^t).
"""
import math

import torch


class KerasAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-7):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state.update(t=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                state["t"] += 1
                state["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
                state["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                lr_t = group["lr"] * math.sqrt(1 - b2 ** state["t"]) / (1 - b1 ** state["t"])
                p.addcdiv_(state["m"], state["v"].sqrt().add_(group["eps"]), value=-lr_t)
        return loss
