from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class TensorMeter:
    """Tracks last value and running arithmetic mean (no history, no grad)."""
    n: int = 0
    last: Optional[torch.Tensor] = None
    avg: Optional[torch.Tensor] = None

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.detach()

        # last
        if self.last is None:
            self.last = x.clone()
        else:
            self.last.copy_(x)

        # avg (running mean)
        if self.avg is None:
            self.avg = x.clone()
            self.n = 1
            return

        self.n += 1
        self.avg.add_(x - self.avg, alpha=(1.0 / self.n))

    def reset(self) -> None:
        self.n = 0
        self.last = None
        self.avg = None
