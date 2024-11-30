from typing import Any
import math

class batch_RunningMeanStd:
    def __init__(self):  # shape:the dimension of input data
        self.n = 0
        self.mean = 0
        self.S = 0
        self.std = math.sqrt(self.S)

    def update(self, x):
        self.n += 1
        if self.n == 1:
            self.mean = x.mean().item()
            self.std = x.std().item()
        else:
            old_mean = self.mean
            self.mean = old_mean + (x.mean().item() - old_mean) / self.n
            self.S = self.S + (x.mean().item() - old_mean) * (x.mean().item() - self.mean)
            self.std = math.sqrt(self.S / self.n)


class batch_Normalization:
    def __init__(self):
        self.running_ms = batch_RunningMeanStd()
        
    def __call__(self, x, update=True) -> Any:
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        return x