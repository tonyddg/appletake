import torch
import numpy as np
import math

def MaintainCosineLR(init_lr: float = 4e-3, maintain_ratio: float = 0.3, adjust_ratio: float = 0.9, eta_min_rate: float = 1e-2):
    def get_init_lr_times(remain_ratio: float):
        progress_ratio = 1 - remain_ratio
        
        if progress_ratio < maintain_ratio:
            return init_lr
        elif progress_ratio < adjust_ratio:
            decay_ratio = (progress_ratio - maintain_ratio) / (adjust_ratio - maintain_ratio)
            return init_lr * (eta_min_rate + 0.5 * (1 - eta_min_rate) * (1 + math.cos(decay_ratio * torch.pi)))
        else:
            return init_lr * eta_min_rate

    return get_init_lr_times

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    fun = MaintainCosineLR()
    t = np.arange(0, 1, 0.01)
    plt.plot(t, np.array([fun(1 - float(i)) for i in t]))
    plt.savefig("./tmp/plot.png")
