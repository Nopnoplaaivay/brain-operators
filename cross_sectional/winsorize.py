import numpy as np

def winsorize(x, std=4):
    mu = np.nanmean(x, axis=0, keepdims=True)
    sigma = np.nanstd(x, axis=0, keepdims=True)
    lower = mu - std * sigma
    upper = mu + std * sigma
    return np.clip(x, lower, upper)
