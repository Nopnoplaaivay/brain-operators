import numpy as np
import pandas as pd


def trade_when(x, y, z):
    out = np.where(z == True, 0, np.where(x == True, y, np.nan))
    df = pd.DataFrame(out)
    df.ffill(axis=0, inplace=True)
    df.fillna(0, inplace=True)

    return df