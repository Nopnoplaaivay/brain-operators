import numpy as np
import pandas as pd


def trade_when(x: pd.DataFrame, y: pd.DataFrame, z: pd.DataFrame) -> pd.DataFrame:
    x_ = x.values
    y_ = y.values
    z_ = z.values

    out = np.where(z_ == True, 0, np.where(x_ == True, y_, np.nan))
    df = pd.DataFrame(out)
    df.ffill(axis=0, inplace=True)
    df.fillna(0, inplace=True)

    return df