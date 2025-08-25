import numpy as np
import pandas as pd

from transformational import trade_when


if __name__ == "__main__":
    import time
    
    np.random.seed(42)
    T, N = 50000, 3
    
    x = (np.random.random((T, N)) < 0.1).astype(int) 
    y = np.random.randn(T, N)
    y[np.random.random((T, N)) < 0.2] = np.nan 
    z = (np.random.random((T, N)) < 0.05).astype(int)  

    df_x = pd.DataFrame(x)
    df_x.columns = [f"x_{i}" for i in range(N)]
    df_y = pd.DataFrame(y)
    df_y.columns = [f"y_{i}" for i in range(N)]
    df_z = pd.DataFrame(z)
    df_z.columns = [f"z_{i}" for i in range(N)]

    start = time.time()
    result0 = trade_when(df_x, df_y, df_z)
    time0 = time.time() - start


    df_results = pd.DataFrame(result0)
    df_results.columns = [f"tradewhen_{i}" for i in range(N)]

    df_final = pd.concat([df_x, df_y, df_z, df_results], axis=1)
    df_final.astype(np.float32)
    # df_final.to_csv("trade_when_output.csv", index=False)

    print(f"Original trade_when time: {time0:.4f} seconds")