import pandas as pd
import numpy as np
import scipy.stats as st

def many_series_to_one():
    all_idxs = range(1000)
    n = 10
    urv = st.uniform()
    all_series = []
    for i in range(100):
        this_idxs = np.random.choice(all_idxs, size=n, replace=False)
        data = urv.rvs(n)
        this_series = pd.Series(
            data=data,
            index=this_idxs,
        )
        all_series += [this_series]

    s = all_series[0]
    for i, t in enumerate(all_series[1:]):
        s = s.add(t,fill_value=0.0)
        if s.index.has_duplicates:
            print(i+1)
            print(s)
            break
    return s, all_series
