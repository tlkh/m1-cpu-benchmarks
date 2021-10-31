import os
os.environ["MODIN_ENGINE"] = "dask"
from dask.distributed import Client
client = Client(n_workers=10)
import time
import string
import numpy as np
import modin.pandas as pd

runs = 3

min_value, max_value = 0, 9223372036854775807

results = {
    "datagen": 0,
    "inner_merge": 0,
    "outer_merge": 0,
}

def benchmark():
    st = time.time()
    cols = list(string.printable)[:75]
    size = (int(1e6), len(cols))
    df1 = pd.DataFrame(np.random.randint(min_value,max_value,size=size), columns=cols).astype("float")
    cols = list(string.printable)[-75:]
    size = (int(1e6), len(cols))
    df2 = pd.DataFrame(np.random.randint(min_value,max_value,size=size), columns=cols).astype("float")
    et = time.time()
    duration = et-st
    results["datagen"] += duration
    
    st = time.time()
    inner_merge = pd.merge(df1, df2, how="inner")
    et = time.time()
    duration = et-st
    results["inner_merge"] += duration
    
    st = time.time()
    outer_merge = pd.merge(df1, df2, how="outer")
    et = time.time()
    duration = et-st
    results["outer_merge"] += duration
    
for i in range(runs):
    benchmark()
    
results["datagen"] /= runs
results["inner_merge"] /= runs
results["outer_merge"] /= runs

print("")
print("Results")
print("=======")
print("datagen:", round(results["datagen"],1))
print("inner_merge:", round(results["inner_merge"],1))
print("outer_merge:", round(results["outer_merge"],1))
    