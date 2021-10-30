from tqdm import tqdm
from time import time, sleep
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nproc', type=int, default=1)
parser.add_argument('--runs', type=int, default=3)
args = parser.parse_args()

os.environ["OMP_NUM_THREADS"] = str(args.nproc)
os.environ["OPENBLAS_NUM_THREADS"] = str(args.nproc)
os.environ["MKL_NUM_THREADS"] = str(args.nproc)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.nproc)
os.environ["NUMEXPR_NUM_THREADS"] = str(args.nproc)

print("\nUsing", args.nproc, "threads for benchmark\n")

size_list = []
tflops_list = {
    "half": [],
    "single": [],
    "double": [],
}

def get_tflops(size, iterations, dtype):
    print(size, dtype)
    A, B = np.random.random((size, size)), np.random.random((size, size))
    if dtype == "half":
        A, B = A.astype(np.float16), B.astype(np.float16)
    elif dtype == "single":
        A, B = A.astype(np.float32), B.astype(np.float32)
    elif dtype == "double":
        A, B = A.astype(np.float64), B.astype(np.float64)
    else:
        print("Unknown dtype", dtype, "using system default")
    
    np.matmul(A, B)
    st = time()
    for i in range(1):
        np.matmul(A, B)
    et = time()
    overhead = et-st
    sleep(2.0)
    st = time()
    for i in range(iterations+1):
        np.matmul(A, B)
    et = time()
    duration = et-st - overhead
    if duration < 3:
        extend_ratio = 3/duration
        new_iterations = int(extend_ratio*iterations)
        print("new_iterations", new_iterations)
        st = time()
        for i in range(new_iterations+1):
            np.matmul(A, B)
        et = time()
        duration = et-st - overhead
        fps = new_iterations/duration
    else:
        fps = iterations/duration
    matmul_flops = 2 * (size**3)
    TFLOPS = fps*matmul_flops/(1e12)
    size_list.append(size)
    tflops_list[dtype].append(TFLOPS)
    sleep(2.0)

for size in tqdm([8, 32, 128, 1024, 2048, 4096, 8192]):
    get_tflops(size, iterations=args.runs, dtype="single")

print(tflops_list)

for size in tqdm([8, 32, 128, 1024, 2048, 4096, 8192]):
    get_tflops(size, iterations=args.runs, dtype="double")

print(tflops_list)

for size in tqdm([8, 32, 128, 1024]):
    get_tflops(size, iterations=args.runs, dtype="half")

print(tflops_list)


