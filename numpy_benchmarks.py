from tqdm import trange
from time import time, sleep
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nproc', type=int, default=1)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--size', type=int, default=4096)
args = parser.parse_args()

os.environ["OMP_NUM_THREADS"] = str(args.nproc)
os.environ["OPENBLAS_NUM_THREADS"] = str(args.nproc)
os.environ["MKL_NUM_THREADS"] = str(args.nproc)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.nproc)
os.environ["NUMEXPR_NUM_THREADS"] = str(args.nproc)

print("\nUsing", args.nproc, "threads for benchmark\n")


timings = {
    "datagen": 0,
    "special": 0,
    "stats": 0,
    "matmul": 0,
    "vecmul": 0,
    "svd": 0,
    "cholesky": 0,
    "eigendecomp": 0,
}

print("Running benchmarks...\n")

for i in trange(args.runs):
    t = time()
    size = args.size
    print("size =", size)
    A, B = np.random.random((size, size)), np.random.random((size, size))
    C, D = np.random.random((size * size,)), np.random.random((size * size,))
    E = np.random.random((int(size / 2), int(size / 4)))
    F = np.random.random((int(size / 2), int(size / 2)))
    F = np.dot(F, F.T)
    G = np.random.random((int(size / 2), int(size / 2)))
    delta = time() - t
    timings["datagen"] += delta

    def run_special_funcs(nparray):
        np.exp(nparray)
        np.sqrt(nparray)
        np.sin(nparray)
        np.log(nparray)

    N = 3
    t = time()
    for i in range(N):
        run_special_funcs(A)
        run_special_funcs(C)
    delta = time() - t
    timings["special"] += delta/N
    sleep(2.0)

    def run_stats(nparray):
        nparray.sum()
        nparray.min()
        nparray.max()
        nparray.cumsum()
        nparray.mean()
        np.median(nparray)
        np.corrcoef(nparray)
        np.std(nparray)

    t = time()
    for i in range(N):
        run_stats(A)
        run_stats(C)
    delta = time() - t
    timings["stats"] += delta/N
    sleep(2.0)

    t = time()
    for i in range(N):
        np.dot(A, B)
    delta = time() - t
    del A, B
    timings["matmul"] += delta/N
    sleep(2.0)

    t = time()
    for i in range(N):
        np.dot(C, D)
    delta = time() - t
    del C, D
    timings["vecmul"] += delta/N
    sleep(2.0)

    t = time()
    for i in range(N):
        np.linalg.svd(E, full_matrices=False)
    delta = time() - t
    del E
    timings["svd"] += delta/N
    sleep(2.0)

    t = time()
    for i in range(N):
        np.linalg.cholesky(F)
    delta = time() - t
    del F
    timings["cholesky"] += delta/N
    sleep(2.0)

    # eigendecomp is slow, set max runs to 3
    t = time()
    for i in range(N):
        np.linalg.eig(G)
    delta = time() - t
    del G
    timings["eigendecomp"] += delta/N
    sleep(2.0)

print("\nDone!\n")

print("Results")
print("=======")
for key in timings.keys():
    timing = round(timings[key]/args.runs, 3)
    print("| "+key+" |", timing, "|")
print("")
