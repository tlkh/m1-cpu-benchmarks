import time
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
key = random.PRNGKey(0)

size = 4096
iterations = 60

mat = random.normal(key, (size, size))
mat_T = mat.T

def apply_matrix(v):
    return jnp.dot(mat, v)

apply_matrix(mat_T).block_until_ready()

time.sleep(1)

st = time.time()
for i in range(iterations):
    apply_matrix(mat_T).block_until_ready()
et = time.time()

duration = et-st
fps = iterations/duration
matmul_flops = 2 * (size**3)
TFLOPS = fps*matmul_flops/(1e12)

print("| MatMul |", TFLOPS, "|")

time.sleep(1)

apply_matrix_jit = jit(apply_matrix)

apply_matrix_jit(mat_T).block_until_ready()

time.sleep(1)

st = time.time()
for i in range(iterations):
    apply_matrix_jit(mat_T).block_until_ready()
et = time.time()

duration = et-st
fps = iterations/duration
matmul_flops = 2 * (size**3)
TFLOPS = fps*matmul_flops/(1e12)

print("| JIT MatMul |", TFLOPS, "|")

time.sleep(1)

batched_x = random.normal(key, (iterations, size, size))

@jit
def vmap_batched_apply_matrix(v_batched):
    return vmap(apply_matrix)(v_batched)

vmap_batched_apply_matrix(batched_x).block_until_ready()

time.sleep(1)

st = time.time()
vmap_batched_apply_matrix(batched_x).block_until_ready()
et = time.time()

duration = et-st
fps = iterations/duration
matmul_flops = 2 * (size**3)
TFLOPS = fps*matmul_flops/(1e12)

print("| JIT+VMAP MatMul |", TFLOPS, "|")
