# M1 CPU Benchmarks

To test CPU-based performance on M1 on various compute tasks.

## Benchmarks

Preface: benchmarking is hard, there are many possible configs/compile options etc for things like NumPy, as well as many different tasks. Constructive feedback is wellcome! 

### Numpy

Benchmark a variety of different functions in NumPy.

[Accelerate](https://developer.apple.com/documentation/accelerate) is Apple's high-performance computation library. NumPy can be compiled with support for Accelerate. M1 is almost 2x faster in MatMul, almost certainly because the Accelerate library is dispatching the MatMul to the "secret" AMX coprocessor.

Comparison between NumPy (with Accelerate) vs NumPy (from conda) vs 5600X. Numpy reported configs for each can be found in the [Setup & Configs](https://github.com/tlkh/m1-cpu-benchmarks#setup--configs) section below.

| Task       | Accelerate | Conda | 5600X |
| ---------- | ---------- | ----- | ----- |
| datagen    | 0.348 | 0.385 | 0.472 |
| special    | 0.447 | 0.459 | 0.599 |
| stats      | 1.017 | 1.253 | 0.961 |
| matmul     | 0.301 | 0.602 | 0.509 |
| vecmul     | 0.011 | 0.015 | 0.009 |
| svd        | 0.469 | 1.591 | 0.372 |
| cholesky   | 0.069 | 0.097 | 0.111 |
| eigendecomp| 4.911 | 7.635 | 3.214 |

Benchmark script: `numpy_benchmarks.py`.

Using a MatMul as a proxy to measure the CPU's TFLOPS:

| FP64 | FP32 | FP16 | 
| ---- | ---- | ---- |
| ![](fp64_tflops.png) | ![](fp32_tflops.png) | ![](fp16_tflops.png)

Benchmark script: `numpy_tflops.py`.

### SpaCy

Overall, the 5600X is still faster when running "real" CPU-based models. On M1, Accelerate (via NumPy) doesn't seem to affect anything but the Transformer model, but that seems to use PyTorch. So maybe Accelerate is linked to PyTorch as well? Installing SpaCy's AppleOps which allows SpaCy to directly call Accelerate provides a dramatic performance improvement.

| config | en_core_web_sm | en_core_web_md | en_core_web_lg | en_core_web_trf |
| ------ | -------------- | -------------- | -------------- | --------------- |
| M1 conda              | 3143 | 2899 | 2853 |  309 |
| M1 conda+AppleOps     | 7826 | 7116 | 6208 |  313 |
| M1 Accelerate         | 3191 | 2899 | 2900 | 1064 |
| M1 Accelerate+AppleOps| 7907 | 7029 | 6384 | 1125 |
| 5600X                 | 9580 | 8748 | 8773 |  487 |
| 5600X + MKL           | 9550 | 8765 | 8800 | 1151 |

Benchmark script: `spacy_benchmarks.py`.

### Jax

Simple benchmark, set up to be similar to the NumPy one to measure TFLOPS. Performs similar to conda install of NumPy, hence probably is not aware of the AMX. Functionally, things like JIT, VMAP work, but in our simple test don't give any performance gains. 

| Task           | M1    | 5600X |
| -------------- | ----- | ----- |
| MatMul         | 0.559 | 0.594 |
| JIT MatMul     | 0.558 | 0.593 |
| JIT+VMAP MatMul| 0.546 | 0.419 |

Benchmark script: `jax_benchmarks.py`.

## Setup & Configs

### Env 1: Generic

The goal is to test an out-of-the-box conda install.

1. `conda install numpy spacy`
2. Were you expecting more?

NumPy config:

```
blas_info:
    libraries = ['cblas', 'blas', 'cblas', 'blas']
    library_dirs = ['/Users/tlkh/miniforge3/envs/py-vanilla/lib']
    include_dirs = ['/Users/tlkh/miniforge3/envs/py-vanilla/include']
    language = c
    define_macros = [('HAVE_CBLAS', None)]
blas_opt_info:
    define_macros = [('NO_ATLAS_INFO', 1), ('HAVE_CBLAS', None)]
    libraries = ['cblas', 'blas', 'cblas', 'blas']
    library_dirs = ['/Users/tlkh/miniforge3/envs/py-vanilla/lib']
    include_dirs = ['/Users/tlkh/miniforge3/envs/py-vanilla/include']
    language = c
lapack_info:
    libraries = ['lapack', 'blas', 'lapack', 'blas']
    library_dirs = ['/Users/tlkh/miniforge3/envs/py-vanilla/lib']
    language = f77
lapack_opt_info:
    libraries = ['lapack', 'blas', 'lapack', 'blas', 'cblas', 'blas', 'cblas', 'blas']
    library_dirs = ['/Users/tlkh/miniforge3/envs/py-vanilla/lib']
    language = c
    define_macros = [('NO_ATLAS_INFO', 1), ('HAVE_CBLAS', None)]
    include_dirs = ['/Users/tlkh/miniforge3/envs/py-vanilla/include']
Supported SIMD extensions in this NumPy install:
    baseline = NEON,NEON_FP16,NEON_VFPV4,ASIMD
    found = 
    not found = ASIMDHP,ASIMDDP
```

### Env 2: Accelerate 

1. Install all relevant packages and dependencies for building NumPy and SpaCy: `conda install numpy spacy transformers pytest hypothesis cython`. We're going to build NumPy and SpaCy from source, but when we install from conda we automatically get all the dependencies to make building from source easier.
2. `pip uninstall numpy -y`
3. `git clone https://github.com/numpy/numpy`
4. `git checkout maintenance/1.21.x`
5. `python setup.py build_ext --inplace -j 10` With 10 threads for compile (feel free to adjust this), my M1 Max finishes in less than 30 seconds. 
6. Run tests: `python runtests.py -v -m full`, and you should get an output similar to `15457 passed, 213 skipped, 23 xfailed in 170.07s` aka the test suite passes.
7. `pip install .`

Now we can check the NumPy config available to us. For mine, it looks like this (relevant section shown):

```
accelerate_info:
    extra_compile_args = ['-I/System/Library/Frameworks/vecLib.framework/Headers']
    extra_link_args = ['-Wl,-framework', '-Wl,Accelerate']
    define_macros = [('NO_ATLAS_INFO', 3), ('HAVE_CBLAS', None)]
blas_opt_info:
    extra_compile_args = ['-I/System/Library/Frameworks/vecLib.framework/Headers']
    extra_link_args = ['-Wl,-framework', '-Wl,Accelerate']
    define_macros = [('NO_ATLAS_INFO', 3), ('HAVE_CBLAS', None)]
lapack_opt_info:
    extra_compile_args = ['-I/System/Library/Frameworks/vecLib.framework/Headers']
    extra_link_args = ['-Wl,-framework', '-Wl,Accelerate']
    define_macros = [('NO_ATLAS_INFO', 3), ('HAVE_CBLAS', None)]
Supported SIMD extensions in this NumPy install:
    baseline = NEON,NEON_FP16,NEON_VFPV4,ASIMD
    found = ASIMDHP,ASIMDDP
    not found = 
```

From here, we can run our NumPy and SpaCy benchmarks that leverage NumPy.

### SpaCy

Note that as of 3.1.4, SpaCy can optionally leverage Accelerate directly! To use it (note: this will replace the SpaCy install, so you might want to set up another environemnt):

```
pip uninstall spacy -y 
pip install 'spacy[apple]'
```

### Jax

1. `conda install pip numpy scipy`
2. `pip install -U https://storage.googleapis.com/jax-releases/mac/jaxlib-0.1.74-cp39-none-macosx_11_0_arm64.whl`
3. `pip install jax`

Still very early days for Jax on ARM/M1, issue being tracked [here](https://github.com/google/jax/issues/5501).

### Reference 5600X

I used the NumPy from NGC PyTorch container, which should be reasonably optimized.

NumPy config:

```
blas_info:
    libraries = ['cblas', 'blas', 'cblas', 'blas']
    library_dirs = ['/opt/conda/lib']
    include_dirs = ['/opt/conda/include']
    language = c
    define_macros = [('HAVE_CBLAS', None)]
blas_opt_info:
    define_macros = [('NO_ATLAS_INFO', 1), ('HAVE_CBLAS', None)]
    libraries = ['cblas', 'blas', 'cblas', 'blas']
    library_dirs = ['/opt/conda/lib']
    include_dirs = ['/opt/conda/include']
    language = c
lapack_info:
    libraries = ['lapack', 'blas', 'lapack', 'blas']
    library_dirs = ['/opt/conda/lib']
    language = f77
lapack_opt_info:
    libraries = ['lapack', 'blas', 'lapack', 'blas', 'cblas', 'blas', 'cblas', 'blas']
    library_dirs = ['/opt/conda/lib']
    language = c
    define_macros = [('NO_ATLAS_INFO', 1), ('HAVE_CBLAS', None)]
    include_dirs = ['/opt/conda/include']
Supported SIMD extensions in this NumPy install:
    baseline = SSE,SSE2,SSE3
    found = SSSE3,SSE41,POPCNT,SSE42,AVX,F16C,FMA3,AVX2
    not found = AVX512F,AVX512CD,AVX512_KNL,AVX512_KNM,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL
```

For NumPy, no diff observed with `MKL_DEBUG_CPU_TYPE=5` flag.
