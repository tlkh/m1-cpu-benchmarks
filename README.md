# m1-cpu-benchmarks

M1 CPU Benchmarks

## Goals

To test CPU-based performance on M1 covering the following areas:

* Synthetics (including NumPy)
* SpaCy

## Setup

### Env 1: Generic

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

Note that as of 3.1.4, SpaCy can optionally leverage Accelerate as well! To use it (note: this will replace the SpaCy install, so you might want to set up another environemnt):

```
pip uninstall spacy -y 
pip install 'spacy[apple]'
```

## Benchmarks

### Numpy

| Task       | Accelerate | Conda |
| datagen    | 0.348 | 0.385 |
| special    | 0.447 | 0.459 |
| stats      | 1.017 | 1.253 |
| matmul     | 0.301 | 0.602 |
| vecmul     | 0.011 | 0.015 |
| svd        | 0.469 | 1.591 |
| cholesky   | 0.069 | 0.097 |
| eigendecomp| 4.911 | 7.635 |



### SpaCy

Numpy from Conda:

en_core_web_sm - token/sec: 3143
en_core_web_md - token/sec: 2899
en_core_web_lg - token/sec: 2853
en_core_web_trf - token/sec: 309

Numpy from Conda + SpaCy+AppleOps:

en_core_web_sm - token/sec: 7826
en_core_web_md - token/sec: 7116
en_core_web_lg - token/sec: 6208
en_core_web_trf - token/sec: 313

With NumPy+Accelerate:

en_core_web_sm - token/sec: 3191
en_core_web_md - token/sec: 2899
en_core_web_lg - token/sec: 2900
en_core_web_trf - token/sec: 1064

With NumPy+Accelerate + SpaCy+AppleOps:

en_core_web_sm - token/sec: 7907
en_core_web_md - token/sec: 7029
en_core_web_lg - token/sec: 6384
en_core_web_trf - token/sec: 1125

