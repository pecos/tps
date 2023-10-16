This directory house various Docker recipes to support testing, development,
and execution of TPS on various systems.

* `test/` -> container used for CI testing
* `test-cpu-cupy/` -> TPS enviroment with CPU-based builts of HYPRE/MFEM, and CUPY
* `test-gpu/` -> TPS enviroment with GPU-based builts of HYPRE/MFEM (different builds ), and CUPY
* `test-cuda-base/` -> An image with common dependencies from which `test-cpu-cupy/` and `test-gpu/` are built from.
* `quartz/` -> container with MV2/PSM targeting execution on Quartz system

Pre-built images are available from: [https://hub.docker.com/u/pecosut](https://hub.docker.com/u/pecosut).
