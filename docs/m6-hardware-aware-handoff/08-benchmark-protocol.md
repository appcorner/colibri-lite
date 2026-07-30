# Benchmark Protocol

Record runtime commit, artifact hash, plan, profile hash, Windows version,
CPU/GPU, RAM, VRAM, storage path, fixture, context, repetitions, and cache
state. Report median and dispersion for prompt/decode tokens/s and TTFT.

Each run records RAM/VRAM peak, process working set, cache hit rate, logical
and physical bytes read, and bytes/token. Cold and warm conditions are distinct
experiments. No result from one timing sample, uncontrolled page cache, or a
different fixture may establish a promotion decision.
