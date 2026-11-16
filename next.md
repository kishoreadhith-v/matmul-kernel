# Roadmap

## Current state

F1 and F2 are done.

- `cublas` is registry entry 0; `naive` is entry 1.
- 1024²: cuBLAS ≈ 2.8 TFLOPS, naive ≈ 35 GFLOPS.
- Correctness is checked against cuBLAS with `atol=1e-2`, `rtol=1e-3`.
- `--sizes`, `--shape`, `--alpha`, `--beta`, `--verify-only`, and
  `--test-suite` work; CSV appends and records the actual alpha/beta.
- `--test-suite` passes all 23 shapes for both kernels, plus a `beta=0` pass at
  `(1000, 999, 1001)`.
- `--verify-only` and `--test-suite` do not time. `--no-verify` reports `SKIP`
  and writes `passed=unknown` instead of a false `PASS`.

Current step: **F3 — registry and kernel variants**, or **kernel 2
(coalesced)** if you would rather start optimizing first.

The goal of the framework is simple: adding a kernel should be a new
`src/kernels/*.cu` file plus one registry line. Everything else — allocation,
initialization, reference, verification, timing, CSV, profiling — belongs to the
harness.

## Ground rules

1. A kernel is a pure function of buffers and scalars.
2. The harness owns allocation, initialization, validation, timing, and reporting.
3. Verify before timing.
4. Benchmark with `beta = 0`; test correctness with `beta != 0`.
5. Test non-multiples and non-square shapes, not only powers of two.
6. Record enough metadata that a result can be reproduced.
7. Do not build infrastructure for kernels you have not written yet.

## F2 — cuBLAS baseline, shapes, and a real test suite

### Goal

Make the framework useful for more than one square size, and add cuBLAS as
registry entry 0 so every benchmark has a reference line.

Status: **done**.

### 2.1 Add the cuBLAS baseline

### 2.2 Add `--kernel all`

### 2.3 Add shapes and sizes

Add:

```text
--sizes 128,256,512,1024
--shape M,N,K
--verify-only
--alpha <float>
--beta <float>
```

Internal changes:

- `BenchOptions` gets `alpha` and `beta`.
- `bench_kernel` uses those instead of the hardcoded `1.0f, 0.0f`.
- CSV writes the actual alpha and beta.
- `--verify-only` runs the reference and comparison but no warmup/timing.
- `--sizes` loops over square sizes.
- `--shape` runs one non-square problem.

Benchmark policy:

- `beta = 0` for timing.
- Correctness tests `beta = 0` and one run with `beta = 0.5`.
- Correctness tests `alpha = 0.5` at least once.

### 2.4 Add the test suite

Add `--test-suite`, which verifies every registered kernel at:

```text
1, 2, 3, 7, 16, 31, 32, 33, 63, 64, 65,
127, 128, 129, 255, 256, 257, 511, 512, 1000
```

plus non-square shapes:

```text
(17, 19, 23)
(128, 256, 64)
(1000, 1000, 1000)
```

Acceptance (met):

- [x] Every kernel passes all shapes, or fails with index, expected value, and actual value.
- [x] `--kernel all --sizes 128,256,512,1024 --csv results/f2.csv` writes one row per kernel per size.
- [x] `make run` includes cuBLAS and naive.
- [x] `--verify-only` and `--test-suite` run without warmup or timing.
- [x] CSV appends across runs and records the actual alpha and beta.
- [x] `--test-suite` also covers `beta=0` at `(1000, 999, 1001)`.

## F3 — Registry and kernel variants

### Goal

Make kernel families and autotuning variants first-class without touching the
measurement code.

### Changes

- Move the registry from `main.cu` to `src/dispatch.cu`.
- Add `--list-kernels` details: name, config, metric.
- Allow multiple variants of one family:

```text
cublas
naive
tiled2d_64x64x8_t8x8
tiled2d_128x128x8_t8x8
tiled2d_128x128x16_t8x8
```

- Keep an explicit registry. Do not use static-initialization magic across
  translation units.
- Add `--kernel all` and, optionally, `--family tiled2d`.

### Acceptance

- Adding a kernel requires one `.cu` file, one CMake source line, and one
  registry line.
- `--list-kernels` shows every variant.
- `--kernel all` runs every variant and writes one CSV row per variant.

## F4 — Real timing statistics

### Goal

Make small differences measurable and reproducible.

### Changes

- Replace the single timing batch with `trials` batches of `inner_reps` launches.
- Report `min`, `median`, `mean`, and `std` of the batch averages.
- Add `--target-ms` to auto-size repetitions for a target measurement time.
- Add `--cache hot|cold`:
  - hot: back-to-back launches;
  - cold: flush L2 with a 256 MB `cudaMemsetAsync` before each timed batch,
    outside the measured region.
- Record `nvidia-smi` clocks and temperature in a metadata file.
- Record `git rev-parse HEAD` in the run metadata.
- Move `cudaGetLastError()` out of the timed launch loop; check it after the
  batch instead.

### Acceptance

- Two identical runs agree within about 2–3%.
- The CSV has useful `min`, `median`, and `std`, not three identical numbers.
- You can see whether 128² results are cache-hot or cold.

## F5 — Automation and reporting

### Goal

One command produces a complete, reproducible result set.

### Changes

- `scripts/bench.sh`:
  - create `results/<run_id>/`;
  - write `metadata.json` or `metadata.txt` with GPU, driver, CUDA version,
    git commit, clocks, temperature, and build type;
  - run `--kernel all --sizes ... --csv results/<run_id>/measurements.csv`;
  - call the plot script.
- `scripts/plot.py`:
  - read the CSV;
  - plot GFLOPS vs size, one line per kernel;
  - add a horizontal cuBLAS line and a roofline hint;
  - generate the README table.
- Ignore `results/` in git. Commit curated summaries or plots separately if
  desired.
- Add a `make bench` target that calls `scripts/bench.sh`.

### Acceptance

- `make bench` produces CSV, plot, and README table.
- The plot clearly shows cuBLAS at the top and the kernel ladder below it.
- Raw profiler reports and per-run CSVs are not committed.

## F6 — Profiling workflow

### Goal

Answer “what is the limiter?” for any kernel without fighting the tooling.

### One-time permission fix

`ncu` currently fails with `ERR_NVGPUCTRPERM` on this machine. Test with:

```bash
sudo ncu --set default --launch-count 1 ./build/matmul --kernel naive --size 1024
```

If that works, make it permanent:

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | \
  sudo tee /etc/modprobe.d/nvidia-profiler.conf
sudo update-initramfs -u
# reboot
```

### Changes

- Add `--profile`:
  - skip timing and verification;
  - run exactly one warmup and one measured launch;
  - or run one launch only, whichever is simpler to target with `ncu`.
- Add `scripts/profile.sh KERNEL SIZE`:

```bash
ncu --set default --launch-count 1 \
    --export "results/profiles/${1}_${2}" \
    ./build/matmul --kernel "$1" --size "$2" --profile
```

- Use `--set detailed` for memory/occupancy work and `--set full` only when
  necessary.
- Add a `make profile KERNEL=naive SIZE=1024` target.
- Use `nsys` for timeline questions, not microarchitecture.
- Use `cuobjdump -res-usage` and `cuobjdump -sass` for static inspection.

### What to read in `ncu`

| Section | Question |
|---|---|
| SpeedOfLight | compute roof or memory roof? |
| LaunchStats | grid, block, registers, shared memory, occupancy |
| Occupancy | theoretical vs achieved; what limits it? |
| MemoryWorkloadAnalysis | global efficiency, sectors/request, L1/L2 hits, bank conflicts |
| ComputeWorkloadAnalysis | pipe utilization and instruction mix |
| SchedulerStats / WarpStateStats | why warps stall |
| SourceCounters | which source line causes the stall |

### Acceptance

- `make profile` produces a `.ncu-rep` for one launch.
- You can write one paragraph naming the limiter for naive and for one tiled
  kernel.
- You know that `ncu` timings are not benchmark timings.

## F7 — Autotuning (optional but recommended)

### Goal

Find the best tile configuration on sm_75, not on an A6000.

### Changes

- Register multiple variants of one kernel family with different `config`
  strings.
- Add a sweep mode, or drive the sweep from a shell/Python script that calls
  `--kernel <variant> --size <n> --csv ...`.
- Keep a table of `family, config, size, gflops`.
- Pick the best config per size and record it in the README.

Do not build a generic search engine. A table of variants plus a loop is enough.

### Acceptance

- The best configuration for 1024 and 4096 is known on your GPU.
- Re-running the sweep is one command.
- The winning config is recorded in the repo.

## Kernel ladder

The framework stays fixed; each kernel is a new file plus a launch wrapper.

| # | Kernel | Concept | What to inspect |
|---:|---|---|---|
| 1 | Naive (done) | one thread per output, no reuse | global load efficiency |
| 2 | Coalesced | map threads to contiguous C/B | sectors/request, L1/L2 hits |
| 3 | Shared-memory tiling | reuse A/B tiles | bank conflicts, occupancy |
| 4 | 1D block tiling | several outputs per thread | register pressure, ILP |
| 5 | 2D block tiling | 2D register tile | registers vs occupancy |
| 6 | Vectorized access | `float4` loads/stores | 128-bit transactions |
| 7 | Bank conflicts (linearize) | remove shared-memory conflicts | shared wavefronts |
| 8 | Bank conflicts (padding) | `BK + 1` padding | shared wavefronts |
| 9 | Autotuned | search BM/BN/BK/TM/TN | best config on sm_75 |
| 10 | Warp tiling | warp-level tiles | reuse and register pressure |
| 11 | Double buffering | overlap load and compute | stall reasons |
| 12 | Split-K (optional) | parallelize over K | workspace, reduction cost |
| 13 | FP16 tensor cores (separate) | different precision | not an FP32 SGEMM result |

Rough relative expectations on the 1650 SUPER, versus cuBLAS:

```text
naive            1–3%
coalesced        8–15%
shared memory    20–35%
1D tiling        45–60%
2D tiling        65–80%
vectorized       75–85%
warp tiling      85–95%
double buffering 90–100%
```

These are sanity checks, not promises. The reference repository's tile
parameters are tuned for an A6000; retune them for sm_75.

## Will the current framework reach the most optimized SGEMM?

Mostly yes. The `GemmProblem` and `KernelLaunch` contracts can stay stable for
the whole FP32 ladder. Each kernel is:

1. a new `src/kernels/*.cu`;
2. a launch wrapper;
3. one CMake source line;
4. one registry entry.

There are four things the current framework does **not** yet handle:

1. **Workspace for split-K.** If a kernel needs scratch memory, add a
   `float* workspace` and `size_t workspace_bytes` to `GemmProblem`, and let the
   harness allocate it. Do this only when you write split-K.
2. **Multiple variants and autotuning.** The registry can already hold multiple
   entries. A sweep mode or script is the missing piece; no kernel-interface
   change.
3. **Reliable timing for 1–2% differences.** F4 is required before late-stage
   tuning, otherwise clock noise dominates.
4. **Profiling.** F6 is required to know why a kernel is slow.

Turing-specific constraints:

- No TF32 and no BF16 tensor cores. FP32 SGEMM is a SIMT FMA problem.
- No `cp.async`; double buffering uses ordinary loads plus `__syncthreads`.
- FP16 tensor cores exist, but that is a different precision and a different
  baseline. Do not mix it into the FP32 SGEMM ladder.
- 4 GB of VRAM: 4096² is comfortable; 8192² with four buffers is about 1 GiB
  and possible, but tight with the display.

So: yes for the main ladder. Expect to add workspace for split-K and a variant
sweep near the end. Expect F4 and F6 to be necessary for the last 10%.

## Definition of done for the framework

- [x] `make` configures and builds from a clean tree.
- [x] `make run` runs cuBLAS plus every registered kernel.
- [ ] `make bench` produces CSV, plot, and README table. (F5)
- [ ] `make profile KERNEL=... SIZE=...` produces one `.ncu-rep`. (F6)
- [x] Adding a kernel is one `.cu`, one CMake line, one registry line.
- [x] Every kernel passes `--test-suite`, including non-multiples and non-square shapes.
- [ ] Timing is stable enough to distinguish a 5% improvement. (F4)
- [ ] Results include git commit, GPU, driver, and clocks. (F4/F5)
- [x] Raw results and profiler reports are gitignored.
- [ ] The framework is roughly 700 lines or less. If it grows past that before
      three kernels exist, cut features.

## Do not build yet

- plugin loading or shared-library kernels;
- YAML/JSON experiment configs;
- a database;
- a web dashboard;
- multi-GPU or multi-stream scheduling;
- CUDA graphs;
- a custom memory pool;
- dtype templates or a generic tensor class;
- a full autotuner;
- a unit-test framework;
- GPU CI.

Add each only when a real kernel or a real measurement problem forces it.

## Suggested order

1. F2.1–2.2: cuBLAS baseline and `--kernel all`. **Done.**
2. F2.3–2.4: shapes, sizes, alpha/beta, test suite, CSV append. **Done.**
3. F3: registry/variants. **← current**
4. F4: trials and statistics.
5. K2–K5: coalescing, shared memory, 1D/2D tiling.
6. F5: bench script and plots.
7. F6: profiling.
8. K6–K11: vectorization, bank conflicts, autotuning, warp tiling, double buffering.
9. F7: autotuning sweep.
10. Optional: split-K and FP16 tensor cores as separate tracks.

## Cheat sheet

Current (F2):

```bash
make
make list
make run
make run KERNEL=naive SIZE=2048 REPS=20
make sizes
make shape
make verify
make test
make csv CSV=results/f2.csv
./build/matmul --kernel all --sizes 128,256,512,1024 --reps 50 --csv results/f2.csv
./build/matmul --kernel naive --shape 128,256,64 --reps 20
./build/matmul --kernel all --test-suite
```

After F4:

```bash
./build/matmul --kernel all --sizes 1024,2048,4096 \
  --warmup 10 --trials 10 --target-ms 100 --cache cold \
  --csv results/f4.csv
```

After F6:

```bash
make profile KERNEL=naive SIZE=1024
ncu --import results/profiles/naive_1024.ncu-rep
```
