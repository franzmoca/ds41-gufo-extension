---
name: optimize-kernel
description: Optimize DeepSeek V4.1 Flash inference on Strix Halo gfx1151 using focused profiles, independent quality checks, and matched end-to-end measurements.
metadata:
  origin: gufo optimize-kernel skill, adapted to the deepseek41 extension
---

# Optimize gfx1151 kernels

Read `docs/models/deepseek-v4.1-flash/{BENCHMARKS,QUALITY,EXPERIMENTS,BACKLOG}.md`,
the analysis page for the path you change (`DECODE.md`, `PREFILL.md` or
`STREAMING.md`) and its launch code. Search `EXPERIMENTS.md` for the mechanism
before trying it: many ideas were already measured and rejected. Establish the
timed scope and arithmetic contract before changing dispatch. Follow the user's
machine, time and Git instructions.

## Fast experiment loop

1. Keep baseline and candidate production binaries (`nix build`, same
   `flake.lock`). `gpu-test` binaries keep assertions and are for correctness,
   not headline timings.
2. Start with one affected workload and one control on streamed `UD-IQ2_XXS`
   (`MXFP8` Engram, DSpark, vision): a 2K and a 24K prompt for prefill, and
   512-token chat, code and long-document replies for decode. Tiny prompts and
   outputs suffice to debug launches.
3. Profile separately with Gufo's `tools/prof/prof.py` (`PREFILL.md` has the
   commands); compare stage totals, launch counts, GPU-busy time and wall span.
   `GUFO_DEEPSEEK41_DSPARK_TRACE` reports each drafting cycle's kind and time.
   When GPU time does not explain latency, look at expert reads,
   synchronization, sampling and cache I/O. Exclude model loading and
   expert-cache warm-up from request-time utilization.
4. Change one mechanism, compile the affected target, then run its existing
   test (`deepseek41flash.<name>`). Use `tools/bench/build.sh <name>` for
   standalone HIP experiments; each result needs correctness alongside
   throughput.
5. Compare unprofiled baseline/candidate under identical model, prompt, depth,
   sampling, cache history and concurrency, in alternating order (ABBA) on a
   fresh server per arm. Repeat only to resolve noise. A microbenchmark gain
   must survive the full inference path.
6. Keep the winning implementation as default. Delete rejected/dead routes;
   do not add environment switches or duplicate tests to preserve experiments.
   Record a retained/rejected `E<n>` section and index row in `EXPERIMENTS.md`,
   update `BACKLOG.md`, current speeds in `BENCHMARKS.md`, quality evidence in
   `QUALITY.md`, and add a `CHANGELOG.md` entry for user-facing gains.

## What decides speed here

- **Decode is bandwidth and launch bound.** Tabulate each kernel's achieved
  bandwidth against the measured ~240 GB/s read ceiling
  (`tools/bench/gfx1151_peak.hip` in Gufo); kernels at 150-225 GB/s still have
  headroom (E102-E103). A dependent decode launch costs a measured, untraced
  gap (E101): count launches before fusing.
- **Streamed experts are SSD reads.** A miss reads the expert from disk into
  unified memory, so running missed experts on the CPU does not help. Measure
  MiB read per token alongside time; compare prefetch and eviction rules
  offline on a `DS41_EXPERT_TRACE` trace before building them. Every byte of
  fixed GPU allocation (prefetch stage, transfer streams, scratch) comes out
  of the expert cache, which is sized from the memory left at startup.
- **DSpark decode varies between runs.** The policy learns draft lengths from
  measured step times. Compare milliseconds per cycle of each cycle kind and
  completed tokens per second, not acceptance fractions; check that draft
  counts match across arms.
- **Bounded replay.** Prefill runs the decoder layers over each prompt's last
  140 tokens; prefill changes must keep `--exact-prefill` working and
  measured.

## Techniques that worked in Gufo

- **Bandwidth and cache:** the 32 MiB MALL can make repeatedly reused small
  matrices look unrealistically fast: rotate weights beyond cache capacity or
  reproduce full-model traffic.
  The same cache is a lever: a batched kernel that sweeps a matrix once per
  token tile can order its workgroups so a chunk of rows stays cached
  (E108). An LDS read costs about 30 vector instructions and a matrix-core
  product about 60; count operand reads per product (E109).
- **GEMV:** coalesce packed weight loads, stage shared activations in LDS,
  fuse reads where useful, and preserve each dot product's accumulation order.
  More fusion can increase VGPR pressure and spills. Integer WMMA helped wider
  Q8 verification only after keeping the original reduction tree. Repeat
  identical inputs at different tile columns: fast-math gave different
  contraction until the product/FMA order was explicit.
- **Verification and concurrency:** reuse quantized weights across token and
  request rows. Tune real ragged shapes, including 3-8 rows and partial tiles.
  Keep each request's KV, acceptance and RNG state independent. Test both shared
  and disjoint expert routing: weight reuse benefits shared experts, but
  single-request experts need a compact path. Qualify draft controllers on
  ordinary and perfect-acceptance prompts. A bounded grid can consume a compact
  device list without downloading its count.
- **Wave mode and compiler:** match helper/caller wave modes per translation
  unit. Inspect VGPR/LDS/private scratch and ISA with Gufo's
  `tools/prof/isa_mix.py`; occupancy alone is not the objective. LLVM's
  real-true16 mode slowed the FP16 routed kernels; their device-only
  `target("no-real-true16")` pragma must stay. Verify generated register use and
  full-model time after scheduling changes.
- **Attention and selection:** exact partial top-k can avoid sorting the full
  context; preserve tie ordering and FP32 ranking. Do not change softmax
  reduction order without numerical qualification. Prefetching V rows before
  their original sequential FMAs hid load latency in draft attention.
- **Fusion and memory:** reuse scratch, allocate rollback depth on demand and
  avoid carrying prefill chunks into draft catch-up. Match production
  allocation in microbenchmarks: gains on `hipMalloc` buffers can disappear with
  mapped weights. Size snapshots by valid KV rows, not context capacity. Finish
  a norm's existing sum tree within one wave to remove barriers without
  reordering additions. Count library reservations: each rocBLAS handle kept
  32 MiB. Reuse tensors after their last read.
- **Graphs:** verify replay actually launches every new operation; side-stream
  work during capture is not automatically included.

## Quality and reporting

Use an independent reference: `tools/ds41/reference.py` (CPU oracle over the
GGUF weights), `dspark_reference.py` and `vision_reference.py`. Two Gufo paths
agreeing does not prove upstream parity. Compare matched token histories and
find the first changed layer if logits drift. Never relax tolerances or replace
goldens to accept a speedup. Test awkward tails (1/8/9/32/33 rows), not only
aligned shapes, and end ragged inputs at the allocation boundary. Moving FP32
expressions into a device helper changed contraction in Gufo.

For state or speculative changes, cover greedy output, sampled rejection and
residual correction, seeded replay, EOS, multi-turn continuation and snapshot
restore. Add images and cancellation when those paths change. Timing-based
controllers are for greedy decoding; sampled execution must retain seeded
replay. Full model sweeps are final qualification, not each iteration.

Report per-request timings from the server's `timings`, with the run order and
every arm's result; state whether replies and draft counts matched.
