---
name: benchmark-model
description: Measure the numbers in DeepSeek V4.1 Flash's BENCHMARKS.md on Strix Halo gfx1151 over HTTP with the production build of this extension, render the tables, and record provenance.
metadata:
  origin: gufo benchmark-model skill, adapted to the deepseek41 extension
---

# Benchmark DeepSeek V4.1 Flash

Fill or refresh `docs/models/deepseek-v4.1-flash/BENCHMARKS.md`. Read it,
`QUALITY.md` (its "Benchmark card" section holds the method) and the latest
`EXPERIMENTS.md` entries first. Follow the user's machine, time and Git
instructions.

The card is Gufo only: no other engine runs DeepSeek V4.1 Flash on this host,
so tables carry no reference columns and no `Gain`.

## Start

Establish the variants, tables, rows and time budget from the conversation
before asking. Prefer one warmed sample per point, run only the requested
sweeps, and repeat only to investigate a concrete discrepancy.

Check the host before the first run and keep it unchanged through the sweep:

- `tuned-adm active` is `accelerator-performance` (activating the desktop's
  power profile switches it back to `balanced`).
- The fans are under PWM control, not the firmware fallback.
- GPU clocks are not pinned (DPM level `auto`).
- Nothing else uses the GPU, and no build runs.

Record the starting Tctl. Start long runs in the background with output in a
log, check progress at short intervals, and do not leave a run unreviewed.
Loading needs cold model files: a privileged page-cache drop
(`sync; echo 3 > /proc/sys/vm/drop_caches`) before each launch; skip loading
if that is unavailable.

## Driver

Upstream Gufo's `model-bench` requires a reference engine and cannot render
this card. Use the driver of the frozen DeepSeek fork (`franzmoca/gufo` main,
`ab0de16d`), pointed at this repository's documentation:

```sh
nix build                                   # this extension: ./result/bin/gufo
EXT=$PWD
git -C /path/to/gufo-fork worktree add /tmp/ds41-bench ab0de16d
cd /tmp/ds41-bench
rm -r docs/models/deepseek-v4.1-flash
ln -s $EXT/docs/models/deepseek-v4.1-flash docs/models/deepseek-v4.1-flash
git apply $EXT/.agents/skills/benchmark-model/model-bench-lookup.patch
B="nix develop -c python3 tools/bench/model-bench.py --model deepseek-v4.1-flash --gufo $EXT/result/bin/gufo"
$B tables                                   # table ids and existing artifacts
$B render --check                           # must say "up to date" before a refresh
$B --gguf udiq2xxs=/models/UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf \
   --dspark /models/DeepSeek-V4.1-Flash-1017-DSpark-draft-Q2K-Q4K.gguf \
   run --target gufo --table single-ar-udiq2xxs,single-dspark-udiq2xxs --fresh
$B render
```

When done, restore the fork worktree's own directory and driver
(`git checkout -- docs/models/deepseek-v4.1-flash tools`).

`model-bench-lookup.patch` adds the server's prompt-lookup counts (`lookup_n`,
`lookup_n_accepted`) to the driver's draft counts; without it the repetition
rows record an `accepted_per_step` near zero.

`multi-dspark` compares each completion with an isolated AR run stored in
`multi-{mixed,repetition}[-<variant>]-gufo-ar.json`, which no table writes.
When the build's greedy output has changed, refresh that reference first: run
the table with a `--config` copy of `bench.json` whose `multi-dspark` has
`"modes": ["ar"]` and `"concurrency": [1]`, then run it normally.

`artifacts/bench.json` declares the variants: `default` (`IQ1_M`, resident),
`iq2xxs`, `udiq1m` and `udiq2xxs` (each with `--stream-experts`). `--gguf`
takes the first shard of each variant; its Engram file must already sit beside
it as the last shard. The tables are `single-ar`, `single-dspark`, `multi-ar`,
`multi-dspark` and `memory`, per variant (`-<variant>` suffix), and `loading`.

`run` flags: `--table` selects tables; `--todo` measures only rows with a
`TODO` cell; `--fresh` discards an artifact's rows instead of merging into them
(use it for a full refresh, so an interrupted run cannot leave mixed-date
rows); `--repetitions N` reports mean ± sd; `--depths` and `--mode ar|dspark`
restrict a sweep; `--drop-caches "<command>"` resets the page cache before each
loading launch. Every row records its server command. Server logs go to the
fork worktree's ignored `artifacts/model-bench/`. `render` replaces the tables
between `<!-- bench:<id> -->` and `<!-- /bench -->` markers and redraws the
charts under `artifacts/charts/`; the prose around them is maintained by hand.

Request timing comes from the server's `timings` (`prompt_n`, `cache_n`,
`prompt_ms`, `predicted_n`, `predicted_ms`), never the client clock.

## Rules

1. Production binary from `nix build` of this repository. Record by hand in the
   card's prose or `QUALITY.md`: the extension revision, the pinned Gufo
   revision, the binary's SHA-256 prefix, the model files' repository and
   quantization, the Engram table, and the SSD the experts stream from. Hosts,
   prompts, generated text and private paths never enter Git.
2. Start a fresh server per depth sweep or concurrency point. Streamed
   variants start with a cold expert cache sized from the memory free at
   startup, so their rows include the warm-up; say so beside the table and log
   the cache size.
3. Warm once, then time. Add repetitions when two runs differ by more than about
   2% or a regression is suspected. DSpark rows vary more than AR rows, because
   the policy learns draft lengths from step times. A sample that generates fewer
   tokens than requested is not a result. Run model jobs sequentially.
4. Greedy DSpark output must match AR at every depth, and every concurrency
   cohort must match its isolated C1 reference; the driver retains completion
   hashes for this. A mismatch or device fault is not a result. Hashes do not
   prove accuracy against the original model: that is `QUALITY.md`'s oracle.
5. A supported cell without a current measurement is `TODO`, never a stale
   number. Keep measurement dates; do not sum stage medians into a headline.
   When a refresh skips a table, blank its old cells to `TODO` rather than
   leave mixed dates.
6. Retained JSON goes to `docs/models/deepseek-v4.1-flash/artifacts/`; raw
   samples and traces stay out of Git.

## Card format

Sections, in order: single user AR, single user DSpark, multiple users AR,
multiple users DSpark, prompt lookup, native prefill, vision encoder, loading
time, memory occupation. Each table section has one table per variant, resident
`IQ1_M` first, separated by `---`. Put model, quantization and mode in the
first column header with the row dimension (depth, users or workload).
Throughput headers say `tok/s`; loading and memory keep seconds and GiB. Keep
acceptance statistics in artifacts, not in the tables. Prompt lookup, native
prefill (`gufo bench`) and the vision encoder are measured by hand; update
their prose and dates with the numbers.

Memory is the peak device total minus free (`hipMemGetInfo`), AR, no
projector, with the idle value before the server starts. On this unified-memory
host it tracks system-wide use. Streamed variants fill the memory their
sessions leave, minus 2 GiB, with the expert cache, so their rows show the
cache's size rather than a requirement; say so beside the table.

## Finish

Run `render --check`, review the diff of `BENCHMARKS.md` and every changed
artifact, update the dates and provenance in the card's opening paragraph and
`QUALITY.md`, and report which cells changed by more than the run-to-run noise.
