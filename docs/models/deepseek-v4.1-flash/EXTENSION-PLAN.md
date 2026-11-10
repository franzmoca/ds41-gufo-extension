# DeepSeek V4.1 Flash as a Gufo extension: migration plan

Status: done. The extension is published as
[franzmoca/ds41-gufo-extension](https://github.com/franzmoca/ds41-gufo-extension)
and builds against upstream Gufo `main` (2026-11-07). On 2026-11-08 the
conversation, tools, long-context, cache (disk restart included), image and
reasoning-budget suites, the Pi agent replay, `gufo prompt`, `gufo bench` and
every listed build passed. The fork no longer receives DeepSeek V4.1 changes.
The decisions and port status below are the record of 2026-11-06; the expert
streaming library is still open.
Covers the `deepseek41` backbones
(ISTA IQ1_M and IQ2_XXS, Unsloth UD-IQ1_M and UD-IQ2_XXS, ISTA Coder), the DSpark
drafters, the vision sidecar and SSD expert streaming.

## Goal

Move DeepSeek V4.1 Flash off the `franzmoca/gufo` fork into an independent source
extension, `franzmoca/deepseek41flash-gufo-extension`, built against a pinned Gufo
with no model code in core. After the extension is published, the expert cache and
SSD streaming become a separate library that later extensions can reuse (see
[Expert streaming library](#expert-streaming-library)).

Structure follows `franzmoca/gemma4-gufo-extension`.

## Decisions

| Topic | Decision | State |
| --- | --- | --- |
| Repository | `franzmoca/deepseek41flash-gufo-extension` | Proposed |
| Gufo pin during the port | Fork `gemma-main` `98c4547b` (upstream plus #1046 and #1047), API 0.2.9; upstream revision once both merge | Proposed |
| Parity baseline | `feat/ds41-coder` `123592ce` (`feat/ds41` plus the Coder) | Proposed |
| Coder | In the first version | Agreed |
| LoRA and direction adapters | After the first version; need only #1046 | Done (E98) |
| Expert cache and SSD streaming | One package, kept inside the extension until the extension is published; a separate library comes later | Agreed |
| Upstream PRs | Opened only with measurements from the port | Agreed |
| Fork after switch-over | `main` frozen with a warning that redirects to both extensions; `upstream-main` kept for upstream work | Agreed; timing open |

Fork `main` keeps serving DS4.1 until switch-over. Whether that waits for MXFP8
and UD-IQ1_M is decided later.

## Port status

Local repository `~/projects/deepseek41flash-gufo-extension`, built against the
`gemma-main` pin. It holds only compiled files: 59 of the package's 68, 2.2 MiB,
no Qwen kernels. DS4.1 used one Qwen file, the hipBLASLt wrapper of its vision
encoder, now a small `vision/lt_gemm` with the same heuristic plan path.

Smoke checks on 2026-11-06, Coder with the IQ4_NL Engram and the Q2K DSpark
drafter, against the fork at `123592ce`:

| Check | Result |
| --- | --- |
| 13 HTTP requests: greedy, sampled, logprobs, tool calls and results, required tool, 2,447-token prompt, two concurrent requests, one image | Replies byte-identical in 4 runs (fork, extension, extension, fork) |
| Timing, same runs | On par, except the 2,447-token first prefill: 2,093 ms vs 1,906 ms (U3) |
| Disk cache restart | Restores 2,444 of 2,447 tokens in about 7 ms. Same replies as the fork after restart. Sampled requests change across a restart on both builds; greedy ones do not |
| `--stream-experts` on IQ2_XXS with a disk cache | Loads, serves the same requests with sensible replies, 95.8-97.7% route hits; not compared with the fork |
| `gufo prompt`, `chat`, `bench` with `--prefill-batch` | Run, C1 and C2 |
| 9 moved CTest targets, including 6 GPU kernel checks | Pass, none skipped |

Not yet done: the phase 0 baseline and parity sweep, oracle PPL, functional suites,
`pi_agent.py` replay, IQ1_M and UD-IQ2_XXS, `tools/`, cards, and a build with the
in-tree V4 provider serving next to this one.

## What upstream main can already do

Tested with an upstream main `5d2bf02f` build, `gufo prompt`, on the real shards:

| Files | Result |
| --- | --- |
| ISTA IQ1_M, UD-IQ2_XXS or Coder with the IQ4_NL Engram | Opens; fails only on the missing provider |
| Any quant with the MXFP8 Engram | `unsupported storage type: per_layer_token_embd.weight` (type 43) |
| UD-IQ1_M with the IQ4_NL Engram | `Inconsistent split GGUF shard metadata` (5 shards; the Engram says 6) |

Merged and usable: provider sampling defaults (top-p 1), native tool dialects
(the V4.1 DSML spelling is in `json_constraint_test`), block formats, shared
provider options (`--dspark-model`), public link targets, reasoning effort,
`--mmproj`, cache adapter contract.

## Upstream PRs

### Required

| # | PR | Without it | State |
| --- | --- | --- | --- |
| U1 | #1046 provider options (`ext-api`) | No `--stream-experts`, `--expert-cache-bytes`, `--prefill-batch`, `--exact-prefill`; streamed quants cannot run | Open, awaiting `ext-api` |

### Wanted, not blocking

| # | Change | What it does | Temporary state | Evidence still needed |
| --- | --- | --- | --- | --- |
| U2 | Open GGUFs the reader rejects: #966 (`claim_file`, other author, open, unreviewed), or a reader change | Lets MXFP8 Engrams and UD-IQ1_M load | IQ4_NL only. UD-IQ1_M: patched split metadata on an Engram copy (`set_split.py`). MXFP8: placeholder last shard plus a shard 1 copy with one tensor fewer, Engram opened by our parser (untested) | Whether #966 fits our probe, identity and drivers. Avoid adding type 43 to core: Unsloth's meaning clashes with upstream ggml's |
| U3 | Replay hooks: `TextModelRunner::CanContinue` and the whole prompt's end during prefill (`984ad880`) | Runs the 140-row decoder replay once per prompt instead of at every checkpoint | Replay at every checkpoint, the behaviour before `984ad880`. Without a PR, a prompt context could carry the prompt end and `Adapter::Validate` could refuse; the disk restore path is unread, so a good checkpoint may be invalidated | Measured on one 2,447-token first prefill: 2,081 ms per checkpoint vs 1,908 ms with a whole-prompt end forced in a probe build (-8.3%), same output, balanced order. The prompts tested up to 417 tokens were on par; longer ones are unmeasured |
| U4 | Reserve field in `hip::ComponentCacheHooks` | The heap gate keeps 4 GiB; the fork keeps 1 GiB (`kCacheBackingReserve`) | Limited cache on IQ1_M + DSpark; smaller `--expert-cache-bytes` on streamed quants. The rest of the fork's budget change is reachable through `retained_snapshot_ceiling_bytes` and `host_ceiling` | Measured 2026-11-10 (card refresh on `255e1739`): IQ1_M + DSpark has 2.2–2.6 GB of heap free after load, commits no backing and refuses every capture, so multi-user DSpark cannot be prepared and does not run. Streamed UD-IQ2_XXS commits the 2.2 GB grow-ahead, then stops at the gate (2.3–2.7 GB free) below its 3.2 GB budget |
| U5 | `SoftmaxTotals` in `SamplerState` (E88) | Top-p totals from the device instead of two host passes over the row | Host passes. Sampled DSpark about 2-4% slower (long request 26.9 to 28 tok/s with E88); greedy unaffected | Repeat on the extension |
| U6 | Stream count in `hip::ComponentCacheHooks` | The host's transfer pool has 64 streams; the fork uses 16 for DS4.1 because 64 kept about 0.4 GiB more host memory resident | 64 streams | Done: gufo-org/gufo#1108; the extension keeps 16 plus one per session (E104) |

Draft #1039 (layered prefill) changes the same prefill API; design U3 against it
and adopt it later if useful.

### Not needed

| Fork change | Reason |
| --- | --- |
| In-place restore | The DS4.1 adapter never implemented `ResidentRows`, so restores already load everything. Not evaluated; restores measure 0.8-2.9 ms on the card |
| `SetPromptLookahead` | The DS4.1 runner never overrode it. BACKLOG item 9 (64.6 ms 3-row pass per chat request) remains open; the extension path is upstream's in-pass checkpoint, not this hook |
| `AwaitSnapshot` (capture join) | Not required: it is scheduler code with no runner part. It may be active for DS4.1 on the fork when requests decode as a batch, and leaving the fork drops it; never measured on DS4.1 |
| Window rows (merged, #1010) | The DS4.1 adapter does not use them |
| 2 MiB pinned pages | No current restore benefit on DS4.1 IQ2_XXS DSpark ([Gemma plan, P6](https://github.com/franzmoca/gufo/blob/feat/gemma/docs/models/gemma-4-31b/EXTENSION-PLAN.md) on `feat/gemma`); no end-to-end number |
| Prompt-lookup agreement | Extension keeps its own `prompt_lookup.hpp`, as Gemma does |
| Bicubic resize | The vision prompt already has its own |
| Qwen GEMM in the vision encoder | Reduced to `vision/lt_gemm` |
| `-Xclang` per-file `-real-true16` | The external validator rejects `-Xclang`. A device-only `target("no-real-true16")` pragma in the two FP16 kernels gives the same result: dense instruction stream identical to the flag, routed 66,171 vs 66,216 instructions (83,629 without either), no 16-bit moves, identical replies |
| `model_bench` per-variant tables | Card refreshes keep a pinned fork tool until upstream's fits |

## Inventory of the fork outside `src/models/deepseek_v41_flash`

| Code on the fork | Destination |
| --- | --- |
| `src/models/deepseek_v41_flash/**` (~37k lines including kernels), `tests/models/deepseek_v41_flash/**`, `tools/ds41/**`, `tools/bench/ds41_*`, `docs/models/deepseek-v4.1-flash/**`, `README_DEEPSEEK.md` | Extension repository, includes rewritten to `<gufo/ext/...>` |
| `DeepSeekTextRunner` V4.1 path and cache backing gate in `src/cli/serve/inference_backend.cpp` | Extension `runner.cpp` on `ext::TextModelRunner`; upstream's V4 `runner.cpp` is the reference |
| V4.1 parts of `src/cli/prompt/prompt.cpp` and `src/cli/bench/bench.cpp` | Extension `drivers.cpp` |
| `--prefill-batch`, `--exact-prefill`, `--stream-experts`, `--expert-cache-bytes`, `--dspark-model` in `serve.cpp` | Manifest options, read with `FindOption` (U1) |
| `ToolFormat::kDeepSeek41`, V4.1 scanners in `openai_chat.cpp` | Registered tool dialect and parser |
| `kDeepSeekV41Flash` sampling preset | Provider `sampling_defaults` |
| `arch == "deepseek41"` shared Engram rule, `kMXFP8`, `kI64` in the GGUF reader and dequant | U2; `kI64` was not needed to open any tested set |
| `src/cli/serve/logging.hpp` includes | Public `src/core/logging/logging.hpp` surface |
| `src/core/direction_adapter.*`, `src/core/lora_adapter.*`, `tools/adapters/*`, `write_gguf` (`feat/ds41-experimental`) | Extension (`direction_adapter.*`, `lora_adapter.*`, `tools/ds41/direction_adapter.py`) |
| `gufo_core`, `gufo_cache_hip` links | `gufo_ext_core`, `gufo_ext_cache`, `gufo_ext_hip`, plus `roc::hipblas`, `roc::hipblaslt` and `ROCWMMA_INCLUDE_DIR`; check `roc::hipblaslt` from outside the tree |
| `licenses/xolotl-expert-rank.txt`, `licenses/ista-daslab-kept-experts.txt`, vendored ds4 and llama.cpp notices | Extension manifest `licenses` |
| `tools/ci/kernel-resources.json` DS4.1 allowances | As the Gemma extension handles its kernels |
| `tests/functional/devlab.json`, `impact.json` entries, `deepseek4` preset use | Extension `functional/`; harness fixes only if upstream's harness still fails |

Keep unchanged: the `no-real-true16` pragma in the routed and dense FP16 kernels
(they call builtins directly because the force-included HIP math wrappers cannot
inline into it), `-ffast-math -fno-finite-math-only` on the package, and the format exclusion of
`runtime/` and `kernels/rocm/` (never run a format fix over them).

## Expert streaming library

Expert cache and SSD streaming form one package: `runtime/expert_store.{h,cpp}`
(about 2,080 lines) and `tools/ds41/{pack_expert_tails,expert_histogram,prefetch_sim,replay_sim,prefetch_fit,rank_eval}.py`.

### What it would own

- The expert cache: size-class slots, admission, eviction, seeding, fitting after
  the sessions, statistics.
- SSD reads: read pool, `O_DIRECT`, read merging, tails files, demand and prefetch
  stages, background jobs.
- Per-layer device weight tables and hit/miss route rows.
- Routing traces and the simulation tools.

### What stays in each model

- Binding GGUF tensors to expert extents.
- The ranking data (`expert_rank.inc`, kept-expert remaps).
- Router predictions, and when `prepare` runs.
- The kernels, which must read weights through the tables. This is the largest
  cost of adopting the library: a model with fixed-stride expert kernels has to
  change them.

### What has to change

| DS4.1 coupling today | Generic form |
| --- | --- |
| `DS41_N_LAYER`, `DS41_N_EXPERT`/`DS41_MAX_EXPERT`, `DS41_N_EXPERT_USED`, `DS41_MAX_PREFILL_BATCH` in `std::array` sizes and loops (69 uses on the Coder branch, which already reads the expert count at load) | Runtime `ExpertLayout` |
| `ds41_model`, `ds41_weights` at creation | Layout filled by the model: per layer and projection, the shard, offset and bytes per expert |
| Three fixed projections (gate, up, down) | Projection count in the layout |
| `kDefaultExpertRank` compiled in; `seed_default` | `Seed(span<ranking>)` from the model |
| `DS41_EXPERT_*` trace and histogram formats, `GUFOETL1` tails header | Keep the file formats; rename the environment and text identifiers, keeping existing tails valid |
| Tuning constants (96 split rows, 16 recent rows, 1,024 sweep tokens, seed 15/16) | Options defaulting to today's values |
| `src/cli/serve/logging.hpp` | Public logging |

Sketch:

```cpp
namespace gufo::expert_stream {
struct ExpertLayout {
  uint32_t layers, experts, routes_per_token, max_batch_rows;
  uint32_t projections;
  std::vector<ExpertExtent> extents;  // per layer and projection
};
std::unique_ptr<Store> Store::Create(ExpertLayout, StoreOptions);
// AttachTails, Seed, SetReserve, Prefetch, Predict, Expect, Stats, Trace.
bool Store::Prepare(layer, routes, n_tokens, tables, overlap, routes_ready);
}
```

### Packaging

Every extension links into one `gufo` binary, so two extensions vendoring the
same sources would collide on CMake targets and symbols. The library therefore
exists once in a build:

- Its own repository, `franzmoca/gufo-expert-streaming`, with `"kind": "library"`.
- Models name it in manifest `depends`, and builds list it in
  `GUFO_EXTENSION_DIRS`, pinned in each model's flake.
- It includes only `gufo/ext/...` and its own files.

The RFC describes library manifests and `depends`, but no external library has
been built this way yet. Test it first; if it does not work, that is a small
`ext-api` PR.

### Upstreaming

Interest is absent today: #427 closed as not planned, and #901 remains open.
A later proposal also needs reads through `src/core/platform/` leaves instead of
direct `O_DIRECT`, `ioctl` and `statfs` calls, Windows support or an explicit
exclusion, a public link target, and a named maintainer. Keeping the library on
the public API only keeps that path open.

### Gates

Every extraction step:

- Identical outputs against the previous step.
- Identical read and hit counts on a fixed trace.
- Balanced timing A/B on a cache-fitting build (GSQ IQ2_XXS) and an overflowing
  one (UD-IQ2_XXS).

The library is not called generic until a second model runs on it.

## Phases

### 0. Baseline and repository

- Done locally: the repository with the Gemma extension's layout, pinned to
  `gemma-main`. Publishing needs go-ahead.
- Deferred until the GPU is free for it: capture the baseline on `feat/ds41-coder`: functional suites including
  `long-context` and `cache`, standard speed cells, oracle PPL, a `pi_agent.py`
  replay, disk restart restore. Use temporary symlink directories with the IQ4_NL
  Engram, including `.expert-tails` links, so the `ds41` links stay untouched.
- Exit: results retained for IQ1_M, UD-IQ2_XXS and Coder, resident and streamed
  as each runs today.

### 1. Skeleton (done)

- `deepseek41` provider for `gguf:deepseek41`, targets split as host, HIP and
  vendored MMQ.
- Options in the manifest. Linux only, as on the fork.
- Exit: builds, passes include isolation, `gufo diagnose` lists the provider.

### 2. Engine, runner, adapter (done; see [Port status](#port-status))

- Move the package; port runner, continuation adapter, drivers, tool dialect and
  sampling defaults.
- Stream experts through the store unchanged, inside the extension.
- Replay at every checkpoint, no `SoftmaxTotals`, IQ4_NL only.
- Exit: greedy output matches the baseline on one resident and one streamed quant.
  The resident side matches; the streamed comparison is still open.
- The expert cache fits after the host commits the cache backing, by wrapping the
  public `commit_backing`; earlier, the backing fails with `bad_alloc`.

### 3. DSpark, vision, tests and tools (`tools/` and `functional/` remain)

- DSpark (`--dspark-model`, joint verification), `--mmproj`, CTest targets,
  `functional/`, `tools/`.
- #1047 is not needed for DSpark defaults on this pin's `speculative_defaults`;
  the pin carries it, so this is unconfirmed without it.

### 4. Parity and gap measurements

- Outputs, oracle PPL and functional suites against phase 0; actual drafts and
  loaded mode verified; disk restart restore.
- Timings per request and phase at 5% / 10 ms, alternating with the baseline.
- Measure the gaps: per-checkpoint replay (U3), cache gate on IQ1_M + DSpark (U4),
  sampled DSpark (U5).
- Exit: a table deciding which of U2-U5 are worth opening.

### 5. Upstream PRs and switch-over

- Open the justified PRs after review of their text; comment on #966 and #901 only
  after approval.
- Move cards to the extension and publish it. The `ds41` script builds pinned Gufo
  plus the extension.
- Freeze fork `main` with a warning that redirects to the Gemma and DS4.1
  extensions; keep `upstream-main` for upstream work. Timing, including whether it
  waits for MXFP8 and UD-IQ1_M, is decided at this point.
- Full card sweep at the end.

### 6. Expert streaming library

After the extension is published. Until then the store stays unchanged inside it.

1. Interface inside the extension: runtime `ExpertLayout`, no `ds41_*` types,
   gates above.
2. CPU tests with a fake layout and fake reads.
3. Move to its own repository and test `depends` from outside the tree.
4. Second consumer. Until a new model exists, a cheap proof is a test build with a
   synthetic layout at a different shape. A real model confirms it.

### 7. Adapters (done)

`--adapter`, `--adapter-scale`, the adapter loaders and the converter from
`feat/ds41-experimental` are on the extension as provider options and
extension sources; no Gufo change was needed ([E98](EXPERIMENTS.md#e98--lora-adapters-retained)).

## Open questions

- When to freeze fork `main`: after U2 lands, or earlier without MXFP8 and UD-IQ1_M?
- Does a metadata-only library in an external directory work with `depends`?
- Do `dspark` and `--dspark-model` work from an external provider alongside the
  in-tree V4 provider, beyond what #1033 states?
- Does GLM 5.3 Flash have per-layer routed experts at all? Not checked; it decides
  how much of the library it can use.
