# Changelog

Notable user-facing changes are recorded here. Versions follow `version` in
[extension.json](extension.json).

## Unreleased

### Performance

* **dspark:** greedy decoding resumes drafting as soon as a probe shows that it
  pays, and every request starts with a fresh draft policy. Greedy chat decodes
  4-7% faster and code up to 7.5% faster on streamed builds, with identical
  output; sampled decoding is unchanged
  ([E99](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e99--greedy-dspark-reset-the-policy-per-request-probes-set-the-saving-retained)).
* **dspark:** the drafter's attention reads its keys once per head instead of
  once per head and row, 4× faster; drafting cycles take ~1% less time
  ([E102](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e102--dspark-attention-one-block-per-head-retained)).

## 0.1.0 (2026-11-08)

First release as a Gufo source extension, built on upstream Gufo `main`
(extension API 0.2.14).

### Features

* DeepSeek V4.1 Flash on one Strix Halo: `IQ1_M` and the expert-pruned Coder
  resident, larger builds with `--stream-experts`, optionally reading expert
  tails from a second drive.
* DSpark speculative decoding and prompt lookup, greedy and sampled.
* Gufo's OpenAI-compatible text API with reasoning, native tool calls, image
  input, concurrent sessions, and RAM and disk prompt caching.
* `IQ4_NL`, `Q5_K`, `Q8_0` and `MXFP8` Engram tables.
* `tools/ds41/engram_split.py` sets the shard position of an Engram table, on a
  copy or in place, so one table can serve builds with different shard counts.

### Bug Fixes

* Resident `IQ1_M` with DSpark and vision loads with a RAM prompt cache inside
  the host memory reserve.
