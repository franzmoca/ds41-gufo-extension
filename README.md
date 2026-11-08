# DeepSeek V4.1 Flash for Gufo

A [Gufo](https://github.com/gufo-org/gufo) source extension that runs
DeepSeek-V4.1-Flash-1017, a 763B-parameter mixture-of-experts model, on one AMD
Strix Halo (`gfx1151`) with 128 GB of unified memory on Linux. It includes DSpark
drafting, prompt lookup, image input, tool calls, prompt caching and streaming of
routed experts from the SSD.

Maintainer: Francesco Moca ([@franzmoca](https://github.com/franzmoca)).
[Report bugs here](https://github.com/franzmoca/ds41-gufo-extension/issues), with
the GGUF file names when a model does not load.

> [!IMPORTANT]
> **The port is in progress.** Until it finishes, the
> [`feat/ds41-coder`](https://github.com/franzmoca/gufo/tree/feat/ds41-coder) branch
> of the fork remains the complete version. Its
> [docs](https://github.com/franzmoca/gufo/tree/feat/ds41-coder/docs/models/deepseek-v4.1-flash)
> hold the benchmarks, quality reports, streaming design and experiments.

> [!WARNING]
> **Use a dedicated, headless machine.** The large builds leave about 2 GiB of host
> memory free. A desktop, browser or other GPU workload can push the server or the
> system out of memory.

## Models

| Build | Backbone | Runs | On this extension |
| --- | ---: | --- | --- |
| [ISTA GSQ-RCO Coder](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF) `IQ2_XXS`, 192 of 384 experts | 67.2 GiB | Resident, about 44 GiB free | Measured |
| [ISTA GSQ-RCO](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) `IQ2_XXS` | 128.7 GiB | `--stream-experts` | Measured |
| [Unsloth](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) `UD-IQ2_XXS` | 154.0 GiB | `--stream-experts` | Measured |
| Unsloth `UD-IQ1_M` | 130.2 GiB | `--stream-experts` | Runs, see below |
| ISTA GSQ-RCO `IQ1_M` | 112.4 GiB | Resident | Not yet run |
| ISTA GSQ-RCO `IQ2_XS` | 144.6 GiB | `--stream-experts` | Not yet run |
| Unsloth `UD-IQ1_S` | 114.0 GiB | `--stream-experts` | Not yet run |

Every build also needs an Engram file as its last shard, read from the SSD: ISTA's
`IQ4_NL` (103 GiB) or Unsloth's `MXFP8` (189 GiB, decode about 8-10% slower on
streamed `UD-IQ2_XXS`; prefill unchanged). The recommended drafter is Lucebox's
`DSpark-draft-Q2K-Q4K` (4.6 GiB resident).

## Build

The extension needs Gufo's extension API 0.2.14 (upstream `main`).

```sh
nix build   # in this repository: ./result/bin/gufo
```

Or without Nix, with [Gufo's prerequisites](https://github.com/gufo-org/gufo#build-from-source):

```sh
git clone https://github.com/gufo-org/gufo
git clone https://github.com/franzmoca/ds41-gufo-extension
cd gufo
cmake --preset release -DGUFO_EXTENSION_DIRS="$PWD/../ds41-gufo-extension"
cmake --build --preset release --parallel 4
```

## Run

Keep all files of a build in one directory. For example, streamed `UD-IQ2_XXS`:

```sh
mkdir ds41 && cd ds41
hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF \
  --include "UD-IQ2_XXS/*-0000[1-5]-of-00006.gguf" --local-dir .
hf download ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF \
  DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf --local-dir .
hf download Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF \
  DeepSeek-V4.1-Flash-1017-DSpark-draft-Q2K-Q4K.gguf --local-dir .
ln -s ../DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf \
  UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00006-of-00006.gguf

gufo serve llm \
  -m UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf \
  --stream-experts \
  --speculative dspark \
  --dspark-model DeepSeek-V4.1-Flash-1017-DSpark-draft-Q2K-Q4K.gguf \
  --cache-disk ./cache --host 0.0.0.0 --port 8080
```

For image input, download `mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf` from the ISTA
repository and add `--mmproj`.

For the `MXFP8` Engram, download `engram/DeepSeek-V4.1-Flash-1017-engram-MXFP8.gguf_file`
from the Unsloth repository and link it as the last shard in place of the `IQ4_NL`
file. It also declares itself shard 6 of 6.

| Option | Effect |
| --- | --- |
| `--dspark-model PATH` | DSpark drafter for `--speculative dspark` |
| `--stream-experts` | Keep dense weights resident and stream routed experts from the SSD |
| `--expert-cache-bytes N` | Expert cache size (default: the free memory, minus 2 GiB) |
| `--prefill-batch N` | Most prompt tokens per prefill batch, 16 to 4096 (default: 2048) |
| `--exact-prefill` | Run the decoder layers over every prompt token, at about half the prefill speed |

Useful Gufo options: `--context N` per session (default: the native 1M tokens),
`--sessions N` (about 0.9 GiB each at 1M), `--think off|on|auto`,
`--cache-ram-bytes N`, and `--cache-disk DIR`, which keeps prompt-cache checkpoints
across restarts.

### Engram for `UD-IQ1_M` and `UD-IQ1_S`

These builds have five shards, but the ISTA and `MXFP8` Engram files declare
themselves shard 6 of 6, so Gufo rejects a link to them. Download Unsloth's shard 5
with the other shards. It holds the same `IQ4_NL` tables (103 GiB).

If you already have the ISTA file, change the shard numbers on a copy instead, in the
directory that holds it:

```sh
# The script edits the file in place, so never run it on the original.
# On Btrfs or XFS, --reflink uses no extra space; elsewhere this copies 103 GiB.
cp --reflink=auto DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf Engram-IQ4_NL-shard5.gguf

curl -LO https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/resolve/main/engram/set_split.py
python3 set_split.py Engram-IQ4_NL-shard5.gguf 5 5

ln -s ../Engram-IQ4_NL-shard5.gguf \
  UD-IQ1_M/DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00005-of-00005.gguf
```

## Streaming and expert tails

With `--stream-experts`, dense weights stay resident and routed experts are read
from the SSD into a cache that fills the free memory, so free memory directly
reduces disk reads. Streaming is never chosen automatically: a model that does not
fit fails to load and states what it needs. The automatic RAM prompt cache is 1 GiB
with `--cache-disk` and 3 GiB without, so the expert cache gets the rest.

**Put expert tails on a second NVMe if you have one.** When
`MODEL.gguf.expert-tails` sits beside the first shard, as a file or a link, every
expert read becomes a head read from the GGUF and a tail read from that file, queued
together, so both drives work on each miss. Output does not change. Create it with
this repository's tool, which needs Python 3 with NumPy:

```sh
python3 tools/ds41/pack_expert_tails.py \
  --model /path/to/ds41/UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf \
  --out /second-drive/UD-IQ2_XXS.expert-tails
ln -s /second-drive/UD-IQ2_XXS.expert-tails \
  /path/to/ds41/UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf.expert-tails
```

| Build | Tails file |
| --- | ---: |
| `IQ2_XXS` | 61.6 GiB |
| `UD-IQ1_M` | 61.5 GiB |
| `UD-IQ2_XXS` | 73.0 GiB |

Measured on the fork with two PCIe 4.0 x4 drives, one cold miss takes 0.87 ms instead
of 1.32-1.47 ms, and six misses reach 10.7 GB/s. The smaller the expert cache is
against the model, the more it helps:

| Expert cache | Prefill gain | Decode gain |
| --- | ---: | ---: |
| About 100 GiB | 5-17% | 1-3% |
| 88 GiB | 18-27% | 2-5% |
| 64 GiB | 47-65% | 12-23% |

At load the engine checks the file against the model and, on any mismatch, logs why
and reads only the GGUF. On btrfs the packer turns off copy-on-write for the file,
which direct reads need. Details are in the
[streaming design](https://github.com/franzmoca/gufo/blob/feat/ds41-coder/docs/models/deepseek-v4.1-flash/STREAMING.md#split-reads-across-two-drives).

## Memory

Measured on the fork with resident `IQ1_M`, the native 1M context and one session:

| Component | Memory |
| --- | ---: |
| Backbone | 112.3 GiB |
| Session | 2.3 GiB |
| DSpark drafter, Q2_K/Q4_K | 4.59 GiB |
| Vision projector | 0.90 GiB |
| Engram tables | 103 GiB on disk, not resident |
| **Host memory left** with the drafter, server ready | **1.75 GiB** |

On this extension, streamed `UD-IQ2_XXS` with the drafter and vision projector has a
95-96 GiB expert cache and about 2 GiB free, and the resident Coder build leaves about
44 GiB free.

## Host setup

- Set the UMA carve-out to its minimum in the firmware; the model lives in GTT.
- Let the GPU map nearly all memory with
  `amdgpu.gttsize=126976 ttm.pages_limit=32505856`. Then
  `/sys/class/drm/card*/device/mem_info_gtt_total` should report about 133 GB.
- `CmaFree` in `/proc/meminfo` should be near 0. On Ubuntu 26.04, boot with `kho=off`.
- `ulimit -l unlimited` lets the prompt cache pin its memory.

## Against the fork

Streamed `UD-IQ2_XXS`, `IQ4_NL` Engram, DSpark, vision, one session, disk cache, cold
expert cache, balanced order, average of two runs, November 7, 2026:

| Greedy | This extension | Fork |
| --- | ---: | ---: |
| Prefill, 3.5K tokens (tok/s) | 595 | 586 |
| Prefill, 24K tokens (tok/s) | 1,045 | 1,041 |
| Decode, chat (tok/s) | 19.4 | 19.4 |
| Decode, code (tok/s) | 28.5 | 28.7 |
| Decode, long document (tok/s) | 27.3 | 26.9 |

With the `MXFP8` Engram, on upstream `main` (decode averages four runs per build,
November 7-8, 2026; prefill is one run):

| Greedy | This extension | Fork |
| --- | ---: | ---: |
| Prefill, 3.5K tokens (tok/s) | 589 | 586 |
| Prefill, 24K tokens (tok/s) | 1,050 | 1,022 |
| Decode, chat (tok/s) | 17.8 | 17.9 |
| Decode, code (tok/s) | 26.1 | 26.1 |
| Decode, long document (tok/s) | 25.2 | 25.3 |

Decode varies more between runs than between the two builds: DSpark learns its draft
lengths from measured step times, and the expert cache size depends on the memory free
at startup.

Generated text matches the fork on these requests. A smoke run also covered tool
calls, images, concurrent requests and prompt-cache reuse.

**Known gaps:**

- Gufo's 64 cache transfer streams leave 0.2-0.6 GiB less expert cache than the fork.
- Gufo's 4 GiB heap reserve can trim the prompt-cache backing at startup (seen once
  in four runs, 829 MiB instead of 1 GiB, with no speed change).
- Not yet run here: resident `IQ1_M`, `gufo bench` and `gufo prompt`,
  `--exact-prefill`, the functional suites other than `reasoning-budget`, and the
  quality checks.

Source layout and origin are in [UPSTREAM.md](UPSTREAM.md).

## License

MIT ([LICENSE](LICENSE)). Code derived from DS4 keeps its license
([LICENSE.ds4](LICENSE.ds4)), code copied from Gufo keeps Gufo's
([LICENSES/gufo.txt](LICENSES/gufo.txt)), and the vendored llama.cpp kernels are
described in [VENDOR.md](kernels/rocm/mmq/VENDOR.md). Model weights are downloaded
separately under their own licenses.
