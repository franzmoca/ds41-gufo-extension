# Tools

The scripts need Python 3 with NumPy. The oracle, `dspark_reference.py` and
`vision_reference.py` also need PyTorch, and `vision_reference.py` Pillow.
`gufo/gguf.py` is Gufo's GGUF reader, copied here so the scripts run outside a Gufo
checkout.

## Model files

| Script | Use |
| --- | --- |
| `ds41/engram_split.py` | Set the shard an Engram file declares, on a copy or in place, to share one table across builds |
| `ds41/pack_expert_tails.py` | Pack every routed expert's tail into one file for a second drive |
| `ds41/import_expert_rank.py` | Convert Xolotl's published expert ranking to the built-in `runtime/expert_rank.inc` |
| `ds41/import_pruned_experts.py` | Convert an expert-pruned release's kept-experts list to a built-in table |

## Expert streaming

These replay an expert trace, which the engine writes with `DS41_EXPERT_TRACE=PATH`.

| Script | Use |
| --- | --- |
| `ds41/prefetch_sim.py` | Compare prefetch policies offline |
| `ds41/prefetch_fit.py` | Fit the simulator's costs to measured A/B ratios |
| `ds41/evict_sim.py` | Compare eviction rules, LRU and the offline optimum |
| `ds41/replay_sim.py` | Read stalls of the decoder's bounded replay |
| `ds41/rank_eval.py` | Evaluate the built-in expert ranking |
| `ds41/route_occupancy.py` | Routed tile occupancy per layer |
| `ds41/expert_histogram.py` | Build an expert-cache seed from oracle routes |
| `ds41/stream_io_bench.py` | Read-only SSD benchmark over real expert spans |
| `ds41/engram_row_bench.py` | Cold random Engram row reads |

## References

| Script | Use |
| --- | --- |
| `ds41/reference.py` | CPU oracle over the GGUF weights (official `model.py` and `engram.py`) |
| `ds41/quants.py`, `ds41/gguf_model.py` | Dequantization and split-GGUF access for the oracle |
| `ds41/dspark_reference.py` | CPU oracle for one DSpark draft |
| `ds41/vision_reference.py` | CPU reference for the vision encoder |
| `ds41/chat_goldens.py` | Chat-template goldens from the official encoder (`tests/fixtures/encoding-1017.json`) |

`dspark_reference.py` and `vision_reference.py` compare against dumps from GPU probes
(`ds41_load_probe`, `ds41_vision_probe`) that this extension does not build yet;
[QUALITY.md](../docs/models/deepseek-v4.1-flash/QUALITY.md) describes the checks.

## Kernel benchmarks

`bench/` holds standalone `gfx1151` benchmarks of the decoder's kernels. Each one
includes the production source from `kernels/rocm/`. Build them into `/tmp` from a
Gufo development shell:

```sh
nix develop github:gufo-org/gufo -c tools/bench/build.sh ds41_expert_gemv_tune
```
