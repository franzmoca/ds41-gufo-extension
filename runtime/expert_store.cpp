/* DeepSeek V4.1 Flash's use of the model-neutral expert store
 * (expert_stream/store.h): where the routed experts are in the GGUF shards,
 * the memory the engine keeps for itself, the built-in ranking that warms
 * the cache, expert-pruned builds and histogram files. */
#include "expert_store.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "model.h"
#include "model_data_internal.h"

namespace {

constexpr uint64_t kGiB = 1ull << 30;
/* Headroom kept above the cache for everything allocated after it, when the
 * cache is sized before the sessions exist (tools, benches): the shared batch
 * scratch (~1.4 GiB at 1M context), one session's FP4 state (~0.9 GiB at 1M)
 * and transient prefill peaks. The server sizes the cache after its sessions
 * instead (ds41_expert_store_set_reserve). */
constexpr uint64_t kReserve = 4 * kGiB;
/* The batch scratch grows with the prefill batch: 0.55 GiB measured from
 * 2,048 to 4,096 tokens. Left out of the load seed's headroom, a 4,096-token
 * batch took the device heap under the reserve the prompt cache's backing
 * must keep, and the server ran without a prompt cache (E113). */
constexpr uint64_t kReserveBatch = 2048;
constexpr uint64_t kReservePerToken = 320 << 10;

constexpr std::array<uint16_t, DS41_N_LAYER * DS41_MAX_EXPERT>
    kDefaultExpertRank = {
#include "expert_rank.inc"
};
constexpr bool valid_default_rank() {
  std::array<bool, DS41_N_LAYER * DS41_MAX_EXPERT> seen{};
  for (uint32_t index : kDefaultExpertRank) {
    if (index >= seen.size() || seen[index])
      return false;
    seen[index] = true;
  }
  return true;
}
static_assert(valid_default_rank());

/* An expert-pruned build: its routed experts are original experts kept[l *
 * experts + j] of the full model, recognized by a fingerprint of its routing
 * biases since the GGUF does not carry the mapping. */
struct PrunedBuild {
  uint64_t fingerprint;
  uint32_t experts;
  std::array<uint16_t, DS41_N_LAYER * DS41_MAX_EXPERT> kept;
};
#include "pruned_gsq_rco_coder.inc"
constexpr const PrunedBuild* kPrunedBuilds[] = {&kGsqRcoCoder};
constexpr bool valid_pruned(const PrunedBuild& b) {
  if (!b.experts || b.experts >= DS41_MAX_EXPERT)
    return false;
  for (uint32_t l = 0; l < DS41_N_LAYER; ++l)
    for (uint32_t j = 0; j < b.experts; ++j) {
      const uint32_t e = b.kept[l * b.experts + j];
      if (e >= DS41_MAX_EXPERT || (j && e <= b.kept[l * b.experts + j - 1]))
        return false;
    }
  return true;
}
static_assert(valid_pruned(kGsqRcoCoder));

}  // namespace

struct ds41_expert_store {
  std::unique_ptr<expert_stream::Store> store;
  uint32_t n_expert = DS41_MAX_EXPERT;  // routed experts per layer in the GGUF
  /* For a known expert-pruned build, its kept original experts (seeding maps
   * the built-in ranking through them); null for the full model or an
   * unknown pruned build. */
  const PrunedBuild* pruned = nullptr;
};

uint64_t ds41_expert_available_memory() {
  return expert_stream::AvailableMemory();
}

ds41_expert_store* ds41_expert_store_create(const ds41_model* model,
                                            const ds41_weights* weights,
                                            uint64_t cache_limit_bytes,
                                            uint32_t prefill_batch) {
  if (!model || !weights || model->n_shards > DS41_MAX_SHARDS ||
      weights->n_expert < DS41_N_EXPERT_USED ||
      weights->n_expert > DS41_MAX_EXPERT)
    return nullptr;
  auto owner = std::make_unique<ds41_expert_store>();
  owner->n_expert = weights->n_expert;
  if (owner->n_expert < DS41_MAX_EXPERT) {
    uint64_t h = 0xcbf29ce484222325ull;  // FNV-1a 64 over the routing biases
    for (uint32_t l = 0; l < DS41_N_LAYER; ++l) {
      const ds41_tensor* t = weights->layer[l].ffn_exp_probs_b;
      for (uint64_t i = 0; i < t->bytes; ++i)
        h = (h ^ model->map[t->abs_offset + i]) * 0x100000001b3ull;
    }
    for (const PrunedBuild* b : kPrunedBuilds)
      if (b->fingerprint == h && b->experts == owner->n_expert)
        owner->pruned = b;
  }
  /* Each layer's gate, up and down expert tensors, found in the shard that
   * holds them. */
  expert_stream::Layout layout;
  layout.experts = owner->n_expert;
  layout.top_k = DS41_N_EXPERT_USED;
  layout.pieces = 3;
  layout.max_batch_tokens = DS41_MAX_PREFILL_BATCH;
  layout.table_experts = DS41_MAX_EXPERT;  // kernels index a fixed-width table
  layout.layers.resize(DS41_N_LAYER);
  for (uint32_t layer = 0; layer < DS41_N_LAYER; ++layer) {
    const auto& w = weights->layer[layer];
    const ds41_tensor* tensors[] = {w.ffn_gate_exps, w.ffn_up_exps,
                                    w.ffn_down_exps};
    for (const ds41_tensor* t : tensors) {
      if (!t || !t->bytes || t->bytes % owner->n_expert)
        return nullptr;
      expert_stream::Piece piece;
      piece.bytes = t->bytes / owner->n_expert;
      for (uint32_t s = 0; s < model->n_shards; ++s) {
        const uint64_t start = model->shard_offset[s];
        if (t->abs_offset >= start &&
            t->abs_offset - start <= model->shard_size[s] &&
            t->bytes <= model->shard_size[s] - (t->abs_offset - start)) {
          piece.fd = model->shard_fd[s];
          piece.offset = t->abs_offset - start;
          break;
        }
      }
      if (piece.fd < 0)
        return nullptr;
      layout.layers[layer].push_back(piece);
    }
  }
  expert_stream::Policy policy;
  policy.cache_limit_bytes = cache_limit_bytes;
  policy.reserve_bytes = kReserve;
  if (prefill_batch > kReserveBatch)
    policy.reserve_bytes += (prefill_batch - kReserveBatch) * kReservePerToken;
  if (const char* path = std::getenv("DS41_EXPERT_TRACE"); path && *path)
    policy.trace_path = path;
  policy.log_tag = "ds41";
  owner->store = expert_stream::Store::Create(layout, policy);
  if (!owner->store)
    return nullptr;
  return owner.release();
}

void ds41_expert_store_destroy(ds41_expert_store* store) {
  delete store;
}

bool ds41_expert_store_attach_tails(ds41_expert_store* store,
                                    const char* path) {
  if (!store || !path)
    return false;
  if (store->store->AttachTails(path))
    return true;
  fprintf(stderr,
          "ds41: expert tails files are built by "
          "tools/ds41/pack_expert_tails.py\n");
  return false;
}

bool ds41_expert_store_set_reserve(ds41_expert_store* store,
                                   uint64_t reserve_bytes) {
  return store && store->store->SetReserve(reserve_bytes);
}

bool ds41_expert_store_seed_default(ds41_expert_store* store) {
  if (!store)
    return false;
  /* A pruned build takes the ranking's kept experts in the same order,
   * renumbered; an unknown pruned build seeds without a ranking. */
  std::vector<expert_stream::Ranked> ranked;
  if (store->n_expert == DS41_MAX_EXPERT || store->pruned) {
    constexpr uint16_t kDropped = 0xffff;
    std::array<std::array<uint16_t, DS41_MAX_EXPERT>, DS41_N_LAYER> own;
    for (uint32_t l = 0; l < DS41_N_LAYER; ++l) {
      own[l].fill(kDropped);
      for (uint32_t j = 0; j < store->n_expert; ++j)
        own[l][store->pruned ? store->pruned->kept[l * store->n_expert + j]
                             : j] = static_cast<uint16_t>(j);
    }
    ranked.reserve(kDefaultExpertRank.size());
    for (size_t i = 0; i < kDefaultExpertRank.size(); ++i) {
      const uint32_t layer = kDefaultExpertRank[i] / DS41_MAX_EXPERT;
      const uint16_t expert =
          own[layer][kDefaultExpertRank[i] % DS41_MAX_EXPERT];
      if (expert != kDropped)
        ranked.push_back(
            {layer, expert,
             static_cast<uint64_t>(kDefaultExpertRank.size() - i)});
    }
  }
  const char* source =
      !store->pruned
          ? (ranked.empty() ? "no ranking for this expert-pruned build"
                            : "built-in V4.1 ranking")
          : "built-in V4.1 ranking of the kept experts";
  return store->store->Seed(std::move(ranked), source);
}

bool ds41_expert_store_seed(ds41_expert_store* store, const char* path) {
  if (!store || !path)
    return false;
  std::ifstream in(path);
  std::string magic;
  if (!std::getline(in, magic) || magic != "DS41_EXPERT_HISTOGRAM_V1")
    return false;
  std::array<std::array<bool, DS41_MAX_EXPERT>, DS41_N_LAYER> seen{};
  std::vector<expert_stream::Ranked> ranked;
  std::string line;
  while (std::getline(in, line)) {
    uint32_t layer = 0, expert = 0;
    uint64_t count = 0;
    std::string extra;
    std::istringstream row(line);
    if (!(row >> layer >> expert >> count) || (row >> extra))
      return false;
    if (layer >= DS41_N_LAYER || expert >= store->n_expert || !count ||
        seen[layer][expert])
      return false;
    seen[layer][expert] = true;
    ranked.push_back({layer, expert, count});
  }
  if (!in.eof())
    return false;
  std::sort(ranked.begin(), ranked.end(),
            [](const expert_stream::Ranked& a, const expert_stream::Ranked& b) {
              if (a.count != b.count)
                return a.count > b.count;
              return a.layer != b.layer ? a.layer < b.layer
                                        : a.expert < b.expert;
            });
  return store->store->Seed(std::move(ranked), path);
}

bool ds41_expert_store_prepare(
    ds41_expert_store* store, uint32_t layer, const int32_t* routes,
    uint32_t n_tokens, const void** gate, const void** up, const void** down,
    bool (*overlap)(void*, const ds41_expert_overlap*), void* overlap_context,
    hipEvent_t routes_ready) {
  if (!store || !gate || !up || !down)
    return false;
  const void* tables[3] = {};
  if (!store->store->Prepare(layer, routes, n_tokens, tables, overlap,
                             overlap_context, routes_ready))
    return false;
  *gate = tables[0];
  *up = tables[1];
  *down = tables[2];
  return true;
}

bool ds41_expert_store_prefetch(ds41_expert_store* store, uint32_t layer,
                                const int32_t* experts, uint32_t count,
                                bool wide) {
  return store && store->store->Prefetch(layer, experts, count, wide);
}

bool ds41_expert_store_predict(ds41_expert_store* store, uint32_t layer,
                               hipEvent_t done, const int32_t* ids,
                               uint32_t count, bool wide) {
  return store && store->store->Predict(layer, done, ids, count, wide);
}

void ds41_expert_store_expect(ds41_expert_store* store, uint32_t layer) {
  if (store)
    store->store->Expect(layer);
}

void ds41_expert_store_predict_flush(ds41_expert_store* store) {
  if (store)
    store->store->PredictFlush();
}

bool ds41_expert_store_tracing(const ds41_expert_store* store) {
  return store && store->store->Tracing();
}

void ds41_expert_store_trace_prediction(ds41_expert_store* store,
                                        uint32_t layer, uint32_t lead,
                                        const int32_t* ids, uint32_t n_tokens,
                                        uint32_t k) {
  if (store)
    store->store->TracePrediction(layer, lead, ids, n_tokens, k);
}

bool ds41_expert_store_prefetch_idle(const ds41_expert_store* store) {
  return store ? store->store->PrefetchIdle() : true;
}

ds41_expert_store_stats ds41_expert_store_get_stats(
    const ds41_expert_store* store) {
  return store ? store->store->GetStats() : ds41_expert_store_stats{};
}

void ds41_expert_store_log(const ds41_expert_store* store) {
  if (store)
    store->store->Log();
}
