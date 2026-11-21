# The expert store as proposed to Gufo (E127)

`arms.json`: streamed `UD-IQ2_XXS` with the `MXFP8` Engram, DSpark and vision,
one fresh server per run, both arms built with `nix build` against Gufo
`9cde650f`: extension `main` (A, its own store) and the branch on Gufo's
store (B, Gufo `4e529f3`).

- `replay`: the 53-request Pi coding-agent replay at one session, traced at
  an 80 GiB cache, then untraced in the order A B B A. Replies are SHA-256
  prefixes with the per-run tool-call ids removed. Each run lists the
  server's `expert_cache_grant` fields, its `host_pressure_release` events
  and its direct and buffered expert reads.
- `pressure`: B at one session after one short request, with a 2 GiB and a
  4 GiB memory hog started while idle and freed once the store released:
  available memory at each step, the `expert_cache_release` and
  `expert_cache_restore` events, and whether the same request repeated
  gave the same reply.
- `trace_compare`: `tools/expert_stream/trace_compare.py` on the two traced
  runs.

Prompts and replies are left out.
