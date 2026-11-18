#!/usr/bin/env python3
"""Check that two DS41_EXPERT_TRACE files record the same expert-store decisions.

  trace_compare.py A.bin B.bin

Compares the seeded snapshot, every prepare's layer, routes, hit flags and
demand bytes, and every router prediction, ignoring what legitimately varies
between runs: timestamps, read wait and the free memory at the snapshot.
Prints the first difference and exits 1, or a summary and exits 0.

Both runs must replay the same requests with a fixed --expert-cache-bytes,
so the cache size does not depend on the memory free at startup. Tracing
disables the production prediction, so adopted prefetch bytes are zero.
Stdlib only.
"""
import struct
import sys


def records(path):
    data = open(path, "rb").read()
    pos = 0
    used = None

    def take(fmt):
        nonlocal pos
        values = struct.unpack_from("<" + fmt, data, pos)
        pos += struct.calcsize("<" + fmt)
        return values

    def raw(n):
        nonlocal pos
        chunk = data[pos:pos + n]
        pos += n
        return chunk

    while pos < len(data):
        tag = data[pos:pos + 1]
        pos += 1
        if tag == b"S":
            version, n_layer, n_expert, used = take("IIII")
            if version != 1:
                raise SystemExit(f"{path}: trace version {version}")
            tick, limit, cache_bytes, _available = take("QQQQ")
            (n_class,) = take("I")
            classes = take(f"{n_class}Q")
            layers = [take("IQ") for _ in range(n_layer)]
            (n_slots,) = take("I")
            slots = raw(n_slots * 12)
            entries = raw(n_layer * n_expert * 16)
            yield ("S", (n_layer, n_expert, used, tick, limit, cache_bytes, classes, tuple(layers), slots, entries))
        elif tag == b"P":
            _t0, _t1, _t2, layer, n, _wait, demand, adopted = take("QQQIIQQQ")
            ids = raw(n * used * 4)
            hits = raw(n * used)
            yield ("P", (layer, n, demand, adopted, ids, hits))
        elif tag == b"Q":
            layer, lead, n, k = take("IIII")
            yield ("Q", (layer, lead, n, k, raw(n * k * 4)))
        else:
            raise SystemExit(f"{path}: bad record tag {tag!r} at byte {pos - 1}")


FIELDS = {"S": "layers, experts, top-k, tick, limit, cache bytes, classes, layer classes, slots, entries",
          "P": "layer, tokens, demand bytes, adopted bytes, routes, hits",
          "Q": "layer, lead, tokens, k, ids"}


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    a, b = (records(p) for p in sys.argv[1:])
    counts = {"S": 0, "P": 0, "Q": 0}
    tokens = hits = routes = 0
    index = 0
    while True:
        ra, rb = next(a, None), next(b, None)
        if ra is None and rb is None:
            break
        if ra is None or rb is None:
            print(f"record {index}: {'A' if ra is None else 'B'} ends first")
            return 1
        if ra != rb:
            print(f"record {index}: {ra[0]} differs from {rb[0]}")
            if ra[0] == rb[0]:
                names = FIELDS[ra[0]].split(", ")
                for name, x, y in zip(names, ra[1], rb[1]):
                    if x != y:
                        shown = (lambda v: v if not isinstance(v, bytes) else f"{len(v)} bytes")
                        print(f"  {name}: {shown(x)} != {shown(y)}")
            return 1
        kind, body = ra
        counts[kind] += 1
        if kind == "P":
            layer, n, _, _, _, hit_flags = body
            tokens += n if layer == 0 else 0
            routes += len(hit_flags)
            hits += sum(hit_flags)
        index += 1
    rate = 100.0 * hits / routes if routes else 0.0
    print(f"identical: {counts['S']} snapshot, {counts['P']} prepares, {counts['Q']} predictions, "
          f"{tokens} tokens, {rate:.2f}% route hits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
