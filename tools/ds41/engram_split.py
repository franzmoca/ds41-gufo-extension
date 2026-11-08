#!/usr/bin/env python3
"""Set the split position an Engram GGUF declares.

A build's Engram table is the last shard of its split, and the file records
which shard of how many it is. Downloaded as published, every build matches.
To link one table into a build with a different shard count, for example
ISTA's 6-of-6 table into a five-shard Unsloth build, the numbers must change:

  # A copy (free on Btrfs and XFS, a full copy elsewhere):
  python3 tools/ds41/engram_split.py TABLE.gguf TABLE-5of5.gguf --shard 5 --of 5
  # Or the file itself; builds that link it must then match the new numbers:
  python3 tools/ds41/engram_split.py TABLE.gguf --in-place --shard 5 --of 5

Only the two 16-bit split fields change.
"""
import argparse
import mmap
from pathlib import Path
import struct
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gufo.gguf import Reader  # noqa: E402

UINT16 = 2


def split_fields(path):
    """{key: (value offset, value)} for split.no and split.count."""
    with path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
        if data[0:4] != b"GGUF":
            raise SystemExit(f"{path} is not a GGUF file")
        r = Reader(data)
        r.pos = 4
        r.u32()
        r.u64()
        n_kv = r.u64()
        fields = {}
        for _ in range(n_kv):
            key = r.string()
            tag = r.u32()
            if key in ("split.no", "split.count"):
                if tag != UINT16:
                    raise SystemExit(f"{key} has GGUF type {tag}, expected uint16")
                fields[key] = (r.pos, struct.unpack_from("<H", data, r.pos)[0])
            r.skip_value(tag)
    for key in ("split.no", "split.count"):
        if key not in fields:
            raise SystemExit(f"{path} has no {key}; it is not a split shard")
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, nargs="?", help="the copy to write; omit with --in-place")
    parser.add_argument("--in-place", action="store_true", help="change the source file itself")
    parser.add_argument("--shard", type=int, required=True, help="1-based shard number, the last one of the build")
    parser.add_argument("--of", type=int, required=True, help="shards in the build")
    args = parser.parse_args()
    if not 1 <= args.shard <= args.of:
        parser.error("--shard must be between 1 and --of")
    if args.in_place == (args.output is not None):
        parser.error("give either an output path or --in-place")
    if args.output is not None and args.output.exists():
        parser.error(f"{args.output} exists; refusing to overwrite it")
    fields = split_fields(args.source)
    old = (fields["split.no"][1] + 1, fields["split.count"][1])
    target = args.source
    if args.output is not None:
        subprocess.run(["cp", "--reflink=auto", "--", str(args.source), str(args.output)], check=True)
        target = args.output
    with target.open("r+b") as f:
        for key, value in (("split.no", args.shard - 1), ("split.count", args.of)):
            f.seek(fields[key][0])
            f.write(struct.pack("<H", value))
    print(f"{target}: shard {old[0]} of {old[1]} -> {args.shard} of {args.of}")


if __name__ == "__main__":
    main()
