#!/usr/bin/env python3
"""Record an agent session's chat requests once, then replay them verbatim.

An unseeded agent session diverges from run to run (different replies, tool
calls and decode lengths), so two builds or settings never see the same
work. Recording the requests through a proxy and resending them in order,
non-streamed and seeded, gives every run the same prompts; with an
output-preserving change the replies are identical too (E105).

  agent_replay.py record LISTEN_PORT UPSTREAM_URL OUT.jsonl
      Proxy on 127.0.0.1:LISTEN_PORT to UPSTREAM_URL (http://host:port),
      appending every POST body to OUT.jsonl in arrival order. Point the
      agent at http://127.0.0.1:LISTEN_PORT/v1.
  agent_replay.py replay BASE_URL REQUESTS.jsonl[.gz] OUT.json [--seed N]
      Sends each recorded chat request to BASE_URL/v1/chat/completions
      (stream off, seed N unless recorded, default 1234); writes per-request
      usage, timings and replies, and prints totals.
  agent_replay.py compare A.json B.json [...]
      Totals per run, and whether replies match the first run (tool-call ids
      ignored).

Stdlib only. The server must already be running; start a fresh one per run
so the expert and prompt caches start equal.
"""
import gzip
import http.client
import json
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def record(listen, upstream, out):
    target = urllib.parse.urlsplit(upstream)
    lock = threading.Lock()
    seq = [0]

    class Proxy(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def forward(self, method):
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n else None
            if method == "POST":
                with lock:
                    seq[0] += 1
                    with open(out, "a") as f:
                        f.write(json.dumps({"i": seq[0], "path": self.path, "body": json.loads(body)}) + "\n")
            skip = ("host", "content-length", "connection")
            headers = {k: v for k, v in self.headers.items() if k.lower() not in skip}
            conn = http.client.HTTPConnection(target.hostname, target.port or 80, timeout=3600)
            conn.request(method, self.path, body=body, headers=headers)
            resp = conn.getresponse()
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                    self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            # Relay as it arrives, so streamed replies reach the agent live.
            while chunk := resp.read1(65536):
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            conn.close()

        def do_GET(self):
            self.forward("GET")

        def do_POST(self):
            self.forward("POST")

    print(f"recording to {out}; agent base URL http://127.0.0.1:{listen}/v1", flush=True)
    ThreadingHTTPServer(("127.0.0.1", listen), Proxy).serve_forever()


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def replay(base, requests, out, seed):
    rows = []
    for rec in load(requests):
        if not rec["path"].endswith("/chat/completions"):
            continue
        body = dict(rec["body"])
        body.pop("stream", None)
        body.pop("stream_options", None)
        body.setdefault("seed", seed)
        start = time.perf_counter()
        req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        resp = json.load(urllib.request.urlopen(req, timeout=3600))
        wall = time.perf_counter() - start
        usage, timings = resp.get("usage", {}), resp.get("timings", {})
        msg = resp["choices"][0]["message"]
        rows.append({"i": rec["i"], "prompt": usage.get("prompt_tokens"), "gen": usage.get("completion_tokens"),
                     "cached": usage.get("prompt_tokens_details", {}).get("cached_tokens"),
                     "prompt_ms": timings.get("prompt_ms"), "predicted_ms": timings.get("predicted_ms"),
                     "draft_n": timings.get("draft_n"), "draft_accepted": timings.get("draft_n_accepted"),
                     "wall": wall, "content": msg.get("content") or "", "tool_calls": msg.get("tool_calls") or []})
        print(f"{rec['i']:4d} prompt {rows[-1]['prompt']:6d} gen {rows[-1]['gen']:5d} "
              f"{rows[-1]['prompt_ms'] or 0:9.0f} / {rows[-1]['predicted_ms'] or 0:9.0f} ms", flush=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print(summary(rows))


def reply(row):
    calls = json.dumps(row["tool_calls"], sort_keys=True)
    return row["content"] + re.sub(r'"id": "[^"]*"', "", calls)


def summary(rows):
    total = lambda k: sum(r[k] or 0 for r in rows)
    return (f"{len(rows)} requests: prefill {total('prompt_ms') / 1000:.1f} s, decode "
            f"{total('predicted_ms') / 1000:.1f} s ({total('gen')} tokens, {total('draft_n')} drafts), "
            f"wall {total('wall'):.1f} s")


def compare(paths):
    runs = [json.load(open(p)) for p in paths]
    ref = [reply(r) for r in runs[0]]
    for path, rows in zip(paths, runs):
        same = sum(reply(r) == x for r, x in zip(rows, ref))
        print(f"{path}: {summary(rows)}; replies matching the first run {same}/{len(ref)}")


def main(argv):
    if len(argv) == 4 and argv[0] == "record":
        record(int(argv[1]), argv[2], argv[3])
    elif len(argv) in (4, 6) and argv[0] == "replay":
        seed = int(argv[5]) if len(argv) == 6 and argv[4] == "--seed" else 1234
        replay(argv[1], argv[2], argv[3], seed)
    elif len(argv) >= 2 and argv[0] == "compare":
        compare(argv[1:])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
