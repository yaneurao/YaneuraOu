#!/usr/bin/env python3
"""Long paired bench comparison of existing baseline and Finny mask binaries.

Reuses prior accumulator checks only after verifying binary and model SHA256.
Engines remain loaded; exactly one engine searches at a time. No engine source
is changed. The fixed sample count avoids stopping when a desired result appears.
"""

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import time

from bench_sfnn_experiments import Engine, SUMMARY, digest


NAMES = ("baseline", "mask")


def summarize(records):
    complete = sorted(set(r["round"] for r in records
                          if sum(s["round"] == r["round"] for s in records) == 2))
    pairs = {i: {r["variant"]: r for r in records if r["round"] == i} for i in complete}
    ratios = [100 * (pairs[i]["mask"]["nps"] / pairs[i]["baseline"]["nps"] - 1)
              for i in complete]
    summary = {}
    for name in NAMES:
        rows = [r for r in records if r["variant"] == name]
        if rows:
            summary[name] = {
                "samples": len(rows),
                "aggregate_nps": sum(r["nodes"] for r in rows) * 1000
                                 / sum(r["milliseconds"] for r in rows),
                "mean_nps": statistics.mean(r["nps"] for r in rows),
                "stdev_nps": statistics.stdev(r["nps"] for r in rows) if len(rows) > 1 else None,
            }
    if ratios:
        summary["paired"] = {"pairs": len(ratios), "mean_change_percent": statistics.mean(ratios),
                             "median_change_percent": statistics.median(ratios),
                             "individual_changes_percent": ratios}
        if len(ratios) > 1:
            se = statistics.stdev(ratios) / math.sqrt(len(ratios))
            summary["paired"]["standard_error_points"] = se
            # Student t, two-sided 95%, df=11 for the predefined 12-pair run.
            if len(ratios) == 12:
                mean = statistics.mean(ratios)
                summary["paired"]["ci95_percent"] = [mean - 2.200985 * se, mean + 2.200985 * se]
    if all(n in summary for n in NAMES):
        summary["aggregate_change_percent"] = 100 * (summary["mask"]["aggregate_nps"]
                                                     / summary["baseline"]["aggregate_nps"] - 1)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("build/finny-reset-largepages-20261030/results.json"))
    parser.add_argument("--bin-dir", type=Path, default=Path("build/finny-reset-largepages-20261030"))
    parser.add_argument("--output", type=Path, default=Path("build/finny-mask-long-20261030"))
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--pin", default="0-31")
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--nodes", type=int, default=128000000)
    parser.add_argument("--warmup-nodes", type=int, default=16000000)
    args = parser.parse_args()
    if min(args.threads, args.rounds, args.nodes, args.warmup_nodes) <= 0:
        parser.error("counts must be positive")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    reference = args.reference.resolve()
    prior = json.loads(reference.read_text())
    bins = {n: args.bin_dir.resolve() / n for n in NAMES}
    eval_dir = Path(prior["metadata"]["eval_dir"])
    binary_hashes = {n: digest(bins[n]) for n in NAMES}
    model_hashes = {n: digest(eval_dir / n) for n in prior["metadata"]["model_sha256"]}
    if any(binary_hashes[n] != prior["metadata"]["binary_sha256"][n] for n in NAMES):
        raise RuntimeError("Binaries differ from the validated reference")
    if model_hashes != prior["metadata"]["model_sha256"]:
        raise RuntimeError("Model differs from the validated reference")
    checks = {n: prior["checks"][n] for n in NAMES}
    checksums = [re.search(r"test_accumulator passed:.*checksum (\d+)", checks[n]) for n in NAMES]
    if any(m is None for m in checksums) or checksums[0][1] != checksums[1][1]:
        raise RuntimeError("Prior correctness checks missing or inconsistent")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "reference": str(reference), "checks_reused_from": str(reference),
        "architecture": prior["metadata"]["architecture"], "eval_dir": str(eval_dir),
        "target_cpu": prior["metadata"]["target_cpu"],
        "common_cppflags": prior["metadata"]["common_cppflags"],
        "variant_cppflags": {n: prior["metadata"]["variant_cppflags"][n] for n in NAMES},
        "binary_sha256": binary_hashes, "model_sha256": model_hashes,
        "binaries": {n: str(p) for n, p in bins.items()}, "threads": args.threads,
        "pin": args.pin, "rounds": args.rounds, "nodes_per_position": args.nodes,
        "warmup_nodes_per_position": args.warmup_nodes, "hash_mib": 64, "fv_scale": 24,
        "positions": "bench default (4 positions)", "persistent_engines": True,
        "order": "AB, BA alternating; count fixed before measurement",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "lscpu": subprocess.check_output(["lscpu"], text=True),
        "processes_before": subprocess.check_output(["ps", "-eo", "pid,comm,pcpu", "--sort=-pcpu"], text=True),
    }
    engines, records = {}, []
    try:
        for name in NAMES:
            engine = engines[name] = Engine(bins[name], args.pin)
            engine.send(f"setoption name Threads value {args.threads}\n"
                        "setoption name USI_Hash value 64\nsetoption name BookFile value no_book\n"
                        "setoption name USI_OwnBook value false\nsetoption name PvInterval value 100000\n"
                        f"setoption name EvalDir value {eval_dir}\nsetoption name FV_SCALE value 24\nisready\n")
            (output / f"startup-{name}.log").write_text(engine.receive(lambda line: line.strip() == "readyok"))
            print(f"READY {name}; binary/model hashes and prior 100-game check verified", flush=True)
        for round_index in range(args.rounds):
            order = NAMES if round_index % 2 == 0 else NAMES[::-1]
            for name in order:
                engine = engines[name]
                started = time.monotonic()
                count = 0

                def done(line):
                    nonlocal count
                    count += bool(re.search(r"Nodes/second\s*:", line))
                    return count == 2

                engine.send(f"bench 64 {args.threads} {args.warmup_nodes} default nodes\n"
                            f"bench 64 {args.threads} {args.nodes} default nodes\n")
                stdout = engine.receive(done, timeout=1200)
                (output / f"round-{round_index + 1:02}-{name}.log").write_text(stdout)
                matches = SUMMARY.findall(stdout)
                if len(matches) != 2:
                    raise RuntimeError("Expected one warmup and one measured summary")
                milliseconds, nodes, nps = map(int, matches[-1])
                records.append({"round": round_index + 1, "variant": name, "milliseconds": milliseconds,
                                "nodes": nodes, "nps": nps, "wall_seconds": time.monotonic() - started,
                                "bestmoves": re.findall(r"^bestmove .*$", stdout, re.MULTILINE)[-4:]})
                result = {"metadata": metadata, "checks": checks, "records": records,
                          "summary": summarize(records)}
                (output / "results.partial.json").write_text(json.dumps(result, indent=2))
                print(f"round {round_index + 1:02}/{args.rounds} {name:8} {nps:12,} NPS ({milliseconds / 1000:.2f}s)", flush=True)
            print("PAIRED " + json.dumps(summarize(records)["paired"]), flush=True)
        metadata["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (output / "results.json").write_text(json.dumps(
            {"metadata": metadata, "checks": checks, "records": records, "summary": summarize(records)}, indent=2))
        print(json.dumps(summarize(records), indent=2), flush=True)
    finally:
        for engine in engines.values():
            engine.close()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit("Terminated")))
    main()
