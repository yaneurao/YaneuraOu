#!/usr/bin/env python3
"""Build, check and benchmark the baseline and Finny reset configurations.

Example (from the repository root):
  python3 script/bench_finny_reset.py --build \
    --eval-dir ../inbox/evalTuyo12-20261025 \
    --arch SFNN_halfka2_2304_15_64_progress8

Search timing excludes model loading. Runs are sequential,
CPU pinned, and use a balanced order. Raw logs and JSON are retained.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time


VARIANTS = {
    "baseline": "",
    "finny_reset": "-DUSE_FINNY_RESET",
}
# Alternate the first variant to balance timing and temperature drift.
ORDERS = [["baseline", "finny_reset"], ["finny_reset", "baseline"]]
SUMMARY = re.compile(
    r"Total time \(ms\)\s*:\s*(\d+)\s+Nodes searched\s*:\s*(\d+)\s+Nodes/second\s*:\s*(\d+)"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--target-cpu", default="AVX512VNNI")
    parser.add_argument("--compiler", default="clang++")
    parser.add_argument("--pin", default="2", help="taskset CPU list; use one logical CPU per physical core")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--nodes", type=int, default=1000000)
    parser.add_argument("--warmup-nodes", type=int, default=250000)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--extra-cppflags", default="-DUSE_LAZY_EVALUATE -DUSE_FINNY_TABLES", help="common build flags for both variants")
    parser.add_argument("--bin-dir", type=Path, help="reuse the two binaries in this directory")
    parser.add_argument("--checks-from", type=Path, help="reuse accumulator checks from a previous results.json")
    parser.add_argument("--output", type=Path, default=Path("build/finny-reset"))
    args = parser.parse_args()
    if min(args.rounds, args.nodes, args.warmup_nodes, args.games, args.threads) <= 0:
        parser.error("rounds, nodes, warmup-nodes, games and threads must be positive")
    root = Path(__file__).resolve().parents[1]
    eval_dir = args.eval_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    edition = args.arch if args.arch.startswith("YANEURAOU_ENGINE_") else "YANEURAOU_ENGINE_" + args.arch
    bin_dir = args.bin_dir.resolve() if args.bin_dir else output
    checks_from = args.checks_from.resolve() if args.checks_from else None
    binaries = {name: bin_dir / name for name in VARIANTS}
    build_commands = {}
    for name, flags in VARIANTS.items():
        flags = " ".join(part for part in (args.extra_cppflags, flags) if part)
        cmd = ["make", "-C", "source", "normal", "-j8", f"YANEURAOU_EDITION={edition}",
               f"TARGET_CPU={args.target_cpu}", f"COMPILER={args.compiler}",
               f"OBJDIR={output / ('obj-' + name)}", f"TARGET={binaries[name]}",
               f"EXTRA_CPPFLAGS={flags}"]
        build_commands[name] = cmd
        if args.build:
            with (output / f"build-{name}.log").open("w") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        if not binaries[name].is_file():
            raise RuntimeError(f"Missing {binaries[name]}; use --build")

    setup = (
        f"setoption name Threads value {args.threads}\n"
        "setoption name USI_Hash value 64\n"
        "setoption name BookFile value no_book\n"
        "setoption name USI_OwnBook value false\n"
        "setoption name PvInterval value 100000\n"
        f"setoption name EvalDir value {eval_dir}\n"
        "setoption name FV_SCALE value 24\nisready\n"
    )

    def run(name, commands, log_name):
        started = time.monotonic()
        result = subprocess.run(["taskset", "-c", str(args.pin), str(binaries[name])],
                                input=setup + commands + "quit\n", text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
        (output / log_name).write_text(result.stdout)
        if result.returncode or re.search(r"FAILED|unknown command|Error!|params read failed", result.stdout):
            raise RuntimeError(f"{name} failed; see {output / log_name}")
        return result.stdout, time.monotonic() - started

    metadata = {
        "architecture": edition, "eval_dir": str(eval_dir), "fv_scale": 24,
        "target_cpu": args.target_cpu, "pin": args.pin, "threads": args.threads, "hash_mib": 64,
        "nodes_per_position": args.nodes, "positions": "benchmark.cpp Defaults (4 positions)",
        "warmup_nodes_per_position": args.warmup_nodes, "rounds": args.rounds,
        "lscpu": subprocess.check_output(["lscpu"], text=True),
        "compiler": subprocess.check_output([args.compiler, "--version"], text=True),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "build_commands": build_commands,
        "extra_cppflags": args.extra_cppflags,
        "binary_sha256": {name: hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
                          for name, path in binaries.items()},
        "model_sha256": {p.name: hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
                         for p in (eval_dir / "nn.bin", eval_dir / "progress.bin") if p.exists()},
    }
    previous = json.loads(checks_from.read_text()) if checks_from else None
    checks = previous["checks"] if previous else {}
    if checks_from:
        for key in ("binary_sha256", "model_sha256", "architecture", "extra_cppflags"):
            default = "" if key == "extra_cppflags" else None
            if previous["metadata"].get(key, default) != metadata[key]:
                raise RuntimeError(f"Cannot reuse checks: {key} differs")
        if set(checks) != set(VARIANTS) or any("test_accumulator passed:" not in s for s in checks.values()):
            raise RuntimeError("Previous accumulator checks are incomplete")
        metadata["checks_from"] = str(checks_from)
    checksum = None
    for name in VARIANTS:
        if checks_from:
            continue
        stdout, _ = run(name, f"test nn test_accumulator {args.games}\n", f"check-{name}.log")
        match = re.search(r"test_accumulator passed:.*checksum (\d+)", stdout)
        if not match:
            raise RuntimeError(f"No successful accumulator test for {name}")
        if checksum is not None and checksum != match[1]:
            raise RuntimeError("Accumulator-test score checksum differs between variants")
        checksum = match[1]
        checks[name] = match[0]
        print(name, match[0], flush=True)

    records = []
    signature = None
    for round_index in range(args.rounds):
        for name in ORDERS[round_index % len(ORDERS)]:
            stdout, wall_seconds = run(
                name,
                f"bench 64 {args.threads} {args.warmup_nodes} default nodes\n"
                f"bench 64 {args.threads} {args.nodes} default nodes\n",
                f"round-{round_index + 1:02}-{name}.log")
            summaries = SUMMARY.findall(stdout)
            if len(summaries) != 2:
                raise RuntimeError("Expected warmup and measured benchmark summaries")
            milliseconds, nodes, nps = map(int, summaries[-1])
            # One thread + fixed nodes must search the same tree in every build.
            bestmoves = re.findall(r"^bestmove .*$", stdout, re.MULTILINE)[-4:]
            # Include all search-info lines except timing/NPS to check scores/PVs.
            measured = stdout.split("Total time (ms)", 1)[-1]
            infos = re.findall(r"^info depth .*$", measured, re.MULTILINE)
            infos = [re.sub(r"\b(?:nps|time) \d+ ?", "", line) for line in infos]
            current_signature = (nodes, bestmoves, infos)
            if args.threads == 1 and signature is not None and signature != current_signature:
                raise RuntimeError(f"Search tree differs: round {round_index + 1}, {name}")
            signature = current_signature
            record = {"round": round_index + 1, "variant": name, "milliseconds": milliseconds,
                      "nodes": nodes, "nps": nps, "wall_seconds": wall_seconds,
                      "bestmoves": bestmoves}
            records.append(record)
            print(f"round {round_index + 1} {name:7} {nps:9,} NPS", flush=True)
            (output / "results.partial.json").write_text(json.dumps(
                {"metadata": metadata, "checks": checks, "records": records}, indent=2))

    summary = {}
    baseline = {r["round"]: r["nps"] for r in records if r["variant"] == "baseline"}
    for name in VARIANTS:
        rows = [r for r in records if r["variant"] == name]
        nps = [r["nps"] for r in rows]
        ratios = [r["nps"] / baseline[r["round"]] for r in rows]
        mean_ratio = statistics.mean(ratios)
        summary[name] = {
            "aggregate_nps": sum(r["nodes"] for r in rows) * 1000 / sum(r["milliseconds"] for r in rows),
            "median_nps": statistics.median(nps), "mean_nps": statistics.mean(nps),
            "stdev_nps": statistics.stdev(nps) if len(nps) > 1 else 0,
            "paired_mean_change_percent": (mean_ratio - 1) * 100,
            "paired_standard_error_percent": statistics.stdev(ratios) / math.sqrt(len(ratios)) * 100
            if len(ratios) > 1 else None,
        }
    result = {"metadata": metadata, "checks": checks, "summary": summary, "records": records}
    (output / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
