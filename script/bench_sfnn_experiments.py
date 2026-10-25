#!/usr/bin/env python3
"""Compare independent SFNN optimizations against the adopted Finny reset code.

Uses bench's four default positions. Engines stay loaded between measurements;
only one engine searches at a time. Every sample includes a separate warmup.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import signal
import statistics
import subprocess
import threading
import time


COMMON = "-DUSE_LAZY_EVALUATE -DUSE_FINNY_TABLES -DUSE_FINNY_RESET"
VARIANTS = {
    "baseline": "-DNNUE_FT_REGS=8",
    "specialized": "-DNNUE_FT_SPECIALIZED -DNNUE_FT_REGS=8",
    "regs12": "-DNNUE_FT_REGS=12",
    "regs24": "-DNNUE_FT_REGS=24",
    "cancel": "-DNNUE_FINNY_CANCEL -DNNUE_FT_REGS=8",
    "fused16": "-DNNUE_SFNN_FUSED_FT -DNNUE_FT_REGS=8",
}
NAMES = list(VARIANTS)
# Reverse each pair of orders: every variant has the same mean time slot.
ORDER_IDS = [[0, 1, 5, 2, 4, 3], [3, 4, 2, 5, 1, 0],
             [4, 5, 3, 0, 2, 1], [1, 2, 0, 3, 5, 4]]
SUMMARY = re.compile(r"Total time \(ms\)\s*:\s*(\d+)\s+Nodes searched\s*:\s*(\d+)\s+Nodes/second\s*:\s*(\d+)")


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


class Engine:
    def __init__(self, binary, pin):
        self.process = subprocess.Popen(["taskset", "-c", pin, str(binary)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.lines = queue.Queue()

        def read():
            for line in self.process.stdout:
                self.lines.put(line)
            self.lines.put(None)

        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()

    def send(self, commands):
        self.process.stdin.write(commands)
        self.process.stdin.flush()

    def receive(self, done, timeout=600):
        deadline = time.monotonic() + timeout
        lines = []
        while True:
            line = self.lines.get(timeout=max(0.01, deadline - time.monotonic()))
            if line is None:
                raise RuntimeError("Engine exited unexpectedly:\n" + "".join(lines)[-2000:])
            lines.append(line)
            if re.search(r"FAILED|unknown command|Error!|params read failed", line):
                raise RuntimeError("Engine error:\n" + "".join(lines)[-2000:])
            if done(line):
                return "".join(lines)
            if time.monotonic() >= deadline:
                raise TimeoutError("Engine output timed out")

    def close(self):
        if self.process.poll() is None:
            try:
                self.send("stop\nquit\n")
                self.process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.process.stdout.close()
        self.process.stdin.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--output", type=Path, default=Path("build/sfnn-experiments"))
    parser.add_argument("--bin-dir", type=Path)
    parser.add_argument("--checks-from", type=Path)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--pin", default="2")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--nodes", type=int, default=2000000)
    parser.add_argument("--warmup-nodes", type=int, default=500000)
    parser.add_argument("--games", type=int, default=100)
    args = parser.parse_args()
    if min(args.threads, args.rounds, args.nodes, args.warmup_nodes, args.games) <= 0:
        parser.error("counts must be positive")
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    bin_dir = args.bin_dir.resolve() if args.bin_dir else output
    eval_dir = args.eval_dir.resolve()
    previous_path = args.checks_from.resolve() if args.checks_from else None
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    binaries = {n: bin_dir / n for n in NAMES}
    edition = args.arch if args.arch.startswith("YANEURAOU_ENGINE_") else "YANEURAOU_ENGINE_" + args.arch
    builds = {}
    for name, extra in VARIANTS.items():
        flags = (COMMON + " " + extra).strip()
        cmd = ["make", "-C", "source", "normal", "-j8", f"YANEURAOU_EDITION={edition}",
               "TARGET_CPU=AVX512VNNI", "COMPILER=clang++", f"EXTRA_CPPFLAGS={flags}",
               f"OBJDIR={output / ('obj-' + name)}", f"TARGET={binaries[name]}"]
        builds[name] = cmd
        if args.build:
            print("BUILD", name, flush=True)
            with (output / f"build-{name}.log").open("w") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    metadata = {
        "architecture": edition, "eval_dir": str(eval_dir), "target_cpu": "AVX512VNNI",
        "common_cppflags": COMMON, "variant_cppflags": VARIANTS, "build_commands": builds,
        "threads": args.threads, "pin": args.pin, "rounds": args.rounds,
        "nodes_per_position": args.nodes, "warmup_nodes_per_position": args.warmup_nodes,
        "fv_scale": 24, "hash_mib": 64, "positions": "bench default (4 positions)",
        "persistent_engines": True,
        "binary_sha256": {n: digest(p) for n, p in binaries.items()},
        "model_sha256": {n: digest(eval_dir / n) for n in ("nn.bin", "progress.bin")},
        "lscpu": subprocess.check_output(["lscpu"], text=True),
        "compiler": subprocess.check_output(["clang++", "--version"], text=True),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    checks = {}
    if previous_path:
        previous = json.loads(previous_path.read_text())
        for key in ("architecture", "binary_sha256", "model_sha256", "common_cppflags", "variant_cppflags"):
            if previous["metadata"][key] != metadata[key]:
                raise RuntimeError(f"Cannot reuse checks: {key} differs")
        checks = previous["checks"]
        if set(checks) != set(NAMES):
            raise RuntimeError("Incomplete prior checks")
        metadata["checks_from"] = str(previous_path)

    engines = {}
    records = []
    signature = None
    checksum = None
    try:
        for name in NAMES:
            engine = engines[name] = Engine(binaries[name], args.pin)
            engine.send(f"setoption name Threads value {args.threads}\n"
                        "setoption name USI_Hash value 64\nsetoption name BookFile value no_book\n"
                        "setoption name USI_OwnBook value false\nsetoption name PvInterval value 100000\n"
                        f"setoption name EvalDir value {eval_dir}\nsetoption name FV_SCALE value 24\nisready\n")
            boot = engine.receive(lambda line: line.strip() == "readyok")
            (output / f"startup-{name}.log").write_text(boot)
            if not previous_path:
                engine.send(f"test nn test_accumulator {args.games}\n")
                check = engine.receive(lambda line: "test_accumulator passed:" in line)
                (output / f"check-{name}.log").write_text(check)
                match = re.search(r"test_accumulator passed:.*checksum (\d+)", check)
                if checksum is not None and checksum != match[1]:
                    raise RuntimeError(f"Score checksum differs: {name}")
                checksum = match[1]
                checks[name] = match[0]
                print(name, match[0], flush=True)
        for round_index in range(args.rounds):
            for index in ORDER_IDS[round_index % len(ORDER_IDS)]:
                name = NAMES[index]
                engine = engines[name]
                engine.send(f"bench 64 {args.threads} {args.warmup_nodes} default nodes\n"
                            f"bench 64 {args.threads} {args.nodes} default nodes\n")
                count = 0

                def done(line):
                    nonlocal count
                    if re.search(r"Nodes/second\s*:", line):
                        count += 1
                    return count == 2

                stdout = engine.receive(done)
                (output / f"round-{round_index + 1:02}-{name}.log").write_text(stdout)
                summaries = SUMMARY.findall(stdout)
                if len(summaries) != 2:
                    raise RuntimeError("Expected warmup and measured summaries")
                milliseconds, nodes, nps = map(int, summaries[-1])
                infos = re.findall(r"^info depth .*$", stdout.split("Total time (ms)", 1)[-1], re.MULTILINE)
                infos = [re.sub(r"\b(?:nps|time) \d+ ?", "", line) for line in infos]
                moves = re.findall(r"^bestmove .*$", stdout, re.MULTILINE)[-4:]
                current = (nodes, moves, infos)
                if args.threads == 1 and signature is not None and current != signature:
                    raise RuntimeError(f"Search tree differs: {name}, round {round_index + 1}")
                signature = current
                records.append({"round": round_index + 1, "variant": name, "nodes": nodes,
                                "milliseconds": milliseconds, "nps": nps, "bestmoves": moves})
                print(f"round {round_index + 1} {name:12} {nps:10,} NPS", flush=True)
                (output / "results.partial.json").write_text(json.dumps(
                    {"metadata": metadata, "checks": checks, "records": records}, indent=2))
        baseline = {r["round"]: r["nps"] for r in records if r["variant"] == "baseline"}
        summary = {}
        for name in NAMES:
            rs = [r for r in records if r["variant"] == name]
            ratios = [100 * (r["nps"] / baseline[r["round"]] - 1) for r in rs]
            summary[name] = {
                "aggregate_nps": sum(r["nodes"] for r in rs) * 1000 / sum(r["milliseconds"] for r in rs),
                "mean_nps": statistics.mean(r["nps"] for r in rs),
                "stdev_nps": statistics.stdev(r["nps"] for r in rs) if len(rs) > 1 else 0,
                "paired_mean_change_percent": statistics.mean(ratios),
                "paired_standard_error_percent": statistics.stdev(ratios) / len(rs)**0.5 if len(rs) > 1 else None,
            }
        (output / "results.json").write_text(json.dumps(
            {"metadata": metadata, "checks": checks, "records": records, "summary": summary}, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        for engine in engines.values():
            engine.close()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit("Terminated")))
    main()
