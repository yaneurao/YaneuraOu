#!/usr/bin/env python3
"""Audit USE_FINNY_RESET with one independently launched engine per sample.

Uses literal OFF/ON flags recorded in build-manifest.json. Captures a raw USI
transcript, external elapsed time, process CPU counters and pre-search memory
maps. Reads smaps only while the engine is idle, never during timed search.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import select
import shlex
import signal
import subprocess
import time


SUMMARY = re.compile(r"Total time \(ms\)\s*:\s*(\d+)\s+Nodes searched\s*:\s*(\d+)\s+Nodes/second\s*:\s*(\d+)")
FLAGS = {"off": "-DUSE_LAZY_EVALUATE -DUSE_FINNY_TABLES",
         "on": "-DUSE_LAZY_EVALUATE -DUSE_FINNY_TABLES -DUSE_FINNY_RESET"}


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def build(out):
    commands, units, macros = {}, {}, {}
    for name, flags in FLAGS.items():
        command = ["make", "-C", "source", "normal", "-B", "-j8",
                   "YANEURAOU_EDITION=YANEURAOU_ENGINE_SFNN_halfka2_2304_15_64_progress8",
                   "TARGET_CPU=AVX512VNNI", "COMPILER=clang++", f"EXTRA_CPPFLAGS={flags}",
                   f"OBJDIR={out / ('obj-' + name)}", f"TARGET={out / name}"]
        commands[name] = command
        log = out / f"build-{name}.log"
        with log.open("w") as f:
            subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True)
        units[name] = {}
        for line in log.read_text().splitlines():
            if line.startswith("clang++ ") and " -c " in line:
                args = shlex.split(line)
                i = args.index("-o")
                del args[i:i + 2]
                units[name][args[args.index("-c") + 1]] = args
        assert units[name], "No compile commands were captured"
        probe = [a for a in units[name]["main.cpp"] if a not in ("-c", "-MMD", "-MP", "-flto")]
        probe[1:1] = ["-dM", "-E"]
        text = subprocess.check_output(probe, cwd="source", text=True)
        (out / f"macros-{name}.txt").write_text(text)
        macros[name] = {key: bool(re.search(r"^#define " + key + r"(?: |$)", text, re.MULTILINE))
                        for key in ("USE_LAZY_EVALUATE", "USE_FINNY_TABLES", "USE_FINNY_RESET", "USE_AVX512")}
        assert all(macros[name][key] for key in ("USE_LAZY_EVALUATE", "USE_FINNY_TABLES", "USE_AVX512"))
        assert macros[name]["USE_FINNY_RESET"] == (name == "on")
    assert set(units["off"]) == set(units["on"])
    for source, args in units["off"].items():
        assert args == [a for a in units["on"][source] if a != "-DUSE_FINNY_RESET"], source
    manifest = {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "commands": commands, "compile_units": len(units["off"]), "macros": macros,
                "all_compile_units_match_except_reset": True,
                "sha256": {n: digest(out / n) for n in FLAGS}}
    (out / "build-manifest.json").write_text(json.dumps(manifest, indent=2))


class Session:
    def __init__(self, binary, prefix):
        self.prefix = prefix
        self.log = prefix.with_suffix(".log").open("w")
        self.commands = prefix.with_suffix(".usi").open("w")
        self.p = subprocess.Popen(["taskset", "-c", "0-31", str(binary)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        self.pending = b""
        self.samples = []
        self.phase = "setup"
        self.next_sample = 0

    def send(self, text):
        self.commands.write(text)
        self.commands.flush()
        self.p.stdin.write(text.encode())
        self.p.stdin.flush()

    def stat(self):
        raw = Path(f"/proc/{self.p.pid}/stat").read_text()
        f = raw[raw.rindex(") ") + 2:].split()
        return {"monotonic": time.monotonic(), "phase": self.phase,
                "user_ticks": int(f[11]), "system_ticks": int(f[12]),
                "threads": int(f[17]), "rss_pages": int(f[21]),
                "minor_faults": int(f[7]), "major_faults": int(f[9])}

    def until(self, predicate, timeout=600):
        deadline = time.monotonic() + timeout
        lines = []
        while time.monotonic() < deadline:
            if time.monotonic() >= self.next_sample:
                self.samples.append(self.stat())
                self.next_sample = time.monotonic() + 1
            if b"\n" not in self.pending:
                if not select.select([self.p.stdout], [], [], 1)[0]:
                    continue
                block = os.read(self.p.stdout.fileno(), 65536)
                if not block:
                    raise RuntimeError("Engine closed its output; inspect " + str(self.prefix))
                self.pending += block
                continue
            raw, self.pending = self.pending.split(b"\n", 1)
            line = raw.decode("utf-8", errors="replace") + "\n"
            self.log.write(line)
            lines.append(line)
            if re.search(r"FAILED|unknown command|Unknown command|Error!|params read failed", line):
                raise RuntimeError(line)
            if predicate(line):
                self.log.flush()
                return "".join(lines)
        raise TimeoutError(str(self.prefix))

    def memory(self):
        raw = Path(f"/proc/{self.p.pid}/smaps").read_text()
        self.prefix.with_suffix(".smaps").write_text(raw)
        mappings, current = [], None
        for line in raw.splitlines():
            if re.match(r"^[0-9a-f]+-[0-9a-f]+ ", line):
                current = {"mapping": line}
                mappings.append(current)
            elif current is not None and ":" in line:
                k, v = line.split(":", 1)
                if k in ("Size", "Rss", "AnonHugePages", "KernelPageSize", "MMUPageSize"):
                    current[k + "_KiB"] = int(v.split()[0])
        return [m for m in mappings if m.get("Size_KiB", 0) >= 65536]

    def close(self):
        try:
            if self.p.poll() is None:
                self.send("quit\n")
                self.p.wait(timeout=10)
        finally:
            if self.p.poll() is None:
                self.p.kill()
                self.p.wait()
            self.log.close()
            self.commands.close()
            self.p.stdin.close()
            self.p.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/finny-reset-verified-20261030"))
    parser.add_argument("--eval-dir", type=Path, default=Path("../inbox/evalTuyo12-20261025"))
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--hash", type=int, default=64)
    parser.add_argument("--nodes", type=int, default=128000000)
    parser.add_argument("--warmup-nodes", type=int, default=16000000)
    args = parser.parse_args()
    if min(args.hash, args.nodes, args.warmup_nodes) <= 0:
        parser.error("limits must be positive")
    os.chdir(Path(__file__).resolve().parents[1])
    out, ev = args.output.resolve(), args.eval_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.build:
        build(out)
    manifest = json.loads((out / "build-manifest.json").read_text())
    assert manifest["all_compile_units_match_except_reset"]
    for name in FLAGS:
        assert manifest["sha256"][name] == digest(out / name)
        assert "EXTRA_CPPFLAGS=" + FLAGS[name] in manifest["commands"][name]
    metadata = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "measurement_runner_sha256": digest(Path(__file__).resolve()),
                "build": manifest, "flags": FLAGS, "threads": 32, "pin": "0-31",
                "hash_mib": args.hash, "nodes_per_position": args.nodes,
                "warmup_nodes_per_position": args.warmup_nodes,
                "finny_cache_survives_separate_bench_warmup": False,
                "rounds": 2, "one_process_at_a_time": True, "eval_dir": str(ev),
                "model_sha256": {n: digest(ev / n) for n in ("nn.bin", "progress.bin")}}
    setup = ("usi\ncompiler\nconfig\nsetoption name Threads value 32\n"
             f"setoption name USI_Hash value {args.hash}\nsetoption name BookFile value no_book\n"
             f"setoption name USI_OwnBook value false\nsetoption name EvalDir value {ev}\n"
             "setoption name FV_SCALE value 24\nisready\n")
    checks, records = {}, []
    for name in FLAGS:
        session = Session(out / name, out / ("check-" + name))
        try:
            session.send(setup)
            session.until(lambda l: l.strip() == "readyok")
            session.send("test nn test_accumulator 100\n")
            text = session.until(lambda l: "test_accumulator passed:" in l)
            checks[name] = re.search(r"test_accumulator passed:.*checksum (\d+)", text)[0]
            print(name, checks[name], flush=True)
        finally:
            session.close()
    assert checks["off"].split("checksum ")[1] == checks["on"].split("checksum ")[1]
    for round_index, order in enumerate((("off", "on"), ("on", "off")), 1):
        for name in order:
            session = Session(out / name, out / f"round-{round_index:02}-{name}")
            try:
                session.send(setup)
                session.until(lambda l: l.strip() == "readyok")
                session.send("getoption Threads\ngetoption USI_Hash\ngetoption FV_SCALE\nisready\n")
                session.until(lambda l: l.strip() == "readyok")
                session.phase = "warmup"
                session.send(f"bench {args.hash} 32 {args.warmup_nodes} default nodes\n")
                session.until(lambda l: bool(re.search(r"Nodes/second\s*:", l)))
                memory = session.memory()  # Engine is idle here.
                session.phase = "measurement"
                starts = []

                def done(line):
                    if line.startswith("Position: 1/4 "):
                        starts.append(session.stat())
                    return bool(re.search(r"Nodes/second\s*:", line))

                session.send(f"bench {args.hash} 32 {args.nodes} default nodes\n")
                text = session.until(done)
                end = session.stat()
                milliseconds, nodes, nps = map(int, SUMMARY.search(text).groups())
                assert nps == nodes * 1000 // milliseconds
                assert len(starts) == 1
                start = starts[0]
                cpu = (end["user_ticks"] + end["system_ticks"] - start["user_ticks"] - start["system_ticks"]) / os.sysconf("SC_CLK_TCK")
                record = {"round": round_index, "variant": name, "milliseconds": milliseconds,
                          "nodes": nodes, "nps": nps, "external_search_seconds": end["monotonic"] - start["monotonic"],
                          "search_cpu_seconds": cpu, "start": start, "end": end,
                          "large_mappings_before_bench": memory, "samples": session.samples}
                records.append(record)
                (out / "results.partial.json").write_text(json.dumps({"metadata": metadata, "checks": checks, "records": records}, indent=2))
                print(f"round {round_index} {name}: {nps:,} NPS, bench {milliseconds/1000:.3f}s, external {record['external_search_seconds']:.3f}s, CPU {cpu:.2f}s", flush=True)
            finally:
                session.close()
    summary = {name: sum(r["nodes"] for r in records if r["variant"] == name) * 1000
               / sum(r["milliseconds"] for r in records if r["variant"] == name) for name in FLAGS}
    summary["change_percent"] = 100 * (summary["on"] / summary["off"] - 1)
    metadata["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out / "results.json").write_text(json.dumps({"metadata": metadata, "checks": checks, "records": records, "summary": summary}, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit("Terminated")))
    main()
