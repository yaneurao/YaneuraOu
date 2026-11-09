#!/usr/bin/env python3
"""MakefileのPGOターゲットを補助する。Clang/GCCともtournamentで訓練する。"""

import argparse
import json
import os
from pathlib import Path
import queue
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time


def command(value):
    return [value] if Path(value).is_file() else shlex.split(value)


def is_msys_python():
    return os.name == "nt" and bool(os.environ.get("MSYSTEM"))


def shell_command(argv):
    # MSYS2からWindows版Pythonを呼んだ場合も、/mingw64やcompiler wrapperを扱う。
    if is_msys_python():
        argv = [argv[0].replace("\\", "/"), *argv[1:]]
        return [shutil.which("bash") or "bash", "-c", "exec " + shlex.join(argv)]
    return argv


def native_path(value):
    value = str(value)
    if is_msys_python() and value.startswith("/") and not value.startswith("//"):
        value = subprocess.check_output(["cygpath", "-w", value], text=True).strip()
    return Path(value).expanduser().resolve()


def make_path(value):
    # 作業ツリーの親パスに空白があっても、makeには相対パスで渡す。
    try:
        value = os.path.relpath(value)
    except ValueError:  # Windowsの別ドライブ
        value = str(value)
    value = value.replace("\\", "/")
    if any(c.isspace() for c in value):
        raise RuntimeError("PGO_DIRには、sourceからの相対パスに空白がない場所を指定してください。")
    if is_msys_python() and re.match(r"^[A-Za-z]:/", value):
        value = subprocess.check_output(["cygpath", "-u", value], text=True).strip()
    return value


def run(argv, **kwargs):
    # 再帰makeへjobserverを引き継ぐ。呼び出し元のレシピには「+」が必要。
    print("[PGO] " + shlex.join(argv), flush=True)
    return subprocess.run(shell_command(argv), check=True, close_fds=False, **kwargs)


def compiler_info(compiler):
    version = subprocess.check_output(shell_command(command(compiler) + ["--version"]), text=True)
    if "clang" in version.lower():
        return "clang", version
    if re.search(r"gcc|g\+\+|Free Software Foundation", version, re.I):
        return "gcc", version
    raise RuntimeError("PGOはClangまたはGCCで使用してください: " + compiler)


def make_environment():
    env = os.environ.copy()
    flags = []
    if is_msys_python():
        # MSYSのjobserverのfdはWindows版Pythonを通して継承できない。
        # 親の並列数で新しいjobserverを作り、以後のMSYS make同士で共有する。
        makeflags = env.get("MAKEFLAGS", "")
        jobs = re.search(r"(?:^|\s)-j\s*(\d+)(?=\s|$)", makeflags)
        flags = ["-j" + jobs[1]] if jobs else []
        env["MAKEFLAGS"] = re.sub(r"(?:^|\s)--jobserver-(?:auth|fds)=\S+", "", makeflags)
    return flags, env


def check_profile_link(args, work):
    # MSYS2ではcompiler-rt未導入や配置不整合を、本体の長いビルドより前に検出する。
    source = work / "profile-link-check.cpp"
    source.write_text("int main() { return 0; }\n", encoding="utf-8")
    argv = command(args.compiler) + shlex.split(args.cppflags) + shlex.split(args.ldflags) + [
        "-fprofile-generate=" + (work / "profiles").as_posix(), "-fprofile-update=atomic",
        str(source), "-o", str(work / "profile-link-check.exe"),
    ]
    result = subprocess.run(shell_command(argv), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    (work / "profile-link-check.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        print(result.stdout, end="", flush=True)
        raise RuntimeError(
            "ClangのPGOリンク確認に失敗しました。MSYS2 MINGW64では "
            "pacman -S mingw-w64-x86_64-compiler-rt を確認してください。"
            "導入済みならClangとcompiler-rtのバージョン・配置の整合を確認してください。"
            "詳細: " + str(work / "profile-link-check.log"))


def find_profdata(compiler, version, override):
    if override:
        result = command(override)
        subprocess.run(shell_command(result + ["--version"]), check=True, stdout=subprocess.DEVNULL)
        return result
    major = re.search(r"clang version (\d+)", version)
    suffix = ".exe" if os.name == "nt" else ""
    candidates = []
    for part in command(compiler):
        if "clang" in Path(part).name:
            exe = shutil.which(part)
            if exe:
                candidates.append(str(Path(exe).resolve().parent / ("llvm-profdata" + suffix)))
    if major:
        candidates.append("llvm-profdata-" + major[1])
    candidates.append("llvm-profdata")
    for candidate in candidates:
        exe = shutil.which(candidate)
        if exe:
            return [exe]
    raise RuntimeError("llvm-profdataが見つかりません。PGO_PROFDATAで指定してください。")


def publish(binary, target):
    # 最終ビルドの成功後に置き換える。失敗しても従来の実行ファイルは残す。
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=target.name + ".pgo-", dir=target.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        shutil.copy2(binary, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_state(path, state):
    fd, name = tempfile.mkstemp(prefix="state-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(state, stream, ensure_ascii=False, indent=2)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def build(args, state, generate):
    root = Path(state["run_dir"])
    profile = root / "profiles"
    # GCCのgcda名はオブジェクト/実行ファイルのパスに依存するため両段階で統一する。
    obj = root / ("obj" if state["kind"] == "gcc" else "obj-gen" if generate else "obj-use")
    # makeのレシピを経由するコンパイラにも、パス中の空白を一つの引数として渡す。
    if generate:
        flags = ["-fprofile-generate=" + profile.as_posix(), "-fprofile-update=atomic"]
    elif state["kind"] == "clang":
        flags = ["-fprofile-use=" + (root / "search.profdata").as_posix()]
    else:
        flags = ["-fprofile-use=" + profile.as_posix(), "-fprofile-correction", "-Werror=missing-profile"]
    cppflags = args.cppflags + " " + " ".join(shlex.quote(x) for x in flags)
    parallel_flags, env = make_environment()
    argv = command(args.make) + parallel_flags + [
        "-B", "tournament", "COMPILER=" + args.compiler,
        "YANEURAOU_EDITION=" + args.edition, "TARGET_CPU=" + args.cpu,
        "OBJDIR=" + make_path(obj), "TARGET=" + make_path(state["binary"]),
        "EXTRA_CPPFLAGS=" + cppflags, "EXTRA_LDFLAGS=" + args.ldflags,
    ]
    (root / ("build-generate.json" if generate else "build-use.json")).write_text(
        json.dumps(argv, ensure_ascii=False, indent=2), encoding="utf-8")
    run(argv, env=env)


def train(args, state):
    binary = state["binary"]
    # Clangは実行先に依存せずこのディレクトリへ出力する。GCCはコンパイル時の指定を使う。
    env = os.environ.copy()
    env["LLVM_PROFILE_FILE"] = str(Path(state["run_dir"]) / "profiles" / "search-%p.profraw")
    try:
        engine = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                  errors="replace", bufsize=1, env=env)
    except OSError as error:
        raise RuntimeError("訓練用エンジンを実行できません。PGOは実行可能な環境で行ってください。") from error
    output = queue.Queue()

    def reader():
        for line in engine.stdout:
            output.put(line)
        output.put(None)

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()
    deadline = time.monotonic() + args.timeout
    log = open(Path(state["run_dir"]) / "training.log", "w", encoding="utf-8")

    def send(text):
        log.write("> " + text + "\n")
        log.flush()
        engine.stdin.write(text + "\n")
        engine.stdin.flush()

    def until(predicate):
        lines = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("PGO訓練がタイムアウトしました。PGO_TIMEOUTで秒数を指定できます。")
            try:
                line = output.get(timeout=remaining)
            except queue.Empty:
                raise RuntimeError("PGO訓練がタイムアウトしました。PGO_TIMEOUTを確認してください。")
            if line is None:
                raise RuntimeError("PGO訓練が途中で終了しました。training.logを確認してください。")
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
            if re.search(r"Error!|params read failed|unknown command|No such option", line, re.I):
                raise RuntimeError("PGO訓練に失敗しました。評価関数とPGO_EVAL_DIRを確認してください。")
            lines.append(line)
            if predicate(line):
                return "".join(lines)

    try:
        send("usi")
        usi = until(lambda s: s.strip() == "usiok")
        options = set(re.findall(r"^option name (.+?) type ", usi, re.M))
        limits = args.bench.split()
        for option, index in (("USI_Hash", 0), ("Threads", 1)):
            if option in options and len(limits) > index and limits[index].isdigit():
                send("setoption name " + option + " value " + limits[index])
        if "EvalDir" in options and args.eval_dir:
            send("setoption name EvalDir value " + str(native_path(args.eval_dir)))
        if "BookFile" in options:
            send("setoption name BookFile value no_book")
        if "USI_OwnBook" in options:
            send("setoption name USI_OwnBook value false")
        send("isready")
        until(lambda s: s.strip() == "readyok")
        send("bench " + args.bench)
        until(lambda s: bool(re.search(r"Nodes/second\s*:", s)))
        send("quit")
        engine.wait(timeout=max(1, deadline - time.monotonic()))
        if engine.returncode:
            raise RuntimeError("PGO訓練エンジンの終了コード: " + str(engine.returncode))
    finally:
        if engine.poll() is None:
            engine.kill()
            engine.wait()
        reader_thread.join(timeout=1)
        engine.stdin.close()
        engine.stdout.close()
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["all", "generate", "use"])
    for name in ["make", "compiler", "edition", "cpu", "target", "work-dir"]:
        parser.add_argument("--" + name, required=True)
    for name in ["cppflags", "ldflags", "eval-dir", "profdata"]:
        parser.add_argument("--" + name, default="")
    parser.add_argument("--bench", default="1024 1 14 default depth")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    # 「+」付きレシピはmake -n/-tでも実行されるため、生成物を作らず手順だけ示す。
    makeflags = os.environ.get("MAKEFLAGS", "").split()
    short_flags = makeflags[0] if makeflags and not makeflags[0].startswith("-") and "=" not in makeflags[0] else ""
    if any(c in short_flags for c in "ntq") or "--dry-run" in makeflags:
        print("[PGO] dry run: tournament計測用ビルド → bench " + args.bench
              + " → プロファイル集約 → tournament本番ビルド → " + args.target)
        return
    if args.timeout <= 0 or any(c in args.bench + args.eval_dir for c in "\r\n"):
        raise RuntimeError("PGO_TIMEOUTは正の秒数、PGO_BENCH/PGO_EVAL_DIRは1行で指定してください。")
    print("[PGO] START: " + args.edition + " / " + args.cpu + " / " + args.mode, flush=True)
    kind, version = compiler_info(args.compiler)
    profdata = find_profdata(args.compiler, version, args.profdata) if kind == "clang" and args.mode != "generate" else None
    root = native_path(args.work_dir)
    root.mkdir(parents=True, exist_ok=True)
    state_file = root / "state.json"
    config = {name: getattr(args, name) for name in ["compiler", "edition", "cpu", "cppflags", "ldflags"]}
    config.update(source=str(Path.cwd().resolve()), version=version)
    if args.mode == "use":
        if not state_file.is_file():
            raise RuntimeError("先にmake profgenで計測用ビルドと訓練を行うか、make pgoを使ってください。")
        state = json.loads(state_file.read_text(encoding="utf-8"))
        if state["config"] != config:
            raise RuntimeError("profgen時とビルド設定が異なります。同じ設定でmake pgoをやり直してください。")
    else:
        work = Path(tempfile.mkdtemp(prefix="run-", dir=root))
        (work / "profiles").mkdir()
        suffix = Path(args.target).suffix if Path(args.target).suffix.lower() == ".exe" else ""
        state = {"config": config, "kind": kind, "run_dir": str(work),
                 "binary": str(work / ("engine" + suffix))}
        if os.environ.get("MSYSTEM") and kind == "clang":
            check_profile_link(args, work)
        build(args, state, generate=True)
        write_state(state_file, state)
        if args.mode == "generate":
            publish(Path(state["binary"]), Path(args.target).absolute())
            print("[PGO] DONE: instrumented build " + str(Path(args.target).absolute()), flush=True)
            print("[PGO] 計測用エンジンを訓練してから、同じ設定でmake profuseを実行してください。")
            return
        train(args, state)
    work = Path(state["run_dir"])
    if kind == "clang":
        raw = sorted((work / "profiles").glob("*.profraw"))
        if not raw:
            raise RuntimeError("profrawがありません。先に計測用エンジンを実行してください。")
        run(profdata + ["merge", "-output=" + str(work / "search.profdata")] + list(map(str, raw)))
    elif not list((work / "profiles").rglob("*.gcda")):
        raise RuntimeError("gcdaがありません。先に計測用エンジンを実行してください。")
    build(args, state, generate=False)
    publish(Path(state["binary"]), Path(args.target).absolute())
    print("[PGO] DONE: " + str(Path(args.target).absolute()), flush=True)
    print("[PGO] プロファイル・オブジェクト・訓練ログ: " + str(work))


if __name__ == "__main__":
    # Windows版Pythonのパイプ出力は既定でCP932になる。BuilderはUTF-8で読む。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try:
        main()
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        print("[PGO] ERROR: " + str(error), file=sys.stderr)
        sys.exit(1)
