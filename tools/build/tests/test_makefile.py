"""Makefileのコマンド変更検出と、生成ヘッダーの更新日時を検証する。"""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
MAKE = shutil.which("make")
COMPILER = shutil.which("clang++") or shutil.which("g++")


@unittest.skipUnless(MAKE and COMPILER and os.name == "posix", "requires POSIX make and a C++ compiler")
class MakeCommandTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="yo make ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        shutil.copy2(ROOT / "source/Makefile", self.source / "Makefile")
        (self.source / "value.h").write_text("#define HEADER_VALUE 1\n")
        (self.source / "probe.cpp").write_text(
            '#include "value.h"\n#include <cstdio>\n'
            '#ifndef FLAG_VALUE\n#define FLAG_VALUE 0\n#endif\n'
            'int main(){std::printf("%d %d\\n", FLAG_VALUE, HEADER_VALUE);}\n'
        )
        self.base = [
            MAKE, "-j2", "ARCH_GEN_SCRIPT=", "LTOFLAGS=",
            "YANEURAOU_EDITION=YANEURAOU_ENGINE_MATERIAL", "TARGET_CPU=OTHER",
            "SOURCES=probe.cpp", "CUDA_SOURCES=", "OBJC_SOURCES=",
            "OBJDIR=../obj", "TARGET=../engine", "COMPILER=" + COMPILER,
        ]

    def make(self, *settings, goal="normal", success=True, env=None):
        result = subprocess.run(
            [*self.base, *settings, goal], cwd=self.source, env=env,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stdout)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def stamp(self):
        return tuple((self.root / p).stat().st_mtime_ns for p in ("obj/probe.o", "engine"))

    def output(self):
        return subprocess.check_output([str(self.root / "engine")], text=True).strip()

    def test_same_settings_and_link_only_changes(self):
        self.make()
        before = self.stamp()
        self.make()
        self.assertEqual(self.stamp(), before)
        self.make("EXTRA_LDFLAGS=-L.")
        after = self.stamp()
        self.assertEqual(after[0], before[0])
        self.assertNotEqual(after[1], before[1])

    def test_compile_flags_and_header_dependencies(self):
        flags = 'EXTRA_CPPFLAGS=-DFLAG_VALUE=7 -DENGINE_VERSION=\\"fixture\\"'
        self.make()
        before = self.stamp()
        self.make(flags)
        self.assertNotEqual(self.stamp()[0], before[0])
        self.assertEqual(self.output(), "7 1")
        before = self.stamp()
        self.make(flags)
        self.assertEqual(self.stamp(), before)
        (self.source / "value.h").write_text("#define HEADER_VALUE 2\n")
        self.make(flags)
        self.assertEqual(self.output(), "7 2")
        self.assertNotEqual(self.stamp()[0], before[0])

    def test_same_compiler_path_with_changed_version(self):
        wrapper = self.root / "fixture-g++"
        # Use a relative compiler path so its parent directory may contain spaces.
        wrapper.write_text(
            '#!/bin/sh\nif [ "$1" = --version ]; then\n'
            + shlex.quote(COMPILER) + ' --version\nprintf "%s\\n" "$YO_FIXTURE_VERSION"\n'
            'else\nexec ' + shlex.quote(COMPILER) + ' "$@"\nfi\n'
        )
        wrapper.chmod(0o755)
        env = dict(os.environ, YO_FIXTURE_VERSION="one")
        self.make("COMPILER=../fixture-g++", env=env)
        before = self.stamp()
        self.make("COMPILER=../fixture-g++", env=env)
        self.assertEqual(self.stamp(), before)
        env["YO_FIXTURE_VERSION"] = "two"
        self.make("COMPILER=../fixture-g++", env=env)
        self.assertNotEqual(self.stamp()[0], before[0])

    def test_dry_run_and_clean(self):
        self.make()
        before = self.stamp()
        files = list(self.root.rglob(".*.cmd"))
        self.assertTrue(files)
        records = {p: p.read_bytes() for p in files}
        self.make("-n", "EXTRA_CPPFLAGS=-DFLAG_VALUE=9")
        self.assertEqual(self.stamp(), before)
        self.assertEqual({p: p.read_bytes() for p in files}, records)
        self.make(goal="clean")
        self.assertFalse((self.root / "engine").exists())
        self.assertFalse(list(self.root.rglob(".*.cmd")))

    def test_failed_compile_is_retried(self):
        self.make()
        (self.source / "value.h").write_text("#error fixture failure\n")
        self.make(success=False)
        self.assertFalse((self.root / "obj/.probe.o.cmd").exists())
        (self.source / "value.h").write_text("#define HEADER_VALUE 3\n")
        self.make()
        self.assertEqual(self.output(), "0 3")


class GeneratedArchitectureTest(unittest.TestCase):
    def test_unchanged_header_keeps_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = [sys.executable, str(ROOT / "tools/build/nnue_arch_gen.py"),
                    "SFNN_halfka2_1536_7_64_k3k3z", tmp]
            subprocess.run(args, check=True, stdout=subprocess.DEVNULL)
            header = next(Path(tmp).glob("*.h"))
            original = header.read_bytes()
            before = header.stat().st_mtime_ns
            subprocess.run(args, check=True, stdout=subprocess.DEVNULL)
            self.assertEqual(header.stat().st_mtime_ns, before)
            self.assertEqual(header.read_bytes(), original)
            header.write_bytes(b"// stale generated header\n")
            subprocess.run(args, check=True, stdout=subprocess.DEVNULL)
            self.assertEqual(header.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
