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


class WindowsMakeCommandTest(MakeCommandTest):
    def setUp(self):
        super().setUp()
        # Exercise the Windows recipes with the available native compiler.
        self.base += ["OS=Windows_NT", "LDFLAGS=$(EXTRA_LDFLAGS)"]

    def test_long_link_commands_stay_within_shell_limit(self):
        shell = self.root / "limited-sh"
        shell.write_text(
            "#!" + sys.executable + "\n"
            "import os, sys\n"
            "if '-c' in sys.argv:\n"
            "    command = sys.argv[sys.argv.index('-c') + 1]\n"
            "    if len(command) > 8191:\n"
            "        sys.exit('fixture: shell command exceeds 8191 characters')\n"
            "os.execv('/bin/sh', ['sh', *sys.argv[1:]])\n"
        )
        shell.chmod(0o755)
        sources = ["probe.cpp"]
        (self.source / "padding").mkdir()
        for i in range(45):
            name = f"padding/translation_unit_{i:03}.cpp"
            (self.source / name).write_text(f"int padding_{i}() {{ return {i}; }}\n")
            sources.append(name)
        prefix = "../Users/runneradmin/AppData/Local/Temp/tmpbtih1z.tmp"
        settings = (
            "SHELL=../limited-sh", "SOURCES=" + " ".join(sources),
            "OBJDIR=" + prefix + "/obj", "TARGET=" + prefix + "/engine",
        )
        self.make(*settings)
        output = self.root / prefix.removeprefix("../") / "engine"
        before = output.stat().st_mtime_ns
        self.make(*settings)
        self.assertEqual(output.stat().st_mtime_ns, before)

    def test_quoted_compiler_identity_round_trip(self):
        wrapper = self.root / "quoted-g++"
        wrapper.write_text(
            '#!/bin/sh\nif [ "$1" = --version ]; then\n'
            + shlex.quote(COMPILER) + ' --version\nprintf "%s\\n" "$YO_FIXTURE_VERSION"\n'
            'else\nexec ' + shlex.quote(COMPILER) + ' "$@"\nfi\n'
        )
        wrapper.chmod(0o755)
        env = dict(os.environ, YO_FIXTURE_VERSION="compiler's #tag $value \\ path")
        self.make("COMPILER=../quoted-g++", env=env)
        before = self.stamp()
        self.make("COMPILER=../quoted-g++", env=env)
        self.assertEqual(self.stamp(), before)

    def test_drive_letter_record_key_round_trip(self):
        extra = self.source / "drive-key.mk"
        extra.write_text(
            "drive_target := C:/Users/runneradmin/AppData/Local/Temp/engine.exe\n"
            "$(eval $(call cmd_key,$(drive_target)) := tracked)\n"
            "$(if $(filter tracked,$($(call cmd_key,$(drive_target)))),,"
            "$(error drive-letter command record did not round-trip))\n"
            "key-check: ; @true\n"
        )
        self.make("-f", "Makefile", "-f", extra.name, goal="key-check")

    def test_inspection_does_not_create_records_or_directories(self):
        self.make("-n")
        result = self.make("-q", goal="all", success=False)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertFalse((self.root / "obj").exists())
        self.assertFalse((self.root / "engine").exists())
        self.assertFalse(list(self.root.rglob(".*.cmd*")))

    def test_failed_link_does_not_publish_command_record(self):
        self.make()
        before = self.stamp()
        self.make("EXTRA_LDFLAGS=-Wl,--yaneuraou-invalid-link-option", success=False)
        self.assertFalse((self.root / ".engine.cmd").exists())
        self.make()
        self.assertEqual(self.stamp()[0], before[0])
        self.assertNotEqual(self.stamp()[1], before[1])
        self.assertEqual(self.output(), "0 1")


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
