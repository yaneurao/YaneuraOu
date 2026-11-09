import argparse
import importlib.util
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("pgo", Path(__file__).resolve().parents[1] / "pgo.py")
pgo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pgo)


class PgoTest(unittest.TestCase):
    def test_native_python_restarts_msys_jobserver_at_parent_parallelism(self):
        with patch.object(pgo, "is_msys_python", return_value=True), \
             patch.dict(os.environ, {"MAKEFLAGS": "w -j8 --jobserver-auth=3,4 -- COMPILER=clang++"}):
            flags, env = pgo.make_environment()
        self.assertEqual(flags, ["-j8"])
        self.assertNotIn("--jobserver-auth", env["MAKEFLAGS"])
        self.assertIn("COMPILER=clang++", env["MAKEFLAGS"])

    def test_posix_jobserver_is_preserved(self):
        with patch.object(pgo, "is_msys_python", return_value=False), \
             patch.dict(os.environ, {"MAKEFLAGS": "w -j8 --jobserver-auth=3,4"}):
            flags, env = pgo.make_environment()
        self.assertEqual(flags, [])
        self.assertIn("--jobserver-auth=3,4", env["MAKEFLAGS"])

    def test_missing_profile_runtime_is_reported_before_engine_build(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args = argparse.Namespace(compiler="clang++", cppflags="", ldflags="")
            result = argparse.Namespace(returncode=1, stdout="ld.lld: error: missing libclang_rt.profile.a\n")
            with patch.object(pgo.subprocess, "run", return_value=result):
                with self.assertRaisesRegex(RuntimeError, "mingw-w64-x86_64-compiler-rt"):
                    pgo.check_profile_link(args, root)
            self.assertEqual((root / "profile-link-check.log").read_text(), result.stdout)

    def test_msys_compiler_wrapper_is_executed_by_bash(self):
        compiler = ["C:\\some dir\\wrapper", "--version"]
        with patch.object(pgo, "is_msys_python", return_value=True), \
             patch.object(pgo.shutil, "which", return_value="C:/msys64/usr/bin/bash.exe"):
            argv = pgo.shell_command(compiler)
        self.assertEqual(argv[:2], ["C:/msys64/usr/bin/bash.exe", "-c"])
        self.assertEqual(shlex.split(argv[2]), ["exec", "C:/some dir/wrapper", "--version"])

    def test_msys_eval_path_is_converted_for_native_python(self):
        with patch.object(pgo, "is_msys_python", return_value=True), \
             patch.object(pgo.subprocess, "check_output", return_value="/native/model dir\n") as convert:
            actual = pgo.native_path("/c/model dir")
        convert.assert_called_once_with(["cygpath", "-w", "/c/model dir"], text=True)
        self.assertEqual(actual, Path("/native/model dir").resolve())

    def test_make_uses_relative_paths_under_a_parent_with_spaces(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "parent with spaces"
            source = root / "source"
            source.mkdir(parents=True)
            before = Path.cwd()
            try:
                os.chdir(source)
                self.assertEqual(pgo.make_path(root / "obj/pgo/run/obj"), "../obj/pgo/run/obj")
                with self.assertRaisesRegex(RuntimeError, "PGO_DIR"):
                    pgo.make_path(root / "bad directory/run/obj")
            finally:
                os.chdir(before)

    def test_failed_publish_keeps_previous_executable(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            target = root / "engine"
            target.write_bytes(b"previous engine")
            with patch.object(pgo.shutil, "copy2", side_effect=OSError("copy failed")):
                with self.assertRaises(OSError):
                    pgo.publish(root / "new-engine", target)
            self.assertEqual(target.read_bytes(), b"previous engine")
            self.assertEqual(list(root.iterdir()), [target])

    @unittest.skipUnless(os.name == "posix", "Executable test fixture uses a shebang")
    def test_zero_exit_code_does_not_hide_eval_loading_error(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "profiles").mkdir()
            engine = root / "engine"
            engine.write_text(
                "#!/usr/bin/env python3\nimport sys\n"
                "for line in sys.stdin:\n"
                " if line.strip() == 'usi': print('usiok', flush=True)\n"
                " if line.strip() == 'isready':\n"
                "  print('info string Error! : missing eval', flush=True)\n"
                "  sys.exit(0)\n", encoding="utf-8")
            engine.chmod(0o755)
            args = argparse.Namespace(timeout=5, bench="64 1 2 default depth", eval_dir="")
            with self.assertRaisesRegex(RuntimeError, "評価関数"):
                pgo.train(args, {"binary": str(engine), "run_dir": str(root)})


if __name__ == "__main__":
    unittest.main()
