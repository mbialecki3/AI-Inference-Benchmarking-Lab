"""Run the regression suite with concise, human-readable terminal output."""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from typing import TextIO


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIRECTORY = PROJECT_ROOT / "src"
TEST_DIRECTORY = PROJECT_ROOT / "tests"


def _friendly_name(test: unittest.case.TestCase) -> str:
    """Turn a unittest method name into a compact terminal label."""

    module_name, class_name, method_name = test.id().rsplit(".", maxsplit=2)
    name = method_name.removeprefix("test_")
    label = name.replace("_", " ")
    for source, replacement in {
        "onnx": "ONNX",
        "pytorch": "PyTorch",
        "resnet50": "ResNet-50",
        "cuda": "CUDA",
        "cpu": "CPU",
    }.items():
        label = label.replace(source, replacement)
    area = {
        "OnnxExportTests": "ONNX export",
        "OnnxRunnerTests": "ONNX Runtime",
        "PyTorchReferenceTests": "PyTorch reference",
        "PyTorchRunnerTests": "PyTorch runner",
    }.get(class_name, module_name.replace("_", " "))
    return f"{area} — {label}"


class ReadableTestResult(unittest.TestResult):
    """Render one timed, readable result line for each completed test."""

    def __init__(self, stream: TextIO) -> None:
        super().__init__()
        self.stream = stream
        self._started_at: dict[str, float] = {}
        self._problems: list[tuple[str, unittest.case.TestCase, str]] = []

    def startTest(self, test: unittest.case.TestCase) -> None:
        super().startTest(test)
        self._started_at[test.id()] = time.perf_counter()

    def addSuccess(self, test: unittest.case.TestCase) -> None:
        super().addSuccess(test)
        self._write_result(test, "✅")

    def addSkip(self, test: unittest.case.TestCase, reason: str) -> None:
        super().addSkip(test, reason)
        self._write_result(test, f"⏭️  {reason}")

    def addFailure(
        self,
        test: unittest.case.TestCase,
        err: tuple[type[BaseException], BaseException, object],
    ) -> None:
        super().addFailure(test, err)
        self._record_problem("FAILURE", test, err)

    def addError(
        self,
        test: unittest.case.TestCase,
        err: tuple[type[BaseException], BaseException, object],
    ) -> None:
        super().addError(test, err)
        self._record_problem("ERROR", test, err)

    def _record_problem(
        self,
        category: str,
        test: unittest.case.TestCase,
        err: tuple[type[BaseException], BaseException, object],
    ) -> None:
        self._problems.append((category, test, self._exc_info_to_string(err, test)))
        self._write_result(test, "❌")

    def _write_result(self, test: unittest.case.TestCase, status: str) -> None:
        duration_seconds = time.perf_counter() - self._started_at.pop(test.id())
        label = _friendly_name(test)
        if len(label) > 64:
            label = f"{label[:61]}..."
        self.stream.write(f"  {label:.<64} {duration_seconds:>5.2f}s  {status}\n")

    def print_problems(self) -> None:
        """Print tracebacks only after the concise result summary."""

        for category, test, traceback in self._problems:
            self.stream.write(f"\n{category}: {test.id()}\n{traceback}\n")


def main() -> int:
    """Discover and run all regression tests from the project root."""

    sys.path.insert(0, str(SOURCE_DIRECTORY))
    sys.path.insert(0, str(TEST_DIRECTORY))
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(TEST_DIRECTORY),
        pattern="test_*.py",
        top_level_dir=str(TEST_DIRECTORY),
    )
    result = ReadableTestResult(sys.stdout)

    print("\nAI Inference Benchmark Lab — regression tests")
    print("─" * 76)
    started_at = time.perf_counter()
    suite.run(result)
    elapsed_seconds = time.perf_counter() - started_at
    print("─" * 76)

    if result.wasSuccessful():
        print(f"{result.testsRun} passed in {elapsed_seconds:.2f}s")
        return 0

    print(
        f"{len(result.failures)} failed, {len(result.errors)} errored "
        f"out of {result.testsRun} tests in {elapsed_seconds:.2f}s"
    )
    result.print_problems()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
