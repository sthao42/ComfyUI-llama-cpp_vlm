#!/usr/bin/env python3
"""Run all unit and simulation tests for ComfyUI-llama-cpp_vlm."""

import os
import subprocess
import sys

# Ensure root directory is resolved correctly whether run from root or tests/
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TESTS_DIR)

test_files = [
    os.path.join(TESTS_DIR, "test_nodes.py"),
    os.path.join(TESTS_DIR, "test_simulation.py"),
]

all_passed = True
for test_path in test_files:
    rel_path = os.path.relpath(test_path, ROOT_DIR)
    print(f"\n{'='*80}")
    print(f"Running: {rel_path}")
    print(f"{'='*80}\n")

    result = subprocess.run(
        [sys.executable, test_path],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    print(f"Exit code: {result.returncode}")
    if result.returncode != 0:
        all_passed = False

if all_passed:
    print(f"\n{'='*80}")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print(f"{'='*80}\n")
    sys.exit(0)
else:
    print(f"\n{'='*80}")
    print("TEST SUITE FAILED!")
    print(f"{'='*80}\n")
    sys.exit(1)
