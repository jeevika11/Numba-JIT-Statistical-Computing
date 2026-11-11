
import time
import math
import statistics
import numpy as np
from scipy.special import expit
from numba import get_num_threads

from jit_kernels import parallel_sigmoid


def python_sigmoid(values):
    """Calculate sigmoid using a standard Python loop."""

    result = []

    for value in values:
        value = max(-500.0, min(500.0, float(value)))
        result.append(1.0 / (1.0 + math.exp(-value)))

    return np.asarray(result, dtype=np.float64)


def numpy_sigmoid(values):
    """Calculate sigmoid using NumPy vectorization."""

    values = np.clip(values, -500.0, 500.0)

    return 1.0 / (1.0 + np.exp(-values))


def measure_time(function, values, repeats=5):
    """Measure execution time and return the median."""

    timings = []

    for _ in range(repeats):
        start = time.perf_counter()

        function(values)

        elapsed = time.perf_counter() - start
        timings.append(elapsed)

    return statistics.median(timings)


def main():

    print("=" * 65)
    print("NUMBA JIT PERFORMANCE BENCHMARK")
    print("=" * 65)

    # Generate a large numerical dataset.
    rng = np.random.default_rng(42)

    values = rng.normal(
        loc=0.0,
        scale=2.0,
        size=300_000
    ).astype(np.float64)

    print("\nDATASET INFORMATION")
    print("-" * 65)
    print("Number of values:", values.size)
    print("Data type:", values.dtype)
    print("Numba parallel threads:", get_num_threads())

    # --------------------------------------------------
    # 1. Standard Python benchmark
    # --------------------------------------------------
    print("\nRunning standard Python implementation...")

    python_time = measure_time(
        python_sigmoid, values, repeats=3
    )

    python_result = python_sigmoid(values)

    # --------------------------------------------------
    # 2. NumPy benchmark
    # --------------------------------------------------
    print("Running NumPy implementation...")

    numpy_time = measure_time(
        numpy_sigmoid, values
    )

    numpy_result = numpy_sigmoid(values)

    # --------------------------------------------------
    # 3. Numba first-call compilation benchmark
    # --------------------------------------------------
    print("Running first Numba JIT call...")

    start = time.perf_counter()

    first_result = parallel_sigmoid(values)

    first_call_time = time.perf_counter() - start

    # --------------------------------------------------
    # 4. Numba compiled execution benchmark
    # --------------------------------------------------
    print("Running compiled Numba implementation...")

    # Additional warm-up before measuring.
    parallel_sigmoid(values)

    numba_time = measure_time(
        parallel_sigmoid, values
    )

    numba_result = parallel_sigmoid(values)

    # --------------------------------------------------
    # 5. Numerical verification
    # --------------------------------------------------
    reference = expit(values)

    python_check = np.allclose(
        python_result, reference, atol=1e-12
    )

    numpy_check = np.allclose(
        numpy_result, reference, atol=1e-12
    )

    numba_check = np.allclose(
        numba_result, reference, atol=1e-12
    )

    # --------------------------------------------------
    # 6. Display benchmark results
    # --------------------------------------------------
    print("\nBENCHMARK RESULTS")
    print("-" * 65)

    print(f"Standard Python : {python_time:.6f} seconds")
    print(f"NumPy           : {numpy_time:.6f} seconds")
    print(f"Numba first call: {first_call_time:.6f} seconds")
    print(f"Numba compiled  : {numba_time:.6f} seconds")

    print("\nPERFORMANCE COMPARISON")
    print("-" * 65)

    print(
        f"NumPy speedup over Python: "
        f"{python_time / numpy_time:.2f}x"
    )

    print(
        f"Numba speedup over Python: "
        f"{python_time / numba_time:.2f}x"
    )

    print("\nNUMERICAL VERIFICATION")
    print("-" * 65)

    print(
        "Python vs SciPy:",
        "PASS" if python_check else "FAIL"
    )

    print(
        "NumPy vs SciPy:",
        "PASS" if numpy_check else "FAIL"
    )

    print(
        "Numba vs SciPy:",
        "PASS" if numba_check else "FAIL"
    )

    print(
        "Python vs NumPy:",
        "PASS" if np.allclose(
            python_result, numpy_result, atol=1e-12
        ) else "FAIL"
    )

    print(
        "NumPy vs Numba:",
        "PASS" if np.allclose(
            numpy_result, numba_result, atol=1e-12
        ) else "FAIL"
    )

    # --------------------------------------------------
    # 7. Final summary
    # --------------------------------------------------
    checks = [
        python_check,
        numpy_check,
        numba_check,
        np.allclose(python_result, numpy_result, atol=1e-12),
        np.allclose(numpy_result, numba_result, atol=1e-12)
    ]

    passed = sum(checks)

    print("\n" + "=" * 65)
    print(f"Tests passed: {passed}/{len(checks)}")
    print(f"Tests failed: {len(checks) - passed}/{len(checks)}")

    print(
        "\nBENCHMARK STATUS:",
        "ALL TESTS PASSED" if passed == len(checks)
        else "CHECK FAILED TESTS"
    )

    print("=" * 65)


if __name__ == "__main__":
    main()
