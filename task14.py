
import time
import numpy as np
from scipy.special import expit, softmax as scipy_softmax
from scipy.stats import norm

from statistical_functions import (
    sigmoid,
    softmax,
    gaussian_pdf,
    binary_log_likelihood
)

from probability_models import (
    LogisticRegressionModel,
    multiclass_probabilities,
    binary_cross_entropy
)

from jit_kernels import (
    parallel_sigmoid,
    parallel_gaussian_pdf,
    parallel_softmax,
    parallel_log_likelihood,
    parallel_logistic_gradient
)

from benchmark import python_sigmoid, numpy_sigmoid


def run_test(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"{name:<40}: {status}")
    return bool(condition)


def main():

    print("=" * 65)
    print("TASK 14: CUSTOM JIT COMPILATION OF STATISTICAL FUNCTIONS")
    print("NUMBA PARALLEL COMPUTATION AND PERFORMANCE ANALYSIS")
    print("=" * 65)

    # --------------------------------------------------
    # 1. Generate sample data
    # --------------------------------------------------
    rng = np.random.default_rng(42)

    X = rng.normal(size=(500, 3)).astype(np.float64)

    true_weights = np.array([2.0, -1.5, 1.0])

    scores = X @ true_weights

    probabilities = expit(scores)

    y = (probabilities >= 0.5).astype(np.float64)

    print("\nDATASET INFORMATION")
    print("-" * 65)
    print("Number of samples:", len(X))
    print("Number of features:", X.shape[1])
    print("Positive samples:", int(np.sum(y)))
    print("Negative samples:", int(len(y) - np.sum(y)))

    # --------------------------------------------------
    # 2. Train logistic regression model
    # --------------------------------------------------
    print("\nLOGISTIC REGRESSION TRAINING")
    print("-" * 65)

    model = LogisticRegressionModel(
        learning_rate=0.1,
        epochs=500
    )

    model.fit(X, y)

    predictions = model.predict(X)
    predicted_probabilities = model.predict_proba(X)

    accuracy = np.mean(predictions == y)

    print("Final training loss:",
          round(model.loss_history[-1], 6))

    print("Training accuracy:",
          f"{accuracy:.2%}")

    # --------------------------------------------------
    # 3. Multiclass probability calculation
    # --------------------------------------------------
    multiclass_scores = np.array([
        [2.0, 1.0, 0.1],
        [0.5, 2.5, 1.0],
        [1.0, 0.5, 3.0]
    ], dtype=np.float64)

    multi_probabilities = multiclass_probabilities(
        multiclass_scores
    )

    # --------------------------------------------------
    # 4. Prepare JIT test data
    # --------------------------------------------------
    values = np.linspace(
        -5.0, 5.0, 10000
    ).astype(np.float64)

    jit_sigmoid = parallel_sigmoid(values)
    jit_gaussian = parallel_gaussian_pdf(
        values, 0.0, 1.0
    )

    jit_softmax = parallel_softmax(multiclass_scores)

    test_labels = np.array(
        [1, 0, 1, 1, 0], dtype=np.float64
    )

    test_probabilities = np.array(
        [0.9, 0.2, 0.8, 0.7, 0.1],
        dtype=np.float64
    )

    jit_likelihood = parallel_log_likelihood(
        test_labels, test_probabilities
    )

    # --------------------------------------------------
    # 5. Logistic gradient verification
    # --------------------------------------------------
    gradient_weights = np.array(
        [0.2, -0.1, 0.3], dtype=np.float64
    )

    gradient_bias = 0.05

    jit_weight_gradient, jit_bias_gradient = (
        parallel_logistic_gradient(
            X,
            y,
            gradient_weights,
            gradient_bias
        )
    )

    reference_probabilities = expit(
        X @ gradient_weights + gradient_bias
    )

    errors = reference_probabilities - y

    reference_weight_gradient = (
        X.T @ errors
    ) / len(y)

    reference_bias_gradient = np.mean(errors)

    # --------------------------------------------------
    # 6. Performance benchmark
    # --------------------------------------------------
    print("\nPERFORMANCE BENCHMARK")
    print("-" * 65)

    benchmark_values = rng.normal(
        size=100000
    ).astype(np.float64)

    start = time.perf_counter()
    python_sigmoid(benchmark_values)
    python_time = time.perf_counter() - start

    start = time.perf_counter()
    numpy_sigmoid(benchmark_values)
    numpy_time = time.perf_counter() - start

    # First call includes compilation if this signature
    # has not already been compiled in this process.
    start = time.perf_counter()
    parallel_sigmoid(benchmark_values)
    first_numba_time = time.perf_counter() - start

    # Subsequent execution measures compiled performance.
    start = time.perf_counter()
    parallel_sigmoid(benchmark_values)
    compiled_numba_time = time.perf_counter() - start

    print(f"Standard Python: {python_time:.6f} seconds")
    print(f"NumPy:           {numpy_time:.6f} seconds")
    print(f"Numba first call:{first_numba_time:.6f} seconds")
    print(f"Numba compiled:  {compiled_numba_time:.6f} seconds")

    # --------------------------------------------------
    # 7. Final verification tests
    # --------------------------------------------------
    print("\nFINAL VERIFICATION TESTS")
    print("-" * 65)

    tests = []

    tests.append(run_test(
        "Dataset dimensions",
        X.shape == (500, 3)
    ))

    tests.append(run_test(
        "Logistic regression training",
        len(model.loss_history) == 500
        and model.loss_history[-1] < model.loss_history[0]
    ))

    tests.append(run_test(
        "Binary classification predictions",
        predictions.shape == y.shape
        and np.all(np.isin(predictions, [0, 1]))
    ))

    tests.append(run_test(
        "Probability range",
        np.all(predicted_probabilities >= 0)
        and np.all(predicted_probabilities <= 1)
    ))

    tests.append(run_test(
        "Multiclass probability distribution",
        np.allclose(
            np.sum(multi_probabilities, axis=1), 1.0
        )
    ))

    tests.append(run_test(
        "Numba sigmoid vs SciPy",
        np.allclose(jit_sigmoid, expit(values), atol=1e-10)
    ))

    tests.append(run_test(
        "Numba Gaussian PDF vs SciPy",
        np.allclose(
            jit_gaussian, norm.pdf(values), atol=1e-10
        )
    ))

    tests.append(run_test(
        "Numba softmax vs SciPy",
        np.allclose(
            jit_softmax,
            scipy_softmax(multiclass_scores, axis=1),
            atol=1e-10
        )
    ))

    tests.append(run_test(
        "Numba log-likelihood",
        np.isclose(
            jit_likelihood,
            binary_log_likelihood(
                test_labels, test_probabilities
            )
        )
    ))

    tests.append(run_test(
        "Numba logistic gradient",
        np.allclose(
            jit_weight_gradient,
            reference_weight_gradient,
            atol=1e-10
        )
        and np.isclose(
            jit_bias_gradient,
            reference_bias_gradient,
            atol=1e-10
        )
    ))

    # --------------------------------------------------
    # 8. Final summary
    # --------------------------------------------------
    passed = sum(tests)
    total = len(tests)

    print("\n" + "=" * 65)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 65)

    print(f"Tests passed: {passed}/{total}")
    print(f"Tests failed: {total - passed}/{total}")

    if passed == total:
        print("\nTASK 14 STATUS: ALL TESTS PASSED")
    else:
        print("\nTASK 14 STATUS: CHECK FAILED TESTS")

    print("=" * 65)


if __name__ == "__main__":
    main()
