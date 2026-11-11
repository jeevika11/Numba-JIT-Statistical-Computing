
import numpy as np
from numba import jit, prange


@jit(nopython=True, parallel=True, cache=True)
def parallel_sigmoid(values):
    """Calculate sigmoid values in parallel."""

    result = np.empty_like(values)

    for i in prange(values.size):
        x = values[i]

        if x > 500.0:
            x = 500.0
        elif x < -500.0:
            x = -500.0

        result[i] = 1.0 / (1.0 + np.exp(-x))

    return result


@jit(nopython=True, parallel=True, cache=True)
def parallel_gaussian_pdf(values, mean, std):
    """Calculate Gaussian PDF values in parallel."""

    result = np.empty_like(values)

    coefficient = 1.0 / (std * np.sqrt(2.0 * np.pi))

    for i in prange(values.size):
        exponent = -0.5 * ((values[i] - mean) / std) ** 2

        result[i] = coefficient * np.exp(exponent)

    return result


@jit(nopython=True, parallel=True, cache=True)
def parallel_softmax(scores):
    """Calculate row-wise softmax probabilities in parallel."""

    rows, columns = scores.shape

    result = np.empty_like(scores)

    for i in prange(rows):

        maximum = scores[i, 0]

        for j in range(1, columns):
            if scores[i, j] > maximum:
                maximum = scores[i, j]

        total = 0.0

        for j in range(columns):
            result[i, j] = np.exp(scores[i, j] - maximum)
            total += result[i, j]

        for j in range(columns):
            result[i, j] /= total

    return result


@jit(nopython=True, parallel=True, cache=True)
def parallel_log_likelihood(y_true, probabilities):
    """Calculate binary log-likelihood using parallel reduction."""

    epsilon = 1e-15
    total = 0.0

    for i in prange(y_true.size):

        probability = probabilities[i]

        if probability < epsilon:
            probability = epsilon
        elif probability > 1.0 - epsilon:
            probability = 1.0 - epsilon

        total += (
            y_true[i] * np.log(probability)
            + (1.0 - y_true[i]) * np.log(1.0 - probability)
        )

    return total


@jit(nopython=True, parallel=True, cache=True)
def parallel_logistic_gradient(X, y, weights, bias):
    """Calculate logistic regression gradients in parallel."""

    samples, features = X.shape

    errors = np.empty(samples, dtype=np.float64)

    for i in prange(samples):

        score = bias

        for j in range(features):
            score += X[i, j] * weights[j]

        probability = 1.0 / (1.0 + np.exp(-score))

        errors[i] = probability - y[i]

    weight_gradient = np.zeros(features, dtype=np.float64)

    for j in prange(features):

        total = 0.0

        for i in range(samples):
            total += X[i, j] * errors[i]

        weight_gradient[j] = total / samples

    bias_gradient = np.sum(errors) / samples

    return weight_gradient, bias_gradient


if __name__ == "__main__":

    from scipy.special import expit, softmax
    from scipy.stats import norm

    print("=" * 60)
    print("NUMBA JIT PARALLEL KERNEL TEST")
    print("=" * 60)

    # ------------------------------------------
    # 1. Sigmoid verification
    # ------------------------------------------
    values = np.linspace(-5, 5, 1000).astype(np.float64)

    jit_sigmoid = parallel_sigmoid(values)
    scipy_sigmoid = expit(values)

    sigmoid_check = np.allclose(
        jit_sigmoid, scipy_sigmoid, atol=1e-10
    )

    print("\nSigmoid verification:",
          "PASS" if sigmoid_check else "FAIL")

    # ------------------------------------------
    # 2. Gaussian PDF verification
    # ------------------------------------------
    jit_gaussian = parallel_gaussian_pdf(
        values, 0.0, 1.0
    )

    scipy_gaussian = norm.pdf(values, loc=0.0, scale=1.0)

    gaussian_check = np.allclose(
        jit_gaussian, scipy_gaussian, atol=1e-10
    )

    print("Gaussian PDF verification:",
          "PASS" if gaussian_check else "FAIL")

    # ------------------------------------------
    # 3. Softmax verification
    # ------------------------------------------
    scores = np.array([
        [2.0, 1.0, 0.1],
        [0.5, 2.5, 1.0],
        [1.0, 0.5, 3.0]
    ], dtype=np.float64)

    jit_softmax = parallel_softmax(scores)
    scipy_softmax = softmax(scores, axis=1)

    softmax_check = (
        np.allclose(jit_softmax, scipy_softmax, atol=1e-10)
        and np.allclose(np.sum(jit_softmax, axis=1), 1.0)
    )

    print("Softmax verification:",
          "PASS" if softmax_check else "FAIL")

    # ------------------------------------------
    # 4. Log-likelihood verification
    # ------------------------------------------
    y = np.array([1, 0, 1, 1, 0], dtype=np.float64)

    probabilities = np.array([
        0.9, 0.2, 0.8, 0.7, 0.1
    ], dtype=np.float64)

    jit_likelihood = parallel_log_likelihood(
        y, probabilities
    )

    numpy_likelihood = np.sum(
        y * np.log(probabilities)
        + (1 - y) * np.log(1 - probabilities)
    )

    likelihood_check = np.isclose(
        jit_likelihood, numpy_likelihood
    )

    print("Log-likelihood verification:",
          "PASS" if likelihood_check else "FAIL")

    # ------------------------------------------
    # 5. Logistic gradient verification
    # ------------------------------------------
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 4.0],
        [4.0, 3.0]
    ], dtype=np.float64)

    labels = np.array([0, 0, 1, 1], dtype=np.float64)

    weights = np.array([0.2, -0.1], dtype=np.float64)
    bias = 0.05

    jit_w_grad, jit_b_grad = parallel_logistic_gradient(
        X, labels, weights, bias
    )

    numpy_probabilities = expit(X @ weights + bias)
    errors = numpy_probabilities - labels

    numpy_w_grad = (X.T @ errors) / len(labels)
    numpy_b_grad = np.mean(errors)

    gradient_check = (
        np.allclose(jit_w_grad, numpy_w_grad, atol=1e-10)
        and np.isclose(jit_b_grad, numpy_b_grad, atol=1e-10)
    )

    print("Logistic gradient verification:",
          "PASS" if gradient_check else "FAIL")

    # ------------------------------------------
    # Final result
    # ------------------------------------------
    checks = [
        sigmoid_check,
        gaussian_check,
        softmax_check,
        likelihood_check,
        gradient_check
    ]

    passed = sum(checks)

    print("\n" + "=" * 60)
    print(f"Tests passed: {passed}/{len(checks)}")
    print(f"Tests failed: {len(checks) - passed}/{len(checks)}")

    print(
        "\nNUMBA JIT STATUS:",
        "ALL TESTS PASSED" if passed == len(checks)
        else "CHECK FAILED TESTS"
    )

    print("=" * 60)
