
import numpy as np


def sigmoid(x):
    """Calculate the numerically stable sigmoid function."""

    x = np.asarray(x, dtype=np.float64)

    if not np.all(np.isfinite(x)):
        raise ValueError("Input must contain finite values.")

    # Clipping prevents numerical overflow.
    x = np.clip(x, -500, 500)

    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    """Convert scores into a probability distribution."""

    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 1 or x.size == 0:
        raise ValueError("Input must be a non-empty 1D array.")

    if not np.all(np.isfinite(x)):
        raise ValueError("Input must contain finite values.")

    # Subtract maximum for numerical stability.
    shifted = x - np.max(x)

    exp_values = np.exp(shifted)

    return exp_values / np.sum(exp_values)


def binary_log_likelihood(y_true, probabilities):
    """Calculate binary classification log-likelihood."""

    y_true = np.asarray(y_true, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if y_true.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if y_true.size == 0 or y_true.size != probabilities.size:
        raise ValueError("Inputs must have equal non-zero lengths.")

    if not np.all(np.isfinite(y_true)):
        raise ValueError("Labels must contain finite values.")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities must contain finite values.")

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("Labels must contain only 0 and 1.")

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probabilities must be between 0 and 1.")

    epsilon = 1e-15

    probabilities = np.clip(
        probabilities, epsilon, 1.0 - epsilon
    )

    return np.sum(
        y_true * np.log(probabilities)
        + (1 - y_true) * np.log(1 - probabilities)
    )


def gaussian_pdf(x, mean=0.0, std=1.0):
    """Calculate the Gaussian probability density function."""

    x = np.asarray(x, dtype=np.float64)

    if std <= 0:
        raise ValueError("Standard deviation must be positive.")

    if not np.isfinite(mean) or not np.isfinite(std):
        raise ValueError("Mean and standard deviation must be finite.")

    if not np.all(np.isfinite(x)):
        raise ValueError("Input must contain finite values.")

    coefficient = 1.0 / (std * np.sqrt(2.0 * np.pi))

    exponent = -0.5 * ((x - mean) / std) ** 2

    return coefficient * np.exp(exponent)


if __name__ == "__main__":

    print("=" * 55)
    print("STATISTICAL FUNCTIONS TEST")
    print("=" * 55)

    # 1. Sigmoid test
    values = np.array([-2, -1, 0, 1, 2], dtype=float)

    sigmoid_result = sigmoid(values)

    print("\nSigmoid output:")
    print(np.round(sigmoid_result, 4))

    sigmoid_check = (
        np.all(sigmoid_result >= 0)
        and np.all(sigmoid_result <= 1)
        and np.isclose(sigmoid(0), 0.5)
    )

    print("Sigmoid verification:",
          "PASS" if sigmoid_check else "FAIL")

    # 2. Softmax test
    scores = np.array([2.0, 1.0, 0.1])

    softmax_result = softmax(scores)

    print("\nSoftmax output:")
    print(np.round(softmax_result, 4))

    softmax_check = (
        np.all(softmax_result >= 0)
        and np.isclose(np.sum(softmax_result), 1.0)
    )

    print("Softmax verification:",
          "PASS" if softmax_check else "FAIL")

    # 3. Binary log-likelihood test
    actual = np.array([1, 0, 1, 1, 0])

    probabilities = np.array([
        0.9, 0.2, 0.8, 0.7, 0.1
    ])

    likelihood = binary_log_likelihood(
        actual, probabilities
    )

    print("\nBinary log-likelihood:")
    print(round(likelihood, 4))

    print("Log-likelihood verification:",
          "PASS" if np.isfinite(likelihood) else "FAIL")

    # 4. Gaussian PDF test
    x = np.array([-2, -1, 0, 1, 2], dtype=float)

    density = gaussian_pdf(x)

    print("\nGaussian PDF output:")
    print(np.round(density, 4))

    gaussian_check = (
        np.all(density >= 0)
        and np.isclose(density[2], 1 / np.sqrt(2 * np.pi))
    )

    print("Gaussian PDF verification:",
          "PASS" if gaussian_check else "FAIL")

    print("\n" + "=" * 55)

    all_passed = all([
        sigmoid_check,
        softmax_check,
        np.isfinite(likelihood),
        gaussian_check
    ])

    print(
        "STATISTICAL FUNCTIONS STATUS:",
        "ALL TESTS PASSED" if all_passed else "CHECK FAILED TESTS"
    )

    print("=" * 55)
