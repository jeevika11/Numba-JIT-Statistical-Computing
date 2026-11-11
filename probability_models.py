
import numpy as np
from statistical_functions import sigmoid, softmax, binary_log_likelihood


class LogisticRegressionModel:
    """Custom binary logistic regression model."""

    def __init__(self, learning_rate=0.05, epochs=500):
        if learning_rate <= 0 or epochs < 1:
            raise ValueError("Invalid training configuration.")

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be 2D and y must be 1D.")

        if X.shape[0] != y.size or X.shape[0] == 0:
            raise ValueError("X and y must have equal non-zero samples.")

        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("Training data must be finite.")

        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("Binary labels must be 0 or 1.")

        self.weights = np.zeros(X.shape[1], dtype=np.float64)
        self.bias = 0.0
        self.loss_history = []

        epsilon = 1e-15
        sample_count = len(y)

        for _ in range(self.epochs):
            scores = X @ self.weights + self.bias
            probabilities = sigmoid(scores)

            error = probabilities - y

            weight_gradient = (X.T @ error) / sample_count
            bias_gradient = np.mean(error)

            self.weights -= self.learning_rate * weight_gradient
            self.bias -= self.learning_rate * bias_gradient

            updated_probabilities = sigmoid(
                X @ self.weights + self.bias
            )

            clipped = np.clip(
                updated_probabilities,
                epsilon,
                1.0 - epsilon
            )

            loss = -np.mean(
                y * np.log(clipped)
                + (1 - y) * np.log(1 - clipped)
            )

            self.loss_history.append(float(loss))

        return self

    def predict_proba(self, X):
        if self.weights is None:
            raise RuntimeError("Train the model before prediction.")

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2 or X.shape[1] != self.weights.size:
            raise ValueError("Invalid prediction input shape.")

        return sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        if not 0 < threshold < 1:
            raise ValueError("Threshold must be between 0 and 1.")

        return (self.predict_proba(X) >= threshold).astype(int)


def multiclass_probabilities(scores):
    """Calculate multiclass probabilities using softmax."""

    scores = np.asarray(scores, dtype=np.float64)

    if scores.ndim != 2 or scores.shape[1] == 0:
        raise ValueError("Scores must be a 2D array.")

    return np.vstack([
        softmax(row) for row in scores
    ])


def binary_cross_entropy(y_true, probabilities):
    """Calculate average binary cross-entropy."""

    y_true = np.asarray(y_true, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if y_true.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if y_true.size == 0 or y_true.size != probabilities.size:
        raise ValueError("Inputs must have equal non-zero lengths.")

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("Labels must contain only 0 and 1.")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities must be finite.")

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probabilities must be between 0 and 1.")

    epsilon = 1e-15
    probabilities = np.clip(
        probabilities, epsilon, 1.0 - epsilon
    )

    return float(-binary_log_likelihood(y_true, probabilities)
                 / y_true.size)


if __name__ == "__main__":

    print("=" * 55)
    print("PROBABILITY MODELS TEST")
    print("=" * 55)

    # Binary classification dataset.
    X = np.array([
        [1.0, 2.0],
        [1.5, 2.0],
        [2.0, 1.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [6.0, 5.0]
    ])

    y = np.array([0, 0, 0, 1, 1, 1])

    model = LogisticRegressionModel(
        learning_rate=0.1,
        epochs=500
    )

    model.fit(X, y)

    probabilities = model.predict_proba(X)
    predictions = model.predict(X)

    print("\nBinary probabilities:")
    print(np.round(probabilities, 4))

    print("\nPredicted labels:")
    print(predictions)

    binary_check = np.array_equal(predictions, y)

    print("\nBinary classification verification:",
          "PASS" if binary_check else "FAIL")

    # Multiclass probability test.
    scores = np.array([
        [2.0, 1.0, 0.1],
        [0.5, 2.5, 1.0],
        [1.0, 0.5, 3.0]
    ])

    multi_probabilities = multiclass_probabilities(scores)

    print("\nMulticlass probabilities:")
    print(np.round(multi_probabilities, 4))

    multiclass_check = (
        np.all(multi_probabilities >= 0)
        and np.allclose(
            np.sum(multi_probabilities, axis=1), 1.0
        )
    )

    print("\nMulticlass verification:",
          "PASS" if multiclass_check else "FAIL")

    # Binary cross-entropy test.
    loss = binary_cross_entropy(y, probabilities)

    print("\nBinary cross-entropy:", round(loss, 4))

    loss_check = np.isfinite(loss) and loss >= 0

    print("Cross-entropy verification:",
          "PASS" if loss_check else "FAIL")

    # Training loss verification.
    training_check = (
        len(model.loss_history) == model.epochs
        and model.loss_history[-1] < model.loss_history[0]
    )

    print("Training loss verification:",
          "PASS" if training_check else "FAIL")

    print("\n" + "=" * 55)

    all_passed = all([
        binary_check,
        multiclass_check,
        loss_check,
        training_check
    ])

    print(
        "PROBABILITY MODELS STATUS:",
        "ALL TESTS PASSED" if all_passed else "CHECK FAILED TESTS"
    )

    print("=" * 55)
