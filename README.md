# Task 14: Custom Just-In-Time (JIT) Compilation of Advanced Statistical Functions using Numba

## 1. Overview

This project implements an advanced statistical modeling pipeline using Python, NumPy, SciPy and Numba.

It uses Just-In-Time (JIT) compilation to accelerate nonlinear statistical functions and parallel numerical computations. Numba's `@jit(nopython=True, parallel=True)` decorator and `prange` are used to compile and execute custom numerical kernels.

## 2. Objectives

- Implement nonlinear statistical functions.
- Develop custom sigmoid and softmax functions.
- Calculate Gaussian probability density.
- Implement binary log-likelihood.
- Build a custom logistic regression model.
- Use Numba JIT compilation for numerical acceleration.
- Perform parallel array processing using `prange`.
- Benchmark Python, NumPy and Numba implementations.
- Verify numerical accuracy using SciPy.

## 3. Technologies Used

- Python 3.10+
- NumPy
- Numba
- SciPy

## 4. Project Structure

```text
Task 14/
│
├── statistical_functions.py
├── probability_models.py
├── jit_kernels.py
├── benchmark.py
├── task14.py
├── requirements.txt
└── README.md
```

## 5. Implementation Details

### Statistical Functions

Implements:
- Sigmoid function
- Softmax function
- Binary log-likelihood
- Gaussian probability density function

### Probability Models

Implements custom binary logistic regression, multiclass probability calculation, prediction and binary cross-entropy.

### Numba JIT Kernels

Uses Numba decorators and parallel processing for:
- Parallel sigmoid
- Parallel Gaussian PDF
- Parallel softmax
- Parallel log-likelihood
- Parallel logistic regression gradient

### Performance Benchmarking

Compares:
- Standard Python execution
- NumPy vectorized execution
- Numba first-call execution
- Numba compiled execution

Compilation overhead and subsequent execution time are measured separately.

## 6. JIT Compilation

The implementation uses:

```python
from numba import jit, prange

@jit(nopython=True, parallel=True, cache=True)
def parallel_sigmoid(values):
    result = np.empty_like(values)

    for i in prange(values.size):
        result[i] = 1.0 / (1.0 + np.exp(-values[i]))

    return result
```

`nopython=True` requests native compilation without Python object-mode execution, while `parallel=True` enables supported parallel transformations.

## 7. Installation

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## 8. Run the Project

Run individual modules:

```bash
python statistical_functions.py
python probability_models.py
python jit_kernels.py
python benchmark.py
```

Run the complete project:

```bash
python task14.py
```

## 9. Verification Results

The complete implementation was tested using numerical datasets and SciPy reference calculations.

| Verification | Result |
|---|---|
| Dataset dimensions | PASS |
| Logistic regression training | PASS |
| Binary classification predictions | PASS |
| Probability range | PASS |
| Multiclass probability distribution | PASS |
| Numba sigmoid vs SciPy | PASS |
| Numba Gaussian PDF vs SciPy | PASS |
| Numba softmax vs SciPy | PASS |
| Numba log-likelihood | PASS |
| Numba logistic gradient | PASS |

**Final Result: 10/10 tests passed.**

## 10. Conclusion

The project demonstrates custom JIT compilation of advanced statistical functions using Numba. It incorporates nonlinear probability functions, parallel numerical kernels, logistic regression and performance benchmarking. Numerical outputs are independently verified against SciPy reference implementations.
