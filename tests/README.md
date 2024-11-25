# Tests Directory

The `tests/` directory contains a comprehensive suite of tests to ensure the reliability, correctness, and performance of the **Dynamic Neural Network Refinement** project. The tests are organized into unit tests, integration tests, and end-to-end tests, covering all critical components and functionalities of the project.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Types of Tests](#types-of-tests)
  - [Unit Tests](#unit-tests)
  - [Integration Tests](#integration-tests)
  - [End-to-End Tests](#end-to-end-tests)
- [Running the Tests](#running-the-tests)
- [Test Coverage](#test-coverage)
- [Best Practices](#best-practices)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

Testing is a crucial aspect of the project, ensuring that each component functions as intended and that integrations between components do not introduce regressions. The tests help in maintaining high code quality, facilitating smooth development and deployment processes.

## Directory Structure

```
tests/
├── test_analyzer.py
├── test_hybrid_thresholds.py
├── test_integration.py
├── test_model.py
├── test_nas.py
├── test_e2e.py
└── test_visualization.py
```

## Types of Tests

### Unit Tests

- **Purpose:**  
  Verify the functionality of individual units of code (e.g., classes, functions) in isolation.

- **Location:**  
  - `test_analyzer.py`
  - `test_hybrid_thresholds.py`
  - `test_model.py`
  - `test_nas.py`

- **Example:**  
  Testing whether the `Analyzer` class correctly computes variance, entropy, and sparsity metrics.

### Integration Tests

- **Purpose:**  
  Assess the interactions between different modules or components to ensure they work together seamlessly.

- **Location:**  
  - `test_integration.py`

- **Example:**  
  Testing the training and evaluation pipeline to verify that the model training integrates correctly with data loading and metric computation.

### End-to-End Tests

- **Purpose:**  
  Validate the complete workflow of the application from start to finish, simulating real-world usage scenarios.

- **Location:**  
  - `test_e2e.py`

- **Example:**  
  Testing the entire pipeline, including training, pruning, quantization, and deployment, to ensure that all steps execute correctly and the final model performs as expected.

## Running the Tests

Ensure that you have all dependencies installed and that your environment is correctly set up.

### Using Makefile

The project includes a `Makefile` for convenience. To run all tests, execute:

```bash
make test
```

### Using Pytest

Alternatively, you can run tests using `pytest` directly:

```bash
pytest tests/
```

### Test Output

Test results will be displayed in the console, indicating passed and failed tests. For detailed output, you can run:

```bash
pytest -v tests/
```

## Test Coverage

To assess the test coverage of the project, use the following command:

```bash
make coverage
```

This will generate a coverage report, highlighting areas of the codebase that are well-tested and identifying parts that may require additional tests.

**Note:**  
Ensure that `pytest-cov` is installed to enable coverage reporting.

## Best Practices

- **Isolated Tests:**  
  Each test should run independently without relying on external states or the outcomes of other tests.

- **Descriptive Test Names:**  
  Use clear and descriptive names for test functions to indicate their purpose and what they are verifying.

- **Comprehensive Coverage:**  
  Aim to cover all critical functionalities and edge cases to prevent regressions and ensure robustness.

- **Mocking External Dependencies:**  
  Use mocking techniques to simulate external dependencies, such as database connections or API calls, ensuring that tests remain fast and reliable.

- **Continuous Integration:**  
  Integrate the test suite with CI workflows (e.g., GitHub Actions) to automatically run tests on code pushes and pull requests, ensuring that new changes do not break existing functionalities.

## Contributing

Contributions to the test suite are highly encouraged! To contribute:

1. **Fork the Repository**
2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/add-new-test
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "test: add tests for new feature"
   ```

4. **Push to Your Fork**

   ```bash
   git push origin feature/add-new-test
   ```

5. **Open a Pull Request**

   Provide a clear description of the tests added and the functionalities they cover.

For detailed guidelines, refer to the [Best Practices](../docs/best_practices.md) documentation.

## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPLv3)](../LICENSE).

## Contact

For questions, suggestions, or support, please open an issue on the [GitHub repository](https://github.com/redx94/Dynamic-Neural-Network-Refinement/issues) or contact the maintainer at [qtt@null.net](mailto:qtt@null.net).

---
```
