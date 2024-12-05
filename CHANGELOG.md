# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-12-01

### Added

- Docker containerization with multi-stage Dockerfile.
- Distributed training support using PyTorch DDP.
- Integration and end-to-end tests.
- Data versioning with DVC and data validation with Great Expectations.
- Interactive documentation with MkDocs.
- Dependency vulnerability scanning with Dependabot and Bandit.
- Performance profiling scripts.
- Model serving API with FastAPI.
- Prometheus and Grafana monitoring setup.
- Automated model export to ONNX.
- Pre-commit hooks for code quality enforcement.

### Changed

- Updated `README.md` with new features and instructions.
- Enhanced CI workflows to include new tests and validation steps.
- Refactored codebase for better modularity and scalability.

### Fixed

- Corrected minor bugs in training scripts.
- Improved error handling in API endpoints.
