## [Unreleased]

### Added

- **Codebase Documentation**: Added CI-compliant method docstrings (`Args:`, `Returns:`) and detailed inline "WHY" notes across core modules (`engine.py`, `physics.py`, `signals.py`, `acquisitions.py`, `config.py`, `simulator.py`) explaining transmon physics, Virtual Z gate phase accumulation, and schedule compilation dataflow.
- **CI/CD Pipeline**: Added `.gitlab-ci.yml` supporting `uv` package management, `ruff` linting, `pyright` type checking, and parallel `pytest` execution across Python 3.10 and 3.13.
- **Pre-commit Automation**: Integrated `.pre-commit-config.yaml` with local hooks for `ruff` formatting, `typos`, `vulture`, `mdformat`, and `commitizen`.
- **Repository Governance**: Added `AUTHORS.md`, `CHANGELOG.md`, `LICENSE`, and standard SPDX copyright header tags across Python files.

### Changed

- **Dependency Management**: Consolidated build configurations, dependencies, and `pytest` settings into `pyproject.toml`.
- **Code Formatting**: Reformatted codebase using `ruff format` to standardize double-quote string styling.

### Fixed

- **QCoDeS Compatibility**: Pinned `qcodes < 0.38.0` in `pyproject.toml` to prevent `ModuleNotFoundError` on `qcodes.utils.helpers` when imported by `quantify-core`.
- **CI Test Collection**: Added `libgl1`, `libglib2.0-0`, and `QT_QPA_PLATFORM="offscreen"` to the CI `before_script` to resolve `libGL.so.1` collection errors during headless GUI test runs.

### Removed

- **Legacy Configuration**: Removed `requirements.txt` and `pytest.ini`.
- **Docker Assets**: Cleaned up unneeded container configurations (`Dockerfile`, `.dockerignore`, `docker-compose.yml`).
