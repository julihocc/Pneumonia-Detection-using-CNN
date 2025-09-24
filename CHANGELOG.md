# Changelog

All notable changes to this project will be documented in this file.

The format loosely follows Keep a Changelog and adheres to Semantic Versioning.

## [2.1.0] - 2025-09-23
### Added
- Simple CNN pneumonia detection notebook: `notebooks/pneumonia_detection_simple.ipynb`
- Script variant of simple CNN workflow: `notebooks/pneumonia_detection_simple.py`
- Legacy notebook relocated under `notebooks/` for clarity.
- Added `.context` and `outputs/` to `.gitignore`.
- Added `CLAUDE.md` documentation (developer / AI assistant guidance).

### Changed
- Version bump across codebase to 2.1.0.
- Refactored parts of modern notebook and imports to resolve issues.
- Improved installation instructions and clarified performance notes in README / docs.
- Development dependency structure refined.
- Enhanced pneumonia detection notebook with more robust error handling.

### Fixed
- Import path and environment resolution issues (general refactors).
- Stability improvements in simple notebook flow (error handling fix commit `2cea226`).

### Removed
- Outdated egg-info and cached artifacts.
- Stale `.pyc` files and redundant temporary entries.
- Removed `.ipynb` blanket ignore to allow tracking curated notebooks.

### Housekeeping
- Multiple ignore rule improvements (`.gitignore`).
- Cleanup: removed cached / transient directories.

### Full Commit Log (since 2.0.0)
- 3bff086 merge: integrate simple CNN notebook and release 2.1.0
- 2325e22 chore(release): bump version to 2.1.0 (simple CNN notebook, legacy relocation, gitignore updates)
- ba38779 Ignores output directories and session context
- 750c85a notebooks: add simple CNN pneumonia detection notebook and script; move legacy notebook; update .gitignore with outputs/ and .context
- a2c34ab Refactors and fixes import issues
- af78082 Removes .ipynb from .gitignore
- 7e108e5 Updates dependencies
- 0cb2cb6 Removes egg-info directory
- d6d3a40 Removes cached files
- b85a6f2 Removes stale pyc files
- 291b27a Updates .gitignore to exclude common files
- efef29d Adds CLAUDE.md documentation
- 2b9ed22 Improves installation and clarifies performance
- 8124f82 Refactors development dependencies
- 2cea226 fix: enhance pneumonia detection notebook with robust error handling

## [2.0.0] - 2025-09-23
### Major Refactor
Initial modern MLOps release refactoring legacy monolithic code into modular architecture with training pipeline, API serving, MLflow integration, CLI tools, Docker environment, and extensive documentation.

(See tag `v2.0.0` annotated message for full details.)

---
Next: For future releases, add unreleased section at top and automate changelog generation via a GitHub Action if desired.
