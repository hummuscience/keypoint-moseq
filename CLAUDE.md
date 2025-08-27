# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Keypoint MoSeq is an unsupervised machine learning method for animal behavior analysis. It learns stereotyped movement patterns from keypoint tracking data and when they occur over time. The package provides tools for fitting Motion Sequencing (MoSeq) models to keypoint tracking data.

## Development Commands

### Environment Setup
```bash
# Install in development mode
pip install -e .

# For CUDA support
pip install -e .[cuda]

# For documentation development
pip install -e .[dev]
```

### Documentation
```bash
# Build documentation
cd docs
make html

# Clean and rebuild
make clean html
```

### Testing
```bash
# Run tests with pytest (if tests are available)
pytest
```

## Architecture

### Core Modules

**keypoint_moseq/** - Main package directory
- `io.py` - Data I/O functions for loading/saving keypoints, models, and results
- `fitting.py` - Model fitting and training functions
- `analysis.py` - Post-processing and analysis tools
- `viz.py` - Visualization utilities for trajectories, grid movies, dendrograms
- `calibration.py` - Confidence score calibration tools
- `util.py` - Data formatting and utility functions

### Data Flow

1. **Setup**: Create project with `setup_project()`, configure via `config.yml`
2. **Data Loading**: Load keypoints from DeepLabCut/SLEAP/etc via `load_keypoints()`
3. **Preprocessing**: Remove outliers, format data with `format_data()`
4. **Calibration**: Learn keypoint error-confidence relationship via `noise_calibration()`
5. **PCA**: Fit PCA model to reduce dimensionality
6. **Model Fitting**: 
   - Initialize model with `init_model()`
   - Fit AR-HMM with `fit_model(ar_only=True)`
   - Fit full model with `fit_model(ar_only=False)`
7. **Results**: Extract and save results with `extract_results()`, `save_results_as_csv()`

### Key Configuration Parameters

Located in `{project_dir}/config.yml`:
- `bodyparts`: Keypoint names
- `use_bodyparts`: Subset for modeling (exclude tail for mice)
- `anterior_bodyparts`/`posterior_bodyparts`: For rotational alignment
- `video_dir`: Path to experiment videos
- `fps`: Frame rate
- `latent_dim`: PCA dimensions (typically ≤10)
- `kappa`: Controls syllable duration (higher = longer)
- `sigmasq_loc`: Expected centroid movement per frame

### Model States

Models contain:
- `z`: Syllable sequences
- `x`: Latent pose trajectories
- `v`: Centroid positions
- `h`: Heading angles
- AR parameters, transition matrices, noise estimates

Results are saved to `{project_dir}/{model_name}/results.h5` with structure:
```
results.h5
├── recording_name/
│   ├── syllable
│   ├── latent_state
│   ├── centroid
│   └── heading
```

### Dependencies

Core dependencies include JAX for computation, jax-moseq for models, and various visualization libraries. The package requires Python ≥3.10 and uses JAX with double precision enabled by default.