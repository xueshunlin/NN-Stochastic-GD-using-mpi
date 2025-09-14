# Project: Stochastic Gradient Descent for Neural Networks using MPI

This project implements **Stochastic Gradient Descent (SGD)** for a single-hidden-layer neural network using **MPI parallelization**.  
The model is applied to the NYC Taxi dataset (`nytaxi2022.csv`) to predict the **total fare amount** from a set of trip features.

## Project Structure
```t
proj/
  README.md                 # How to run, dependencies, and experiment reproducibility
  requirements.txt          # Python dependencies (or conda env.yml for environment setup)
  src/
    data.py                 # Data loading, cleaning, feature engineering, global normalization
    model.py                # Single hidden-layer neural network: forward/backward, three activations
    sgd_mpi.py              # Main training program (MPI Allreduce / Scatterv implementation)
    eval.py                 # Parallel RMSE computation and training curve plotting
    utils.py                # Utilities: timing, logging, early stopping, random seed
  scripts/
    run_grid.sh             # Batch script for running activation × batch size × process count experiments
  report/
    figs/                   # Training curves and timing comparison figures
  nytaxi2022.csv            # Dataset file (or placed under a `data/` folder)
```