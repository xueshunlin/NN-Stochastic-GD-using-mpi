# Project: Stochastic Gradient Descent for Neural Networks using MPI

This project implements **Stochastic Gradient Descent (SGD)** for a single-hidden-layer neural network using **MPI parallelization**.  
The model is applied to the NYC Taxi dataset (`nytaxi2022.csv`) to predict the **total fare amount** from a set of trip features.

## Project Structure
```
proj/
├── README.md                           # How to run, dependencies, and experiment reproducibility
├── requirements.txt                    # Python dependencies (or conda env.yml)
├── src/
│   ├── data.py                         # Data loading, cleaning, feature engineering, global normalization
│   ├── model.py                        # Single hidden-layer NN (forward/backward, three activations)
│   ├── sgd_mpi.py                      # MPI training loop (Allreduce / Scatterv)
│   ├── eval.py                         # Parallel RMSE and training curve plotting
│   └── utils.py                        # Timing, logging, early stopping, random seed utilities
├── scripts/
│   └── run_grid.sh                     # Grid over activation × batch size × process count
├── report/
│   └── figs/                           # Training curves and timing comparison figures
└── nytaxi2022.csv                      # Dataset file (or place under `data/`)
```
