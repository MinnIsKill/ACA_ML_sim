<!-- GoL --->
# Conway's Game of Life

## Dependencies
### Python libraries
- PyQt5
- PyTorch
- numpy

### Others
- Python3

## Basic Workflow
```bash
$ python3 Game_of_Life.py
```

<!-- SmoothLife --->
---
# SmoothLife

## Dependencies
### Python libraries
- PyQt5
- PyTorch
- numpy
- matplotlib

### Others
- Python3

## Basic Workflow
1. Train the Machine Learning model (may take a while)
```bash
$ python3 SmoothLife_predictor_trainer_run.py
```
2. Start the simulation
```bash
$ python3 SmoothLife.py
```

<!-- Main Application --->
---
# Main Application (all models together)

## Dependencies
### Python libraries
- PyQt5
- PyTorch
- numpy
- matplotlib

### Others
- Python3

## Basic Workflow
1. Train the SmoothLife Machine Learning model (may take a while)
```bash
$ python3 SmoothLife_predictor_trainer_run.py
```
2. Start the hub
```bash
$ python3 main.py
```