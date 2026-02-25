# PINN-QCD-Thermodynamics: Quasi-Particle Model Inversion

This repository contains the implementation of a Physics-Informed Neural Network (PINN) designed to invert lattice QCD thermodynamics data into quasi-particle masses and a temperature-dependent background correction $B(T)$.

## Overview
The code uses a double-branch ResNet architecture to learn the temperature dependence of quasi-particle masses ($m_g, m_{u/d}, m_s$) by fitting to HotQCD lattice data for:
- Entropy Density ($s/T^3$)
- Trace Anomaly ($\Delta/T^4$)
- Pressure ($P/T^4$)
- Energy Density ($\epsilon/T^4$)

The model enforces thermodynamic consistency and physical constraints (e.g., high-temperature limits).

## Files
- `PINN.py`: Main training script.
- `training_log.txt`: Training process logs (loss values, etc.).
- `result_plot_epoch_*.png`: Visualization of the fitting results and predicted masses.

## Usage
Run the script to start training:
```bash
python PINN.py
```
The script will automatically generate plots and logs in the current directory.

## Requirements
- PyTorch
- NumPy
- Matplotlib
- SciPy
