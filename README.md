# GPU-Accelerated 2D Heat Diffusion Equation Solver

A high-performance parallel implementation of the 2D heat diffusion equation solver using CUDA (via CuPy) and Python, demonstrating significant speedups over CPU-based computations.

## Overview

This project implements numerical solvers for the 2D heat diffusion equation using both CPU (NumPy) and GPU (CuPy) backends. The heat equation describes temperature distribution over time in a 2D domain, solved using finite difference methods with explicit and implicit (Jacobi iteration) schemes.

## Mathematical Foundation

The 2D heat diffusion equation is:
```
∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
```

Where:
- `u(x,y,t)`: Temperature at position (x,y) and time t
- `α`: Thermal diffusivity coefficient
- Discretized using finite differences on a uniform grid
- Explicit scheme: Forward Euler in time
- Implicit scheme: Jacobi iterative method

## Implementation Features

- **CPU Implementation**: Vectorized NumPy operations
- **GPU Implementation**: CUDA kernels via CuPy with two variants:
  - Basic global memory kernels
  - Optimized shared memory kernels
- **Stability Analysis**: CFL condition enforcement (Δt ≤ h²/4α)
- **Convergence Criteria**: Jacobi iterations with ε = 10⁻⁶ tolerance

## Performance Results

Benchmarks on NVIDIA A100 GPU comparing CPU vs GPU implementations:

### 256×256 Grid
- **Explicit Scheme**: 0.074s (CPU) → 0.077s (GPU), 0.96× speedup
- **Jacobi Method**: 6.99s (CPU) → 3.37s (GPU), **2.08× speedup**

### 512×512 Grid
- **Explicit Scheme**: 0.306s (CPU) → 0.039s (GPU), **7.89× speedup**
- **Jacobi Method**: 20.47s (CPU) → 3.64s (GPU), **5.62× speedup**

### 1024×1024 Grid
- **Explicit Scheme**: 1.94s (CPU) → 0.038s (GPU), **51× speedup**
- **Jacobi Method**: 97.76s (CPU) → 3.71s (GPU), **26.35× speedup**

## Key Findings

- GPU acceleration scales dramatically with problem size
- Achieved up to **51× speedup** for large grids (1024×1024)
- Shared memory optimization provides marginal improvements
- Memory bandwidth is the primary bottleneck for stencil computations
- Jacobi iterations benefit significantly from GPU parallelization

## Usage

Open the Jupyter notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anshulk-cmu/GPU_2D-Heat-Diffusion-Equation/blob/main/2D_Heat_Diffusion_Equation_Solver.ipynb)

The notebook includes:
- Complete mathematical derivations
- CPU and GPU implementation code
- Comprehensive benchmarks
- Visualization of heat diffusion patterns
- Performance comparison plots

## Requirements

```
cupy-cuda12x
numpy
matplotlib
scikit-image
scipy
```

## Applications

- Thermal analysis and heat transfer simulations
- Diffusion processes in materials science
- Image processing (Gaussian blur, denoising)
- Financial modeling (Black-Scholes PDE)
- Scientific computing education

---

**GPU**: NVIDIA A100 | **Framework**: CuPy 13.x | **Python**: 3.10+
