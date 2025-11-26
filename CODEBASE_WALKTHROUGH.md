# GPU 2D Heat Diffusion Equation: Codebase Walkthrough

## Overview
This project implements a high-performance 2D heat diffusion equation solver using CUDA GPU acceleration via CuPy, demonstrating up to **51× speedup** over CPU implementations on NVIDIA A100 GPU.

## Mathematical Foundation
The solver implements the 2D heat equation: `∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)` where `u(x,y,t)` represents temperature distribution and `α = 0.01 m²/s` is thermal diffusivity. The implementation uses finite difference methods with a 5-point stencil for spatial discretization and both explicit (Forward Euler) and implicit (Jacobi iteration) time-stepping schemes.

## Code Architecture

### Core Components (gpu_2d_heat_diffusion_equation.py:1-1550)
1. **CPU Implementations** (lines 273-327): Vectorized NumPy solvers for explicit scheme and Jacobi iteration baseline
2. **GPU CUDA Kernels** (lines 332-408): Three optimized kernels - basic explicit, shared memory explicit, and Jacobi iteration
3. **GPU Solvers** (lines 411-462): Wrapper functions managing device memory and kernel execution
4. **Validation Suite** (lines 545-892): Analytical solutions, error metrics, convergence analysis, and energy conservation tests
5. **Benchmarking** (lines 465-533): Systematic performance measurement across multiple grid sizes
6. **Visualization** (lines 1083-1367): Temperature evolution animations, 3D surface plots, and performance charts

## Performance Results

**Key Speedups (1024×1024 grid):**
- Basic Explicit: **50.64× faster** than CPU
- Shared Memory: **51.07× faster** than CPU
- Jacobi Implicit: **26.35× faster** than CPU

**Validation:**
- CPU-GPU L2 errors: < 1e-14 (machine precision)
- Spatial convergence order: 2.0 (second-order accurate)
- Energy conservation: 0.00% change over 500 timesteps

**Hardware Utilization (2048×2048):**
- Memory bandwidth: 452 GB/s (22% of A100 peak)
- Compute throughput: 104 GFLOPS
- The code is memory-bandwidth limited, not compute-limited

## Generated Outputs
The codebase produces eight visualization files including temperature evolution animations, 3D surface plots, convergence analysis charts, and comprehensive speedup comparisons, all stored in the `plots/` directory.
