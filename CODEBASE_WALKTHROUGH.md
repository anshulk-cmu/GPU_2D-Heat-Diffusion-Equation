# GPU 2D Heat Diffusion Equation: Codebase Walkthrough

## Overview
This project implements a high-performance 2D heat diffusion equation solver using CUDA GPU acceleration via CuPy, optimized for **Google Colab with NVIDIA L4 GPU**, demonstrating up to **12-18× speedup** over CPU implementations on large grids (1024×1024).

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

## Performance Results (L4 GPU)

**Key Speedups:**
- **256×256 grid**: 1.0-1.5× (GPU overhead dominates)
- **512×512 grid**: 5-8× speedup
- **1024×1024 grid**: **12-18× speedup** (optimal)
- **Jacobi Implicit**: 2-22× speedup (depending on grid size)

**Validation:**
- CPU-GPU L2 errors: < 1e-14 (machine precision)
- Spatial convergence order: 2.0 (second-order accurate)
- Energy conservation: < 1e-10% drift over 500 timesteps

**Hardware Utilization (L4):**
- Memory bandwidth: ~65-85 GB/s (22-28% of L4's 300 GB/s peak)
- Compute throughput: ~20-30 GFLOPS
- The code is memory-bandwidth limited, not compute-limited

## GPU Comparison
L4 achieves moderate speedups (12-18×) compared to high-end HPC GPUs like A100 (45-55×) due to L4's GDDR6 memory bandwidth (300 GB/s) versus A100's HBM2e (2048 GB/s). However, L4's free availability on Google Colab makes it ideal for education and prototyping.

## Generated Outputs
The codebase produces eight visualization files including temperature evolution animations, 3D surface plots, convergence analysis charts, and comprehensive speedup comparisons, all stored in the `plots/` directory.
