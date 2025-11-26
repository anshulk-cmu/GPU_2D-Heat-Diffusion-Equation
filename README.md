# GPU-Accelerated 2D Heat Diffusion Equation Solver

A high-performance parallel implementation of the 2D heat diffusion equation solver using CUDA (CuPy) and Python, demonstrating up to **51× speedup** over CPU implementations on NVIDIA A100 GPU.

## Overview

This project implements numerical solvers for the 2D heat diffusion equation using both CPU (NumPy) and GPU (CuPy/CUDA) backends. The solver uses finite difference methods with explicit (Forward Euler) and implicit (Jacobi iteration) time-stepping schemes to solve parabolic PDEs governing thermal diffusion processes.

## Mathematical Foundation

**Governing Equation:**
```
∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
```

Where:
- `u(x,y,t)`: Temperature distribution at position (x,y) and time t
- `α = 0.01 m²/s`: Thermal diffusivity coefficient
- Domain: [0,1] × [0,1] meters

**Discretization:**
- **Spatial**: 5-point stencil with central differences (2nd-order accurate)
- **Temporal**: Forward Euler (explicit) and Backward Euler with Jacobi iterations (implicit)
- **Stability**: CFL condition enforced with r = 0.20 (Δt ≤ h²/4α)
- **Boundary Conditions**: Dirichlet (0K on all edges)

## Implementation Features

### CPU Implementation (NumPy)
- Vectorized operations for efficient memory access
- Explicit scheme: Direct time-stepping
- Jacobi solver: Iterative convergence (ε = 10⁻⁶, max 10,000 iterations)

### GPU Implementation (CUDA via CuPy)

**Two optimization variants implemented:**

1. **Basic Global Memory Kernels**
   - Direct global memory access with coalesced reads/writes
   - Grid-stride loop structure for arbitrary problem sizes
   - 16×16 thread blocks (256 threads per block)

2. **Shared Memory Optimization**
   - 18×18 shared memory tiles for data reuse
   - Halo exchange for ghost cells at block boundaries
   - Reduced global memory traffic by ~3×

**Thread Configuration:**
- Block size: 16×16 = 256 threads
- Warps per block: 8
- Theoretical occupancy: 50%

## Hardware Specifications

**NVIDIA A100-SXM4-80GB (Ampere Architecture)**
- Compute Capability: 8.0
- Memory: 85.17 GB HBM2e
- Multiprocessors: 108 SMs
- Memory Bandwidth: ~2048 GB/s (peak)
- L2 Cache: 41.94 MB
- Clock Rate: 1.41 GHz
- Peak Performance: ~19.5 TFLOPS (FP64)

## Performance Results

### CPU Baseline (NumPy)

| Grid Size | Explicit (100 steps) | Jacobi (10 steps) | Iterations |
|-----------|---------------------|-------------------|------------|
| 256×256   | 0.0742 s | 6.9930 s | 10,000 |
| 512×512   | 0.3062 s | 20.4723 s | 10,000 |
| 1024×1024 | 1.9353 s | 97.7565 s | 10,000 |

### GPU Performance & Speedup

#### 256×256 Grid
- **Basic Explicit**: 0.077 s → **0.96× speedup**
- **Shared Explicit**: 0.087 s → 0.85× speedup
- **Basic Jacobi**: 3.368 s → **2.08× speedup** (10,000 iterations)
- **Shared Jacobi**: 3.399 s → 2.06× speedup

#### 512×512 Grid
- **Basic Explicit**: 0.039 s → **7.85× speedup**
- **Shared Explicit**: 0.039 s → **7.89× speedup**
- **Basic Jacobi**: 3.676 s → **5.57× speedup** (10,000 iterations)
- **Shared Jacobi**: 3.641 s → **5.62× speedup**

#### 1024×1024 Grid (Best Performance)
- **Basic Explicit**: 0.038 s → **50.64× speedup** ⚡
- **Shared Explicit**: 0.038 s → **51.07× speedup** ⚡
- **Basic Jacobi**: 3.774 s → **25.90× speedup** (10,000 iterations)
- **Shared Jacobi**: 3.710 s → **26.35× speedup**

### Memory Bandwidth Efficiency (2048×2048 Grid)

| Metric | Value | Peak Capability | Efficiency |
|--------|-------|-----------------|------------|
| Memory Bandwidth | 452.47 GB/s | 2048 GB/s | 22.2% |
| Compute Throughput | 103.69 GFLOPS | ~5000 GFLOPS | 2.1% |
| Kernel Execution Time | 44.4 ms | — | — |

*Note: Stencil computations are memory-bandwidth limited, not compute-limited*

## Numerical Validation

### GPU-CPU Consistency (50 time steps)

| Grid Size | L2 Error | L∞ Error | Relative L2 | Status |
|-----------|----------|----------|-------------|--------|
| 128×128 | 5.99e-16 | 2.13e-14 | 1.35e-16 | ✅ PASSED |
| 256×256 | 6.95e-16 | 2.84e-14 | 1.24e-16 | ✅ PASSED |
| 512×512 | 4.57e-16 | 2.84e-14 | 7.52e-17 | ✅ PASSED |

**Energy Conservation (256×256, 500 steps):**
- Initial Energy: 5.053319e+04
- Final Energy: 5.053319e+04
- Change: **0.0000%** ✅ EXCELLENT

## Key Findings

### Performance Insights
1. **Dramatic scaling with problem size**: 0.96× → 51× speedup as grid increases
2. **Shared memory provides minimal benefit** (~1% improvement), indicating memory bandwidth is not the primary bottleneck at these scales
3. **Jacobi method benefits significantly** from GPU parallelization (2-26× speedup)
4. **Memory bandwidth utilization** scales from 0.5% (256²) to 22.2% (2048²)
5. **GPU overhead dominates** at small grid sizes, causing CPU-parity or slower performance

### Technical Achievements
- ✅ Machine precision accuracy (errors < 1e-14)
- ✅ Perfect energy conservation
- ✅ Robust numerical stability
- ✅ Validated against analytical solutions
- ✅ Production-ready CUDA kernels

## Project Structure

```
2D_Heat_Diffusion_Equation_Solver.ipynb
├── Installation & GPU Verification
├── Mathematical Foundations
│   ├── Governing equations
│   ├── Finite difference discretization
│   ├── Stability analysis (CFL condition)
│   └── Numerical parameters
├── CPU Implementation
│   ├── Explicit Forward Euler solver
│   ├── Jacobi iterative solver
│   └── Performance benchmarking
├── GPU Implementation
│   ├── Basic CUDA kernels
│   ├── Shared memory optimization
│   └── Performance profiling
├── Numerical Validation
│   ├── CPU-GPU consistency checks
│   ├── Energy conservation tests
│   ├── Convergence analysis
│   └── Manufactured solution validation
└── Visualizations
    ├── Temperature evolution animations
    ├── Performance scaling plots
    └── Error analysis charts
```

## Usage

**Open in Google Colab:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anshulk-cmu/GPU_2D-Heat-Diffusion-Equation/blob/main/2D_Heat_Diffusion_Equation_Solver.ipynb)

**The notebook includes:**
- Complete mathematical derivations with theory
- CPU and GPU implementation with detailed comments
- Comprehensive benchmarking suite
- Numerical validation and error analysis
- Interactive visualizations of heat diffusion
- Performance profiling and optimization analysis

## Requirements

```bash
cupy-cuda12x      # GPU acceleration
numpy             # CPU computation
matplotlib        # Plotting
scikit-image      # Image processing
scipy             # Scientific computing
```

**Hardware Requirements:**
- NVIDIA GPU with CUDA compute capability ≥ 3.5
- 2-8 GB GPU memory (depending on grid size)
- CUDA Toolkit 12.x

## Applications

- 🔥 **Thermal Analysis**: Heat transfer in materials and electronics
- 🧪 **Materials Science**: Diffusion processes in solids and fluids
- 🖼️ **Image Processing**: Gaussian blur, denoising, edge detection
- 💹 **Financial Modeling**: Black-Scholes PDE, option pricing
- 🌊 **Fluid Dynamics**: Viscous flow, concentration diffusion
- 🎓 **Education**: Teaching numerical methods and GPU programming

## Citation

If you use this code in your research, please cite:

```bibtex
@software{gpu_heat_diffusion_2025,
  title={GPU-Accelerated 2D Heat Diffusion Equation Solver},
  author={Anshul Kumar},
  year={2025},
  url={https://github.com/anshulk-cmu/GPU_2D-Heat-Diffusion-Equation}
}
```

## License

MIT License - See LICENSE file for details

---

**GPU**: NVIDIA A100-SXM4-80GB | **Framework**: CuPy 13.x | **Python**: 3.10+ | **CUDA**: 12.x
