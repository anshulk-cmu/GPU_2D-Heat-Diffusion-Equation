# Performance Mismatch Analysis: README vs L4 GPU Results

## The Core Issue

**The README contains hardcoded A100 GPU results, but the code generates dynamic results based on the actual hardware it runs on.** Running the same code on Google Colab's L4 GPU produces significantly different performance numbers.

## Hardware Comparison

| Specification | NVIDIA A100-SXM4 | NVIDIA L4 | Ratio |
|--------------|------------------|-----------|-------|
| Memory Bandwidth | 2048 GB/s | 300 GB/s | **6.8× lower** |
| Memory Type | HBM2e | GDDR6 | - |
| Compute Capability | 8.0 | 8.9 | - |
| Memory Size | 80 GB | 24 GB | 3.3× lower |

## Why L4 Results Differ

Since this heat diffusion solver is **memory-bandwidth limited** (README shows only 22% bandwidth utilization on A100), the L4's drastically lower bandwidth (300 vs 2048 GB/s) directly impacts performance:

- **A100 achieves 51× speedup** (1024×1024 grid) due to massive bandwidth
- **L4 expected speedup: ~7-10×** for the same workload
- Small grids (256×256) may show **GPU slower than CPU** on L4 due to overhead

## The Mismatch Explained

1. **README Claims**: Hardcoded from A100 execution (51× speedup, 452 GB/s bandwidth)
2. **Code Behavior**: Dynamically measures actual GPU performance at runtime
3. **L4 Reality**: Lower bandwidth → lower speedups → different benchmark numbers
4. **CPU baseline**: Also varies based on Colab CPU (different from original benchmark environment)

## Recommendation

The README should either:
- Clearly label results as "A100-specific benchmarks"
- Include L4 results table for Colab users
- Add dynamic result generation in the notebook output
