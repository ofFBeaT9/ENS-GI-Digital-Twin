"""
Performance Profiling for ENS-GI Digital Twin
==============================================

Profiles simulation performance to identify bottlenecks.

Usage:
    python profile_performance.py

Outputs:
    - performance_profile.txt: Detailed profiling stats
    - Console: Performance summary and recommendations
"""

import cProfile
import pstats
import io
import time
import numpy as np
from ens_gi_digital import ENSGIDigitalTwin


def profile_basic_simulation():
    """Profile a basic simulation run."""
    print("Profiling basic simulation (10 segments, 1000 ms)...")

    profiler = cProfile.Profile()
    profiler.enable()

    twin = ENSGIDigitalTwin(n_segments=10)
    twin.apply_profile('healthy')
    result = twin.run(1000, dt=0.05, I_stim={5: 10.0}, verbose=False)

    profiler.disable()

    return profiler


def profile_large_network():
    """Profile a large network simulation."""
    print("Profiling large network (20 segments, 500 ms)...")

    profiler = cProfile.Profile()
    profiler.enable()

    twin = ENSGIDigitalTwin(n_segments=20)
    twin.apply_profile('healthy')
    result = twin.run(500, dt=0.05, I_stim={10: 10.0}, verbose=False)

    profiler.disable()

    return profiler


def benchmark_scaling():
    """Benchmark simulation time vs network size."""
    print("\nBenchmarking scaling characteristics...")
    print("=" * 60)

    sizes = [5, 10, 15, 20, 25, 30]
    times = []

    for n in sizes:
        twin = ENSGIDigitalTwin(n_segments=n)
        twin.apply_profile('healthy')

        start = time.time()
        result = twin.run(500, dt=0.1, verbose=False)
        elapsed = time.time() - start

        times.append(elapsed)

        print(f"  {n:2d} segments: {elapsed:6.2f} s  "
              f"(speedup: {500/1000/elapsed:5.1f}× real-time)")

    print("=" * 60)

    # Analyze scaling
    times = np.array(times)
    sizes = np.array(sizes)

    # Fit O(n) and O(n²) scaling
    coeffs_linear = np.polyfit(sizes, times, 1)
    coeffs_quadratic = np.polyfit(sizes, times, 2)

    linear_fit = np.polyval(coeffs_linear, sizes)
    quadratic_fit = np.polyval(coeffs_quadratic, sizes)

    linear_error = np.mean((times - linear_fit)**2)
    quadratic_error = np.mean((times - quadratic_fit)**2)

    print(f"\nScaling Analysis:")
    print(f"  Linear O(n) error:     {linear_error:.6f}")
    print(f"  Quadratic O(n²) error: {quadratic_error:.6f}")

    if quadratic_error < linear_error * 0.8:
        print(f"  → Scaling appears QUADRATIC (network connectivity bottleneck)")
    else:
        print(f"  → Scaling appears LINEAR (good scaling)")

    return sizes, times


def benchmark_timestep():
    """Benchmark simulation time vs timestep size."""
    print("\nBenchmarking timestep scaling...")
    print("=" * 60)

    dts = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    times = []

    for dt in dts:
        twin = ENSGIDigitalTwin(n_segments=10)
        twin.apply_profile('healthy')

        start = time.time()
        result = twin.run(200, dt=dt, verbose=False)
        elapsed = time.time() - start

        times.append(elapsed)
        n_steps = int(200 / dt)

        print(f"  dt={dt:4.2f} ms: {elapsed:6.2f} s  ({n_steps:5d} steps, "
              f"{elapsed*1000/n_steps:.3f} ms/step)")

    print("=" * 60)

    return dts, times


def print_profiling_report(profiler, filename='performance_profile.txt'):
    """Print and save profiling report."""
    # Save to file
    profiler.dump_stats(filename.replace('.txt', '.prof'))

    # Print to console
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions

    report = s.getvalue()

    # Save text report
    with open(filename, 'w') as f:
        f.write(report)

    print(f"\nTop 30 functions by cumulative time:\n")
    print(report)


def print_hotspot_recommendations(profiler):
    """Analyze profiling results and print optimization recommendations."""
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')

    # Get top functions
    stats = ps.stats
    sorted_stats = sorted(stats.items(), key=lambda x: x[1][3], reverse=True)

    print("\n" + "=" * 60)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 60)

    hotspots = []
    for func_key, (cc, nc, tt, ct, callers) in sorted_stats[:10]:
        filename, line, func_name = func_key
        if 'ens_gi' in filename:
            hotspots.append((func_name, ct, tt))

    if hotspots:
        print("\nHotspots in ENS-GI code:")
        for func_name, cumtime, tottime in hotspots[:5]:
            print(f"  {func_name:30s}  {cumtime:6.2f}s cumulative, {tottime:6.2f}s self")

        print("\nOptimization suggestions:")

        # Check for specific bottlenecks
        func_names = [h[0] for h in hotspots]

        if any('step' in name for name in func_names):
            print("  1. Consider reducing timestep (dt) for fewer integration steps")

        if any('gate' in name.lower() or 'alpha' in name.lower() for name in func_names):
            print("  2. Ion channel gating calculations may benefit from vectorization")

        if any('network' in name.lower() or 'connect' in name.lower() for name in func_names):
            print("  3. Network connectivity loop may benefit from sparse matrix operations")

        if any('record' in name.lower() for name in func_names):
            print("  4. Consider recording only every N steps (use record_every parameter)")

        print("  5. For large networks (>20 segments), consider using Numba JIT compilation")
        print("  6. For very large networks, consider GPU acceleration (CuPy)")

    print("=" * 60)


def analyze_memory_usage():
    """Analyze memory usage characteristics."""
    print("\nMemory Usage Analysis...")
    print("=" * 60)

    import sys

    sizes = [5, 10, 20, 50]

    for n in sizes:
        twin = ENSGIDigitalTwin(n_segments=n)
        twin.apply_profile('healthy')

        # Run short simulation
        result = twin.run(100, dt=0.1, verbose=False)

        # Estimate memory
        mem_voltages = result['voltages'].nbytes / 1024 / 1024  # MB
        mem_forces = result['forces'].nbytes / 1024 / 1024
        mem_calcium = result['calcium'].nbytes / 1024 / 1024
        mem_total = mem_voltages + mem_forces + mem_calcium

        print(f"  {n:2d} segments (100ms, dt=0.1):")
        print(f"    Recording memory: {mem_total:.2f} MB")
        print(f"    Per segment:      {mem_total/n:.2f} MB/segment")

    print("\nMemory recommendations:")
    print("  - For long simulations (>10s), reduce timestep or use record_every")
    print("  - For large networks (>50 segments), monitor RAM usage")
    print("  - Recorded arrays scale as O(duration/dt × n_segments)")
    print("=" * 60)


def main():
    """Run complete performance profiling suite."""
    print("\n" + "=" * 60)
    print("ENS-GI DIGITAL TWIN PERFORMANCE PROFILING")
    print("=" * 60)

    # 1. Profile basic simulation
    profiler_basic = profile_basic_simulation()
    print_profiling_report(profiler_basic, 'performance_profile_basic.txt')

    # 2. Profile large network
    profiler_large = profile_large_network()
    print_profiling_report(profiler_large, 'performance_profile_large.txt')

    # 3. Benchmark scaling
    sizes, times = benchmark_scaling()

    # 4. Benchmark timestep
    dts, dt_times = benchmark_timestep()

    # 5. Memory analysis
    analyze_memory_usage()

    # 6. Optimization recommendations
    print_hotspot_recommendations(profiler_basic)

    print("\n" + "=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)
    print("\nOutput files:")
    print("  - performance_profile_basic.txt (text report)")
    print("  - performance_profile_basic.prof (binary, use snakeviz/gprof2dot)")
    print("  - performance_profile_large.txt")
    print("  - performance_profile_large.prof")
    print("\nTo visualize:")
    print("  pip install snakeviz")
    print("  snakeviz performance_profile_basic.prof")
    print("=" * 60)


if __name__ == '__main__':
    main()
