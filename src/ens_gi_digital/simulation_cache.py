"""
ENS-GI Digital Twin — Simulation Cache System
==============================================
Intelligent caching for expensive digital twin simulations used in Bayesian MCMC.

Problem: MCMC requires 5,000-50,000 simulations, each taking 100-500ms.
Solution: LRU cache with parameter-aware key generation and automatic cleanup.

Features:
- Hash-based cache keys for parameter combinations
- LRU eviction when cache exceeds size limit
- Hit/miss statistics tracking
- Disk-persistent cache for session reuse

Usage:
    from .simulation_cache import CachedSimulator, SimulationCache
    from .core import ENSGIDigitalTwin

    twin = ENSGIDigitalTwin(n_segments=10)
    cache = SimulationCache(cache_dir='.cache/simulations', max_size_mb=100)
    simulator = CachedSimulator(twin, cache, duration=2000.0)

    # First call - cache miss
    result = simulator({'g_Na': 120.0, 'g_K': 36.0, 'omega': 0.3})

    # Second call - cache hit (instant)
    result = simulator({'g_Na': 120.0, 'g_K': 36.0, 'omega': 0.3})

    print(cache.stats())  # Hit rate, size, etc.

Author: Mahdad (Bayesian MCMC Integration)
License: MIT
"""

import hashlib
import pickle
import os
import shutil
from pathlib import Path
from typing import Dict, Optional
import numpy as np


class SimulationCache:
    """LRU cache for expensive digital twin simulations."""

    def __init__(self, cache_dir: str = '.cache/simulations',
                 max_size_mb: int = 100):
        """
        Initialize simulation cache.

        Args:
            cache_dir: Directory to store cached results
            max_size_mb: Maximum cache size in megabytes
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.hits = 0
        self.misses = 0

    def _get_cache_key(self, params: Dict[str, float],
                       duration: float, dt: float) -> str:
        """
        Generate unique cache key from parameters and simulation settings.

        Args:
            params: Parameter dictionary
            duration: Simulation duration (ms)
            dt: Time step (ms)

        Returns:
            MD5 hash string
        """
        # Round parameters to 6 decimal places to avoid float precision issues
        rounded = {k: round(v, 6) for k, v in sorted(params.items())}
        key_str = f"{rounded}_{round(duration, 2)}_{round(dt, 3)}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, params: Dict[str, float],
            duration: float, dt: float) -> Optional[Dict]:
        """
        Retrieve cached simulation result if available.

        Args:
            params: Parameter dictionary
            duration: Simulation duration
            dt: Time step

        Returns:
            Cached result dict or None if not found
        """
        key = self._get_cache_key(params, duration, dt)
        cache_file = self.cache_dir / f"{key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.hits += 1
                    # Update access time (for LRU)
                    os.utime(cache_file, None)
                    return pickle.load(f)
            except Exception:
                # Corrupted cache file, remove it
                cache_file.unlink()
                self.misses += 1
                return None

        self.misses += 1
        return None

    def put(self, params: Dict[str, float],
            duration: float, dt: float, result: Dict):
        """
        Store simulation result in cache.

        Args:
            params: Parameter dictionary
            duration: Simulation duration
            dt: Time step
            result: Simulation result to cache
        """
        key = self._get_cache_key(params, duration, dt)
        cache_file = self.cache_dir / f"{key}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            # Cache write failed, not critical
            pass

        # Check cache size and clean if needed
        self._cleanup_if_needed()

    def _cleanup_if_needed(self):
        """Remove oldest cache files if size exceeds limit (LRU eviction)."""
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob('*.pkl'))

        if total_size > self.max_size_mb * 1024 * 1024:
            # Sort by access time (LRU = least recently used)
            files = sorted(self.cache_dir.glob('*.pkl'),
                          key=lambda f: f.stat().st_atime)

            # Remove oldest 20% of files
            n_remove = max(1, len(files) // 5)
            for f in files[:n_remove]:
                try:
                    f.unlink()
                except Exception:
                    pass

    def clear(self):
        """Clear entire cache and reset statistics."""
        for f in self.cache_dir.glob('*.pkl'):
            try:
                f.unlink()
            except Exception:
                pass
        self.hits = 0
        self.misses = 0

    def stats(self) -> Dict[str, float]:
        """
        Return cache statistics.

        Returns:
            Dictionary with hits, misses, hit_rate, size_mb
        """
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob('*.pkl'))
        total_requests = self.hits + self.misses

        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / total_requests if total_requests > 0 else 0.0,
            'size_mb': total_size / (1024 * 1024),
            'num_entries': len(list(self.cache_dir.glob('*.pkl')))
        }


class CachedSimulator:
    """
    Wrapper around ENS-GI digital twin that caches simulation results.

    This is the main interface for Bayesian MCMC - provides a simple
    callable that takes parameters and returns summary statistics.
    """

    def __init__(self, digital_twin,
                 cache: Optional[SimulationCache] = None,
                 duration: float = 2000.0,
                 dt: float = 0.1):
        """
        Initialize cached simulator.

        Args:
            digital_twin: ENSGIDigitalTwin instance
            cache: SimulationCache (creates new if None)
            duration: Simulation duration in ms (default: 2000ms = 2s)
            dt: Time step in ms (default: 0.1ms for speed)
        """
        self.twin = digital_twin
        self.cache = cache or SimulationCache()
        self.duration = duration
        self.dt = dt

    def __call__(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Run simulation with given parameters and return summary statistics.

        Args:
            params: Dictionary of parameters (g_Na, g_K, omega, etc.)

        Returns:
            Dictionary with summary statistics:
            - mean: Mean voltage across all neurons and time
            - std: Standard deviation
            - min: Minimum voltage
            - max: Maximum voltage
            - icc_freq: ICC frequency in cycles per minute (CPM)
            - spike_rate: Neural spike rate (spikes per neuron per second)
        """
        # Check cache first
        cached = self.cache.get(params, self.duration, self.dt)
        if cached is not None:
            return cached

        # Cache miss - run simulation
        # Update twin parameters
        for neuron in self.twin.network.neurons:
            if 'g_Na' in params:
                neuron.params.g_Na = params['g_Na']
            if 'g_K' in params:
                neuron.params.g_K = params['g_K']
            if 'g_CaL' in params:
                neuron.params.g_CaL = params['g_CaL']
            if 'g_L' in params:
                neuron.params.g_L = params['g_L']

        if 'omega' in params:
            self.twin.icc.params.omega = params['omega']
        if 'I_app' in params:
            self.twin.icc.params.I_app = params['I_app']

        # Run simulation
        try:
            result = self.twin.run(self.duration, dt=self.dt, verbose=False)
        except Exception as e:
            # Simulation failed (rare, but possible with extreme parameters)
            # Return neutral values
            return {
                'mean': -65.0,
                'std': 10.0,
                'min': -80.0,
                'max': -50.0,
                'icc_freq': 3.0,
                'spike_rate': 0.0
            }

        # Extract summary statistics
        if 'voltages' in result and result['voltages'] is not None:
            voltages = result['voltages']

            summary = {
                'mean': float(np.mean(voltages)),
                'std': float(np.std(voltages)),
                'min': float(np.min(voltages)),
                'max': float(np.max(voltages)),
            }

            # ICC frequency
            if hasattr(self.twin.icc, 'get_frequency'):
                try:
                    summary['icc_freq'] = float(self.twin.icc.get_frequency())
                except Exception:
                    summary['icc_freq'] = 3.0  # Default
            else:
                summary['icc_freq'] = 3.0

            # Spike rate (threshold-based spike detection)
            spike_threshold = -20.0  # mV
            spikes = np.sum(voltages > spike_threshold)
            n_neurons = voltages.shape[1] if len(voltages.shape) > 1 else 1
            duration_sec = self.duration / 1000.0
            summary['spike_rate'] = float(spikes / (n_neurons * duration_sec))
        else:
            # No voltage data (shouldn't happen, but be defensive)
            summary = {
                'mean': -65.0,
                'std': 10.0,
                'min': -80.0,
                'max': -50.0,
                'icc_freq': 3.0,
                'spike_rate': 0.0
            }

        # Cache result
        self.cache.put(params, self.duration, self.dt, summary)

        return summary

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return self.cache.stats()

    def clear_cache(self):
        """Clear simulation cache."""
        self.cache.clear()
