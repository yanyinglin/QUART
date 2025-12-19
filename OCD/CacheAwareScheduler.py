"""
Cache-Aware Scheduling with KeysManager and KL Divergence Optimization
Implements intelligent scheduling using hierarchical caching and dispersed placement.
"""

import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import subprocess
import json
from kubernetes import client, config
from prometheus_api_client import PrometheusConnect


@dataclass
class CachedParameter:
    """Represents cached model parameters on a server."""
    stage_name: str
    model_name: str
    parameter_size: float  # GB
    cached_time: float
    last_access_time: float
    hit_count: int = 0
    server_name: str = ""


@dataclass
class ServerCache:
    """Represents cache state on a single server."""
    server_name: str
    total_memory: float  # GB
    used_memory: float = 0.0
    cached_parameters: Dict[str, CachedParameter] = field(default_factory=dict)
    gpu_count: int = 0
    available_gpus: int = 0
    
    def memory_utilization(self) -> float:
        """Get memory utilization percentage."""
        if self.total_memory == 0:
            return 0.0
        return (self.used_memory / self.total_memory) * 100.0
    
    def has_cached(self, stage_name: str) -> bool:
        """Check if stage parameters are cached."""
        return stage_name in self.cached_parameters
    
    def add_cached(self, param: CachedParameter) -> bool:
        """Add cached parameter if space available."""
        if self.used_memory + param.parameter_size > self.total_memory * 0.85:
            return False  # Don't exceed 85% memory
        
        self.cached_parameters[param.stage_name] = param
        self.used_memory += param.parameter_size
        param.server_name = self.server_name
        return True
    
    def evict_lru(self) -> Optional[CachedParameter]:
        """Evict least recently used cached parameter."""
        if not self.cached_parameters:
            return None
        
        # Find LRU
        lru_stage = min(
            self.cached_parameters.keys(),
            key=lambda k: self.cached_parameters[k].last_access_time
        )
        
        param = self.cached_parameters.pop(lru_stage)
        self.used_memory -= param.parameter_size
        
        return param


class KeysManager:
    """
    Hierarchical Key-Stage Manager for model parameter caching.
    Uses copy-on-write mechanisms for sub-second instance creation.
    """
    
    def __init__(self, memory_threshold: float = 0.85,
                 cache_eviction_policy: str = "lru"):
        """
        Initialize KeysManager.
        
        Args:
            memory_threshold: Maximum memory utilization (0-1)
            cache_eviction_policy: Cache eviction policy ("lru" or "lfu")
        """
        self.memory_threshold = memory_threshold
        self.cache_eviction_policy = cache_eviction_policy
        
        # Server cache state
        self.server_caches: Dict[str, ServerCache] = {}
        
        # Global cache directory (shared across servers)
        self.cache_dir = "/data/model/openfaas/"
        
        # Scheduling history for cache-aware decisions
        self.scheduling_history: List[Dict] = []
    
    def initialize_server_cache(self, server_name: str, total_memory: float,
                               gpu_count: int):
        """
        Initialize cache tracking for a server.
        
        Args:
            server_name: Server hostname
            total_memory: Total server memory in GB
            gpu_count: Number of GPUs on server
        """
        self.server_caches[server_name] = ServerCache(
            server_name=server_name,
            total_memory=total_memory,
            gpu_count=gpu_count,
            available_gpus=gpu_count
        )
    
    def cache_parameters_cow(self, stage_name: str, model_name: str,
                            parameter_size: float, server_name: str) -> bool:
        """
        Cache model parameters using copy-on-write fork mechanism.
        
        Args:
            stage_name: Pipeline stage name
            model_name: Model name
            parameter_size: Parameter size in GB
            server_name: Target server
        
        Returns:
            True if successfully cached
        """
        if server_name not in self.server_caches:
            print(f"Server {server_name} not initialized")
            return False
        
        server_cache = self.server_caches[server_name]
        
        # Check if already cached
        if server_cache.has_cached(stage_name):
            # Update access time
            cached = server_cache.cached_parameters[stage_name]
            cached.last_access_time = time.time()
            cached.hit_count += 1
            print(f"✓ Cache hit for {stage_name} on {server_name}")
            return True
        
        # Check memory availability
        while server_cache.memory_utilization() > self.memory_threshold * 100:
            evicted = server_cache.evict_lru()
            if evicted:
                print(f"Evicted {evicted.stage_name} from {server_name} (LRU)")
            else:
                break
        
        # Create cached parameter entry
        param = CachedParameter(
            stage_name=stage_name,
            model_name=model_name,
            parameter_size=parameter_size,
            cached_time=time.time(),
            last_access_time=time.time()
        )
        
        # Add to cache
        if server_cache.add_cached(param):
            print(f"✓ Cached {stage_name} on {server_name} "
                  f"(mem: {server_cache.memory_utilization():.1f}%)")
            return True
        else:
            print(f"✗ Failed to cache {stage_name} on {server_name} (insufficient memory)")
            return False
    
    def get_cache_hit_rate(self, server_name: str) -> float:
        """Get cache hit rate for a server."""
        if server_name not in self.server_caches:
            return 0.0
        
        total_hits = sum(
            p.hit_count for p in self.server_caches[server_name].cached_parameters.values()
        )
        total_accesses = len(self.scheduling_history)
        
        if total_accesses == 0:
            return 0.0
        
        return total_hits / total_accesses
    
    def get_servers_with_cached_stage(self, stage_name: str) -> List[str]:
        """Get list of servers that have cached this stage."""
        servers = []
        for server_name, cache in self.server_caches.items():
            if cache.has_cached(stage_name):
                servers.append(server_name)
        return servers
    
    def instantiate_from_cache(self, stage_name: str, server_name: str,
                              gpu_uuid: str) -> bool:
        """
        Instantiate inference process from cached parameters.
        
        Args:
            stage_name: Pipeline stage name
            server_name: Server with cached parameters
            gpu_uuid: Target GPU UUID
        
        Returns:
            True if successful
        """
        if server_name not in self.server_caches:
            return False
        
        server_cache = self.server_caches[server_name]
        
        if not server_cache.has_cached(stage_name):
            print(f"✗ {stage_name} not cached on {server_name}")
            return False
        
        # Update cache statistics
        cached = server_cache.cached_parameters[stage_name]
        cached.last_access_time = time.time()
        cached.hit_count += 1
        
        # In real implementation, this would:
        # 1. Fork process with COW (copy-on-write)
        # 2. Load parameters from server memory to GPU
        # 3. Initialize inference service
        
        print(f"✓ Instantiated {stage_name} from cache on {server_name} -> GPU {gpu_uuid}")
        print(f"  Estimated loading time: ~0.8s (memory-to-GPU)")
        
        return True


class KLDivergenceOptimizer:
    """
    Dispersed placement strategy using KL divergence optimization.
    Optimizes stage distribution to maximize scaling opportunities.
    """
    
    def __init__(self, learning_rate: float = 0.1, max_iterations: int = 100,
                 convergence_threshold: float = 0.01):
        """
        Initialize KL divergence optimizer.
        
        Args:
            learning_rate: Learning rate for gradient descent (alpha_t)
            max_iterations: Maximum optimization iterations
            convergence_threshold: KL divergence threshold for convergence (epsilon)
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
    
    def compute_kl_divergence(self, P: np.ndarray, Q: np.ndarray) -> float:
        """
        Compute KL divergence D_KL(P || Q).
        
        Args:
            P: Current placement distribution (N x M)
            Q: Target (uniform) distribution (N x M)
        
        Returns:
            KL divergence value
        """
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        P_safe = P + epsilon
        Q_safe = Q + epsilon
        
        # D_KL(P || Q) = sum(P * log(P / Q))
        kl_div = np.sum(P_safe * np.log(P_safe / Q_safe))
        
        return kl_div
    
    def compute_gradient(self, P: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """
        Compute gradient of KL divergence.
        
        dD_KL/dP_ij = 1 + log(P_ij / Q_ij)
        
        Args:
            P: Current placement distribution
            Q: Target distribution
        
        Returns:
            Gradient matrix
        """
        epsilon = 1e-10
        gradient = 1 + np.log((P + epsilon) / (Q + epsilon))
        
        return gradient
    
    def optimize_placement(self, num_stages: int, num_servers: int,
                          critical_stages: Set[int],
                          server_capacities: np.ndarray,
                          initial_placement: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Optimize placement using gradient descent.
        
        Args:
            num_stages: Number of pipeline stages (N)
            num_servers: Number of servers (M)
            critical_stages: Set of critical stage indices
            server_capacities: Available capacity on each server
            initial_placement: Initial placement probabilities (N x M)
        
        Returns:
            Optimized placement probabilities (N x M)
        """
        # Initialize placement distribution
        if initial_placement is None:
            # Start with uniform distribution
            P = np.ones((num_stages, num_servers)) / num_servers
        else:
            P = initial_placement.copy()
        
        # Target distribution (uniform for dispersion)
        Q = np.ones((num_stages, num_servers)) / num_servers
        
        # Adjust target based on server capacities
        capacity_weights = server_capacities / np.sum(server_capacities)
        Q = Q * capacity_weights.reshape(1, -1)
        Q = Q / np.sum(Q, axis=1, keepdims=True)  # Renormalize
        
        print(f"\nOptimizing placement for {num_stages} stages across {num_servers} servers")
        print(f"Critical stages: {critical_stages}")
        
        for iteration in range(self.max_iterations):
            # Compute KL divergence
            kl_div = self.compute_kl_divergence(P, Q)
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: KL divergence = {kl_div:.6f}")
            
            # Check convergence
            if kl_div < self.convergence_threshold:
                print(f"Converged at iteration {iteration}")
                break
            
            # Compute gradient
            gradient = self.compute_gradient(P, Q)
            
            # Adaptive learning rate (decreases over iterations)
            alpha_t = self.learning_rate / (1 + 0.01 * iteration)
            
            # Gradient descent update: P_ij^(t+1) = P_ij^(t) - alpha_t * gradient
            P = P - alpha_t * gradient
            
            # Project onto probability simplex (each row sums to 1)
            P = np.maximum(P, 0)  # Non-negative
            P = P / (np.sum(P, axis=1, keepdims=True) + 1e-10)  # Normalize
            
            # Emphasize critical stages (higher weight for dispersion)
            for stage_idx in critical_stages:
                # Encourage more uniform distribution for critical stages
                P[stage_idx] = 0.7 * P[stage_idx] + 0.3 * Q[stage_idx]
                P[stage_idx] = P[stage_idx] / np.sum(P[stage_idx])
        
        return P
    
    def select_server(self, placement_probs: np.ndarray, stage_idx: int,
                     available_servers: List[int]) -> int:
        """
        Select server based on placement probabilities.
        
        Args:
            placement_probs: Placement probability matrix (N x M)
            stage_idx: Index of stage to place
            available_servers: List of available server indices
        
        Returns:
            Selected server index
        """
        if not available_servers:
            return -1
        
        # Get probabilities for this stage on available servers
        probs = placement_probs[stage_idx, available_servers]
        probs = probs / np.sum(probs)  # Renormalize
        
        # Sample from distribution
        selected = np.random.choice(available_servers, p=probs)
        
        return selected


class CacheAwareScheduler:
    """
    Complete cache-aware scheduling system integrating KeysManager and KL optimization.
    """
    
    def __init__(self, prometheus_client: PrometheusConnect):
        """
        Initialize cache-aware scheduler.
        
        Args:
            prometheus_client: Prometheus client for metrics
        """
        self.prometheus = prometheus_client
        self.keys_manager = KeysManager()
        self.kl_optimizer = KLDivergenceOptimizer()
        
        # Scheduling state
        self.stage_to_server: Dict[str, str] = {}
        self.server_to_stages: Dict[str, List[str]] = {}
        
        # Load Kubernetes config
        try:
            config.load_kube_config()
        except:
            config.load_incluster_config()
        
        self.core_v1 = client.CoreV1Api()
    
    def initialize_cluster(self, servers: List[Tuple[str, float, int]]):
        """
        Initialize cluster servers.
        
        Args:
            servers: List of (server_name, memory_gb, gpu_count)
        """
        print("Initializing cluster servers:")
        for server_name, memory_gb, gpu_count in servers:
            self.keys_manager.initialize_server_cache(server_name, memory_gb, gpu_count)
            self.server_to_stages[server_name] = []
            print(f"  - {server_name}: {memory_gb}GB RAM, {gpu_count} GPUs")
    
    def schedule_critical_stage(self, stage_name: str, model_name: str,
                               parameter_size: float,
                               is_critical: bool = False) -> Tuple[str, str]:
        """
        Schedule a pipeline stage with cache awareness.
        
        Args:
            stage_name: Stage deployment name
            model_name: Model name
            parameter_size: Parameter size in GB
            is_critical: Whether this is a critical stage
        
        Returns:
            (server_name, gpu_uuid) for placement
        """
        # Check for cached parameters
        servers_with_cache = self.keys_manager.get_servers_with_cached_stage(stage_name)
        
        if servers_with_cache and is_critical:
            # Prefer servers with cached parameters for critical stages
            print(f"🎯 Critical stage {stage_name} found in cache on {servers_with_cache}")
            server_name = servers_with_cache[0]  # Select first available
            
            # Find available GPU on this server
            server_cache = self.keys_manager.server_caches[server_name]
            if server_cache.available_gpus > 0:
                # In real implementation, query actual GPU availability
                gpu_uuid = f"GPU-{server_name}-{server_cache.available_gpus}"
                
                # Instantiate from cache
                self.keys_manager.instantiate_from_cache(stage_name, server_name, gpu_uuid)
                
                return server_name, gpu_uuid
        
        # No cache hit or not critical - use dispersed placement
        print(f"📍 Scheduling {stage_name} using dispersed placement")
        
        # Get available servers with capacity
        available_servers = []
        server_names = []
        capacities = []
        
        for server_name, cache in self.keys_manager.server_caches.items():
            if cache.available_gpus > 0:
                available_servers.append(len(server_names))
                server_names.append(server_name)
                capacities.append(cache.available_gpus)
        
        if not available_servers:
            print("❌ No available servers with GPU capacity")
            return "", ""
        
        # Use KL divergence optimization for placement
        num_servers = len(server_names)
        placement_probs = self.kl_optimizer.optimize_placement(
            num_stages=1,
            num_servers=num_servers,
            critical_stages={0} if is_critical else set(),
            server_capacities=np.array(capacities)
        )
        
        # Select server
        server_idx = self.kl_optimizer.select_server(
            placement_probs, 0, available_servers
        )
        server_name = server_names[server_idx]
        
        # Select GPU
        server_cache = self.keys_manager.server_caches[server_name]
        gpu_uuid = f"GPU-{server_name}-{server_cache.available_gpus}"
        
        # Cache parameters on selected server
        self.keys_manager.cache_parameters_cow(
            stage_name, model_name, parameter_size, server_name
        )
        
        # Update allocation
        server_cache.available_gpus -= 1
        self.stage_to_server[stage_name] = server_name
        self.server_to_stages[server_name].append(stage_name)
        
        print(f"✓ Scheduled {stage_name} on {server_name} -> {gpu_uuid}")
        
        return server_name, gpu_uuid
    
    def get_scheduling_stats(self) -> Dict:
        """Get scheduling statistics."""
        stats = {
            'total_servers': len(self.keys_manager.server_caches),
            'cache_hit_rates': {},
            'server_utilization': {},
            'dispersed_score': 0.0
        }
        
        for server_name, cache in self.keys_manager.server_caches.items():
            stats['cache_hit_rates'][server_name] = self.keys_manager.get_cache_hit_rate(server_name)
            stats['server_utilization'][server_name] = {
                'memory': cache.memory_utilization(),
                'gpus_used': cache.gpu_count - cache.available_gpus,
                'gpus_total': cache.gpu_count
            }
        
        # Calculate dispersion score (variance of stages per server)
        stages_per_server = [len(stages) for stages in self.server_to_stages.values()]
        stats['dispersed_score'] = 1.0 / (1.0 + np.var(stages_per_server))
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    print("Testing Cache-Aware Scheduler")
    print("=" * 60)
    
    # Mock Prometheus
    class MockPrometheus:
        def custom_query(self, query):
            return []
    
    scheduler = CacheAwareScheduler(MockPrometheus())
    
    # Initialize cluster
    servers = [
        ("server-1", 256, 4),
        ("server-2", 256, 4),
        ("server-3", 256, 4),
        ("server-4", 256, 4),
    ]
    scheduler.initialize_cluster(servers)
    
    # Schedule pipeline stages
    stages = [
        ("bert-submod-0", "bert-21b", 8.6, True),   # Critical
        ("bert-submod-1", "bert-21b", 8.6, False),
        ("bert-submod-2", "bert-21b", 8.6, True),   # Critical
        ("bert-submod-3", "bert-21b", 8.6, False),
        ("bert-submod-4", "bert-21b", 8.6, False),
    ]
    
    print("\n" + "=" * 60)
    print("Scheduling Pipeline Stages")
    print("=" * 60)
    
    for stage_name, model_name, param_size, is_critical in stages:
        print(f"\n{'🔥' if is_critical else '📦'} Scheduling {stage_name} "
              f"({'CRITICAL' if is_critical else 'normal'})")
        server, gpu = scheduler.schedule_critical_stage(
            stage_name, model_name, param_size, is_critical
        )
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Scheduling Statistics")
    print("=" * 60)
    
    stats = scheduler.get_scheduling_stats()
    print(f"\nDispersion Score: {stats['dispersed_score']:.3f}")
    print("\nServer Utilization:")
    for server, util in stats['server_utilization'].items():
        print(f"  {server}: Memory={util['memory']:.1f}%, "
              f"GPUs={util['gpus_used']}/{util['gpus_total']}")
