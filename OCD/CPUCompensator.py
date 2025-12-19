"""
Adaptive CPU Compensation for Pipeline Inference
Implements adaptive CPU allocation based on workload concentration patterns.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from kubernetes import client, config
from prometheus_api_client import PrometheusConnect
import subprocess


class CPUDemandModel:
    """
    Predictive CPU allocation model based on workload concentration.
    Uses multi-factor analysis to predict CPU requirements.
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 1.2, gamma: float = 0.3,
                 delta: float = 0.8, eta_base: float = 1.0,
                 zeta: float = 0.6, kappa: float = 0.4):
        """
        Initialize CPU demand model with empirically determined parameters.
        
        Args:
            alpha, beta: Concurrency factor parameters
            gamma: Logarithmic scaling for diminishing returns
            delta: Network overhead base factor
            eta_base: Base serialization complexity
            zeta, kappa: Memory allocation overhead factors
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.eta_base = eta_base
        self.zeta = zeta
        self.kappa = kappa
    
    def phi_concurrency(self, lambda_conc: float) -> float:
        """
        Compute concurrency overhead factor.
        
        Phi(lambda_conc) = alpha * lambda_conc^beta + gamma * log(1 + lambda_conc)
        
        Args:
            lambda_conc: Request concentration factor (new_load / old_load)
        
        Returns:
            CPU overhead from concurrency
        """
        power_term = self.alpha * (lambda_conc ** self.beta)
        log_term = self.gamma * math.log1p(lambda_conc)
        
        return power_term + log_term
    
    def psi_network(self, tensor_size: float, activation_complexity: float = 1.0) -> float:
        """
        Compute network serialization overhead.
        
        Psi(T_net) = delta * ||T||_F * eta(serialize_complexity)
        
        Args:
            tensor_size: Frobenius norm of activation tensors (GB)
            activation_complexity: Serialization complexity factor
        
        Returns:
            CPU overhead from network operations
        """
        eta = self.eta_base * activation_complexity
        overhead = self.delta * tensor_size * eta
        
        return overhead
    
    def xi_memory(self, fragmentation_index: float, gc_frequency: float) -> float:
        """
        Compute memory allocation overhead.
        
        Xi(M_alloc) = zeta * fragmentation_index + kappa * gc_frequency
        
        Args:
            fragmentation_index: Memory fragmentation level [0-1]
            gc_frequency: Garbage collection frequency (per second)
        
        Returns:
            CPU overhead from memory management
        """
        frag_term = self.zeta * fragmentation_index
        gc_term = self.kappa * gc_frequency
        
        return frag_term + gc_term
    
    def predict_cpu_requirement(self, base_cpu: float, lambda_conc: float,
                               tensor_size: float = 1.0,
                               activation_complexity: float = 1.0,
                               fragmentation_index: float = 0.3,
                               gc_frequency: float = 0.5) -> float:
        """
        Predict total CPU requirement.
        
        C_req = C_base + phi(lambda_conc) + psi(T_net) + xi(M_alloc)
        
        Args:
            base_cpu: Baseline CPU requirement
            lambda_conc: Request concentration factor
            tensor_size: Activation tensor size
            activation_complexity: Serialization complexity
            fragmentation_index: Memory fragmentation level
            gc_frequency: GC frequency
        
        Returns:
            Required CPU cores
        """
        phi = self.phi_concurrency(lambda_conc)
        psi = self.psi_network(tensor_size, activation_complexity)
        xi = self.xi_memory(fragmentation_index, gc_frequency)
        
        c_req = base_cpu + phi + psi + xi
        
        return max(base_cpu, c_req)


class CPUCompensator:
    """
    Adaptive CPU resource orchestration for concentrated workloads.
    Dynamically allocates CPU resources based on workload patterns.
    """
    
    def __init__(self, prometheus_client: PrometheusConnect,
                 min_cpu: float = 1.0, max_cpu: float = 16.0,
                 increment_ratio: float = 0.25,
                 performance_threshold: float = 0.05,
                 adaptation_interval: float = 10.0):
        """
        Initialize CPU compensator.
        
        Args:
            prometheus_client: Prometheus client for metrics
            min_cpu: Minimum CPU cores per stage
            max_cpu: Maximum CPU cores per stage
            increment_ratio: Incremental increase ratio (1/4 in algorithm)
            performance_threshold: Minimum performance improvement to continue
            adaptation_interval: Time to wait between adjustments (seconds)
        """
        self.prometheus = prometheus_client
        self.min_cpu = min_cpu
        self.max_cpu = max_cpu
        self.increment_ratio = increment_ratio
        self.performance_threshold = performance_threshold
        self.adaptation_interval = adaptation_interval
        
        # CPU demand model
        self.demand_model = CPUDemandModel()
        
        # Load Kubernetes config
        try:
            config.load_kube_config()
        except:
            config.load_incluster_config()
        
        self.kube_client = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()
    
    def get_stage_cpu_metrics(self, stage_name: str, namespace: str = "cdgp") -> Dict[str, float]:
        """
        Get CPU-related metrics for a pipeline stage.
        
        Args:
            stage_name: Pipeline stage deployment name
            namespace: Kubernetes namespace
        
        Returns:
            Dictionary with CPU metrics
        """
        metrics = {}
        
        # Current CPU allocation
        try:
            deployment = self.kube_client.read_namespaced_deployment(stage_name, namespace)
            cpu_limit = deployment.spec.template.spec.containers[0].resources.limits.get('cpu', '1')
            if isinstance(cpu_limit, str):
                if cpu_limit.endswith('m'):
                    metrics['cpu_allocated'] = float(cpu_limit[:-1]) / 1000.0
                else:
                    metrics['cpu_allocated'] = float(cpu_limit)
            else:
                metrics['cpu_allocated'] = float(cpu_limit)
        except Exception as e:
            print(f"Error reading CPU allocation: {e}")
            metrics['cpu_allocated'] = self.min_cpu
        
        # CPU utilization
        cpu_query = f'sum(rate(container_cpu_usage_seconds_total{{pod=~"{stage_name}.*",namespace="{namespace}"}}[1m]))'
        cpu_result = self.prometheus.custom_query(query=cpu_query)
        metrics['cpu_usage'] = float(cpu_result[0]['value'][1]) if cpu_result else 0.0
        
        # CPU utilization percentage
        if metrics['cpu_allocated'] > 0:
            metrics['cpu_util_percent'] = (metrics['cpu_usage'] / metrics['cpu_allocated']) * 100
        else:
            metrics['cpu_util_percent'] = 0.0
        
        # Request rate for concentration calculation
        rps_query = f'rate(gateway_function_invocation_started{{function_name="{stage_name}.{namespace}"}}[1m])'
        rps_result = self.prometheus.custom_query(query=rps_query)
        metrics['request_rate'] = float(rps_result[0]['value'][1]) if rps_result else 0.0
        
        # Latency as performance indicator
        latency_query = f'rate(gateway_functions_seconds_sum{{function_name="{stage_name}.{namespace}"}}[1m]) / rate(gateway_functions_seconds_count{{function_name="{stage_name}.{namespace}"}}[1m])'
        latency_result = self.prometheus.custom_query(query=latency_query)
        metrics['latency'] = float(latency_result[0]['value'][1]) if latency_result else 0.0
        
        return metrics
    
    def calculate_concentration_factor(self, prev_request_rate: float,
                                       new_request_rate: float,
                                       prev_replicas: int,
                                       new_replicas: int) -> float:
        """
        Calculate request concentration factor after replica reduction.
        
        Args:
            prev_request_rate: Previous total request rate
            new_request_rate: Current total request rate
            prev_replicas: Previous number of replicas
            new_replicas: Current number of replicas
        
        Returns:
            Concentration factor lambda_conc
        """
        if prev_replicas == 0 or new_replicas == 0:
            return 1.0
        
        # Per-replica load concentration
        prev_load_per_replica = prev_request_rate / prev_replicas if prev_replicas > 0 else 0
        new_load_per_replica = new_request_rate / new_replicas if new_replicas > 0 else 0
        
        if prev_load_per_replica == 0:
            return 1.0
        
        lambda_conc = new_load_per_replica / prev_load_per_replica
        
        return max(1.0, lambda_conc)
    
    def allocate_cpu_resources(self, stage_name: str, namespace: str,
                              new_cpu_cores: float) -> bool:
        """
        Allocate CPU resources using Kubernetes API and cgroup controls.
        
        Args:
            stage_name: Pipeline stage deployment name
            namespace: Kubernetes namespace
            new_cpu_cores: New CPU core allocation
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read current deployment
            deployment = self.kube_client.read_namespaced_deployment(stage_name, namespace)
            
            # Update CPU limits and requests
            cpu_limit = f"{int(new_cpu_cores * 1000)}m"  # Convert to millicores
            
            container = deployment.spec.template.spec.containers[0]
            if container.resources is None:
                container.resources = client.V1ResourceRequirements()
            
            if container.resources.limits is None:
                container.resources.limits = {}
            if container.resources.requests is None:
                container.resources.requests = {}
            
            container.resources.limits['cpu'] = cpu_limit
            container.resources.requests['cpu'] = cpu_limit
            
            # Patch deployment
            self.kube_client.patch_namespaced_deployment(
                name=stage_name,
                namespace=namespace,
                body=deployment
            )
            
            print(f"Updated CPU allocation for {stage_name}: {new_cpu_cores} cores")
            return True
            
        except Exception as e:
            print(f"Error allocating CPU resources: {e}")
            return False
    
    def adaptive_cpu_allocation(self, stage_name: str, namespace: str,
                               prev_replicas: int, new_replicas: int,
                               prev_request_rate: float) -> float:
        """
        Adaptive CPU allocation algorithm.
        
        Args:
            stage_name: Pipeline stage name
            namespace: Kubernetes namespace
            prev_replicas: Previous replica count
            new_replicas: New replica count (after smoothing)
            prev_request_rate: Previous request rate
        
        Returns:
            Final allocated CPU cores
        """
        print(f"\n{'='*60}")
        print(f"Adaptive CPU Allocation for {stage_name}")
        print(f"{'='*60}")
        
        # Get current metrics
        metrics = self.get_stage_cpu_metrics(stage_name, namespace)
        current_cpu = metrics['cpu_allocated']
        current_latency = metrics['latency']
        new_request_rate = metrics['request_rate']
        
        print(f"Current: {current_cpu:.2f} cores, Latency: {current_latency:.3f}s")
        print(f"Replicas: {prev_replicas} -> {new_replicas}")
        print(f"Request rate: {prev_request_rate:.2f} -> {new_request_rate:.2f} req/s")
        
        # Calculate concentration factor
        lambda_conc = self.calculate_concentration_factor(
            prev_request_rate, new_request_rate, prev_replicas, new_replicas
        )
        
        print(f"Concentration factor: {lambda_conc:.2f}x")
        
        # Predict required CPU
        predicted_cpu = self.demand_model.predict_cpu_requirement(
            base_cpu=current_cpu,
            lambda_conc=lambda_conc,
            tensor_size=1.0,  # Can be estimated from model size
            activation_complexity=1.0,
            fragmentation_index=0.3,
            gc_frequency=0.5
        )
        
        print(f"Predicted CPU requirement: {predicted_cpu:.2f} cores")
        
        # Incremental allocation with performance monitoring
        c_current = current_cpu
        c_increment = max(0.5, (predicted_cpu - c_current) * self.increment_ratio)
        
        iteration = 0
        max_iterations = 5
        
        while c_increment > 0.1 and iteration < max_iterations:
            # Check if we've reached max CPU limit
            if c_current >= self.max_cpu:
                print(f"Reached maximum CPU limit: {self.max_cpu} cores")
                print("⚠️ SLO may be violated - consider increasing GPU replicas")
                return c_current
            
            # Proposed new allocation
            c_proposed = min(c_current + c_increment, self.max_cpu)
            
            print(f"\nIteration {iteration + 1}: Testing {c_proposed:.2f} cores...")
            
            # Apply CPU allocation
            success = self.allocate_cpu_resources(stage_name, namespace, c_proposed)
            
            if not success:
                print("Failed to allocate CPU resources")
                break
            
            # Wait for adaptation
            import time
            time.sleep(self.adaptation_interval)
            
            # Check performance improvement
            new_metrics = self.get_stage_cpu_metrics(stage_name, namespace)
            new_latency = new_metrics['latency']
            
            if current_latency > 0:
                improvement = (current_latency - new_latency) / current_latency
            else:
                improvement = 0.0
            
            print(f"Latency: {current_latency:.3f}s -> {new_latency:.3f}s "
                  f"(improvement: {improvement*100:.1f}%)")
            
            if improvement < self.performance_threshold:
                print("Diminishing returns detected - stopping allocation")
                break
            
            # Update for next iteration
            c_current = c_proposed
            current_latency = new_latency
            c_increment *= 0.5  # Reduce increment size
            iteration += 1
        
        print(f"\nFinal CPU allocation: {c_current:.2f} cores")
        return c_current
    
    def compensate_after_smoothing(self, stage_replicas_changes: Dict[str, Tuple[int, int]],
                                   stage_request_rates: Dict[str, float],
                                   namespace: str = "cdgp") -> Dict[str, float]:
        """
        Apply CPU compensation after replica smoothing.
        
        Args:
            stage_replicas_changes: Dict mapping stage -> (old_replicas, new_replicas)
            stage_request_rates: Dict mapping stage -> request_rate
            namespace: Kubernetes namespace
        
        Returns:
            Dict mapping stage names to allocated CPU cores
        """
        cpu_allocations = {}
        
        print("\n" + "="*60)
        print("CPU Compensation Phase")
        print("="*60)
        
        for stage_name, (old_replicas, new_replicas) in stage_replicas_changes.items():
            # Only compensate if replicas were reduced
            if new_replicas < old_replicas:
                print(f"\nStage {stage_name}: Replicas reduced {old_replicas} -> {new_replicas}")
                
                prev_rate = stage_request_rates.get(stage_name, 0.0)
                
                allocated_cpu = self.adaptive_cpu_allocation(
                    stage_name=stage_name,
                    namespace=namespace,
                    prev_replicas=old_replicas,
                    new_replicas=new_replicas,
                    prev_request_rate=prev_rate
                )
                
                cpu_allocations[stage_name] = allocated_cpu
            else:
                print(f"\nStage {stage_name}: No compensation needed (replicas increased or unchanged)")
                metrics = self.get_stage_cpu_metrics(stage_name, namespace)
                cpu_allocations[stage_name] = metrics['cpu_allocated']
        
        return cpu_allocations


# Example usage and testing
if __name__ == "__main__":
    print("Testing CPU Demand Model")
    print("=" * 60)
    
    model = CPUDemandModel()
    
    # Test different concentration factors
    base_cpu = 2.0
    print(f"Base CPU: {base_cpu} cores\n")
    
    for conc_factor in [1.0, 2.0, 4.0, 8.0]:
        required_cpu = model.predict_cpu_requirement(
            base_cpu=base_cpu,
            lambda_conc=conc_factor,
            tensor_size=1.0,
            activation_complexity=1.0,
            fragmentation_index=0.3,
            gc_frequency=0.5
        )
        
        print(f"Concentration {conc_factor}x: {required_cpu:.2f} cores "
              f"(+{required_cpu - base_cpu:.2f})")
    
    print("\n" + "=" * 60)
    print("Component Analysis")
    print("=" * 60)
    
    lambda_conc = 4.0
    phi = model.phi_concurrency(lambda_conc)
    psi = model.psi_network(1.0, 1.0)
    xi = model.xi_memory(0.3, 0.5)
    
    print(f"Concurrency overhead (phi): {phi:.3f}")
    print(f"Network overhead (psi): {psi:.3f}")
    print(f"Memory overhead (xi): {xi:.3f}")
    print(f"Total overhead: {phi + psi + xi:.3f}")
    print(f"Total requirement: {base_cpu + phi + psi + xi:.3f} cores")
