"""
Pipeline-Aware Replica Correction with PID Control
Implements PID-based replica correction using M/M/c queuing theory.
"""

import math
import time
from typing import Dict, List, Tuple
import numpy as np
from prometheus_api_client import PrometheusConnect


class MMCQueue:
    """
    M/M/c queuing model for pipeline stage analysis.
    Uses queuing theory to predict stage delays.
    """
    
    @staticmethod
    def factorial(n: int) -> int:
        """Compute factorial for queuing calculations."""
        if n <= 1:
            return 1
        return math.factorial(n)
    
    @staticmethod
    def compute_p0(rho: float, c: int) -> float:
        """
        Compute probability of empty system P_0.
        
        Args:
            rho: Traffic intensity (lambda/mu)
            c: Number of replicas (servers)
        
        Returns:
            P_0: Probability of empty system
        """
        if rho >= c:
            # System is unstable
            return 0.0
        
        # Calculate sum for normalization
        sum_term = sum([(rho ** n) / MMCQueue.factorial(n) for n in range(c)])
        last_term = (rho ** c) / (MMCQueue.factorial(c) * (1 - rho / c))
        
        p0 = 1.0 / (sum_term + last_term)
        return p0
    
    @staticmethod
    def compute_queue_delay(arrival_rate: float, service_rate: float, 
                           num_replicas: int) -> float:
        """
        Compute expected queuing delay using M/M/c model.
        
        Args:
            arrival_rate: Request arrival rate (lambda)
            service_rate: Service rate per replica (mu)
            num_replicas: Number of replicas (c)
        
        Returns:
            Expected queuing delay T_i in seconds
        """
        if num_replicas < 1:
            return float('inf')
        
        # Traffic intensity
        rho = arrival_rate / service_rate
        
        # Check stability condition
        if rho >= num_replicas:
            return float('inf')
        
        # Compute P_0
        p0 = MMCQueue.compute_p0(rho, num_replicas)
        
        # Queue delay (waiting time in queue)
        numerator = p0 * arrival_rate * (rho ** num_replicas)
        denominator = MMCQueue.factorial(num_replicas) * ((1 - rho / num_replicas) ** 2)
        
        queue_time = numerator / (denominator * service_rate * num_replicas)
        
        # Total delay including service time
        service_time = 1.0 / service_rate
        total_delay = queue_time + service_time
        
        return total_delay


class PIDController:
    """
    PID Controller for dynamic replica adjustment.
    Uses proportional-integral-derivative control for feedback-based scaling.
    """
    
    def __init__(self, kp: float = 1.0, ki: float = 0.1, kd: float = 0.05,
                 min_replicas: int = 1, max_replicas: int = 10):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gain (K_p)
            ki: Integral gain (K_i)
            kd: Derivative gain (K_d)
            min_replicas: Minimum number of replicas
            max_replicas: Maximum number of replicas
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        
        # PID state
        self.integral = 0.0
        self.previous_error = 0.0
        self.last_update_time = time.time()
    
    def compute_control_signal(self, target_latency: float, 
                               current_latency: float) -> float:
        """
        Compute PID control signal for replica adjustment.
        
        Args:
            target_latency: Target latency (T_target)
            current_latency: Current observed latency (T_i)
        
        Returns:
            Delta_r: Replica adjustment signal
        """
        current_time = time.time()
        dt = current_time - self.last_update_time
        
        if dt <= 0:
            dt = 1.0  # Prevent division by zero
        
        # Error signal: e(t) = T_target - T_i(t)
        error = target_latency - current_latency
        
        # Proportional term: K_p * e(t)
        p_term = self.kp * error
        
        # Integral term: K_i * integral(e(t))
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term: K_d * de(t)/dt
        derivative = (error - self.previous_error) / dt
        d_term = self.kd * derivative
        
        # Total control signal
        control_signal = p_term + i_term + d_term
        
        # Update state
        self.previous_error = error
        self.last_update_time = current_time
        
        return control_signal
    
    def adjust_replicas(self, current_replicas: int, target_latency: float,
                       current_latency: float) -> int:
        """
        Adjust replica count using PID control.
        
        Args:
            current_replicas: Current number of replicas (r_i(t))
            target_latency: Target latency in seconds
            current_latency: Current observed latency in seconds
        
        Returns:
            New replica count (r_i'(t))
        """
        # Compute control signal
        delta_r = self.compute_control_signal(target_latency, current_latency)
        
        # Apply adjustment: r_i'(t) = r_i(t) + Delta_r(t)
        new_replicas = current_replicas + delta_r
        
        # Clamp to valid range
        new_replicas = max(self.min_replicas, min(self.max_replicas, new_replicas))
        
        return int(round(new_replicas))
    
    def reset(self):
        """Reset PID controller state."""
        self.integral = 0.0
        self.previous_error = 0.0
        self.last_update_time = time.time()


class ReplicaCorrector:
    """
    Pipeline-aware replica correction system.
    Identifies critical stages and adjusts replicas using PID control.
    """
    
    def __init__(self, prometheus_client: PrometheusConnect,
                 target_latency: float = 0.5,
                 kp: float = 2.0, ki: float = 0.3, kd: float = 0.1,
                 min_replicas: int = 1, max_replicas: int = 8):
        """
        Initialize replica corrector.
        
        Args:
            prometheus_client: Prometheus client for metrics
            target_latency: Target latency for SLO (seconds)
            kp, ki, kd: PID controller gains
            min_replicas, max_replicas: Replica bounds
        """
        self.prometheus = prometheus_client
        self.target_latency = target_latency
        
        # PID controllers for each stage (lazily initialized)
        self.pid_controllers: Dict[str, PIDController] = {}
        
        # PID parameters
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        
        # Stage performance history
        self.stage_metrics: Dict[str, List[Dict]] = {}
    
    def get_or_create_pid(self, stage_name: str) -> PIDController:
        """Get or create PID controller for a stage."""
        if stage_name not in self.pid_controllers:
            self.pid_controllers[stage_name] = PIDController(
                kp=self.kp, ki=self.ki, kd=self.kd,
                min_replicas=self.min_replicas,
                max_replicas=self.max_replicas
            )
        return self.pid_controllers[stage_name]
    
    def get_stage_metrics(self, stage_name: str, namespace: str = "cdgp") -> Tuple[float, float, int]:
        """
        Get current metrics for a pipeline stage.
        
        Args:
            stage_name: Name of the pipeline stage
            namespace: Kubernetes namespace
        
        Returns:
            (arrival_rate, queue_latency, current_replicas)
        """
        # Query arrival rate (RPS)
        rps_query = f'sum(irate(gateway_function_invocation_started{{function_name="{stage_name}.{namespace}"}}[1m]))'
        rps_result = self.prometheus.custom_query(query=rps_query)
        
        arrival_rate = 0.0
        if rps_result and len(rps_result) > 0:
            arrival_rate = float(rps_result[0]['value'][1])
        
        # Query latency (average response time)
        latency_query = f'rate(gateway_functions_seconds_sum{{function_name="{stage_name}.{namespace}"}}[1m]) / rate(gateway_functions_seconds_count{{function_name="{stage_name}.{namespace}"}}[1m])'
        latency_result = self.prometheus.custom_query(query=latency_query)
        
        queue_latency = 0.0
        if latency_result and len(latency_result) > 0:
            queue_latency = float(latency_result[0]['value'][1])
        
        # Query current replicas (from Kubernetes)
        replicas_query = f'kube_deployment_spec_replicas{{deployment="{stage_name}",namespace="{namespace}"}}'
        replicas_result = self.prometheus.custom_query(query=replicas_query)
        
        current_replicas = self.min_replicas
        if replicas_result and len(replicas_result) > 0:
            current_replicas = int(float(replicas_result[0]['value'][1]))
        
        return arrival_rate, queue_latency, current_replicas
    
    def identify_critical_stages(self, stages: List[str], namespace: str = "cdgp") -> List[Tuple[str, float]]:
        """
        Identify critical stages based on queue depth and latency.
        
        Args:
            stages: List of pipeline stage names
            namespace: Kubernetes namespace
        
        Returns:
            List of (stage_name, criticality_score) sorted by criticality
        """
        critical_stages = []
        
        for stage in stages:
            arrival_rate, queue_latency, current_replicas = self.get_stage_metrics(stage, namespace)
            
            # Criticality score: combination of latency violation and queue depth
            latency_violation = max(0, queue_latency - self.target_latency)
            relative_violation = latency_violation / self.target_latency if self.target_latency > 0 else 0
            
            # Higher score = more critical
            criticality_score = relative_violation * (1 + arrival_rate / 10.0)
            
            if criticality_score > 0.1:  # Threshold for considering a stage critical
                critical_stages.append((stage, criticality_score))
        
        # Sort by criticality (highest first)
        critical_stages.sort(key=lambda x: x[1], reverse=True)
        
        return critical_stages
    
    def correct_stage_replicas(self, stage_name: str, namespace: str = "cdgp",
                               service_rate: float = None) -> Tuple[int, int]:
        """
        Correct replica count for a single stage using PID control.
        
        Args:
            stage_name: Name of the pipeline stage
            namespace: Kubernetes namespace
            service_rate: Service rate (mu) if known, otherwise estimated
        
        Returns:
            (current_replicas, recommended_replicas)
        """
        # Get current metrics
        arrival_rate, queue_latency, current_replicas = self.get_stage_metrics(stage_name, namespace)
        
        if arrival_rate == 0:
            # No traffic, keep minimum replicas
            return current_replicas, self.min_replicas
        
        # Estimate service rate if not provided
        if service_rate is None:
            # Estimate from current performance: mu ≈ arrival_rate / (replicas * utilization)
            # Assume 70% utilization for estimation
            estimated_utilization = 0.7
            service_rate = arrival_rate / (current_replicas * estimated_utilization) if current_replicas > 0 else 1.0
        
        # Use queuing theory to predict latency with current replicas
        predicted_latency = MMCQueue.compute_queue_delay(arrival_rate, service_rate, current_replicas)
        
        # Use actual observed latency if available, otherwise use prediction
        observed_latency = queue_latency if queue_latency > 0 else predicted_latency
        
        # Get PID controller for this stage
        pid = self.get_or_create_pid(stage_name)
        
        # Compute recommended replicas using PID control
        recommended_replicas = pid.adjust_replicas(
            current_replicas=current_replicas,
            target_latency=self.target_latency,
            current_latency=observed_latency
        )
        
        # Store metrics history
        if stage_name not in self.stage_metrics:
            self.stage_metrics[stage_name] = []
        
        self.stage_metrics[stage_name].append({
            'timestamp': time.time(),
            'arrival_rate': arrival_rate,
            'observed_latency': observed_latency,
            'predicted_latency': predicted_latency,
            'current_replicas': current_replicas,
            'recommended_replicas': recommended_replicas
        })
        
        # Keep only recent history (last 100 measurements)
        if len(self.stage_metrics[stage_name]) > 100:
            self.stage_metrics[stage_name] = self.stage_metrics[stage_name][-100:]
        
        return current_replicas, recommended_replicas
    
    def correct_pipeline_replicas(self, pipeline_stages: List[str], 
                                  namespace: str = "cdgp") -> Dict[str, int]:
        """
        Correct replicas for all stages in a pipeline.
        
        Args:
            pipeline_stages: List of stage names in order
            namespace: Kubernetes namespace
        
        Returns:
            Dictionary mapping stage names to recommended replica counts
        """
        recommendations = {}
        
        # First, identify critical stages
        critical_stages = self.identify_critical_stages(pipeline_stages, namespace)
        
        print(f"Critical stages detected: {len(critical_stages)}")
        for stage_name, criticality in critical_stages:
            print(f"  - {stage_name}: criticality={criticality:.3f}")
        
        # Correct replicas for each stage
        for stage_name in pipeline_stages:
            current_replicas, recommended_replicas = self.correct_stage_replicas(
                stage_name, namespace
            )
            
            recommendations[stage_name] = recommended_replicas
            
            if current_replicas != recommended_replicas:
                print(f"Stage {stage_name}: {current_replicas} -> {recommended_replicas} replicas")
        
        return recommendations


# Example usage and testing
if __name__ == "__main__":
    # Test queuing model
    print("Testing M/M/c Queuing Model:")
    print("-" * 50)
    
    arrival_rate = 10.0  # 10 requests/second
    service_rate = 3.0   # 3 requests/second per replica
    
    for num_replicas in range(1, 8):
        delay = MMCQueue.compute_queue_delay(arrival_rate, service_rate, num_replicas)
        if delay == float('inf'):
            print(f"Replicas: {num_replicas}, Delay: UNSTABLE (rho/c >= 1)")
        else:
            print(f"Replicas: {num_replicas}, Delay: {delay:.3f}s")
    
    print("\n" + "=" * 50)
    print("Testing PID Controller:")
    print("-" * 50)
    
    pid = PIDController(kp=2.0, ki=0.3, kd=0.1, min_replicas=1, max_replicas=8)
    target_latency = 0.5  # 500ms target
    
    # Simulate scenario
    current_replicas = 3
    for i in range(10):
        # Simulate varying latency
        simulated_latency = 0.8 + 0.2 * math.sin(i * 0.5)  # Oscillating latency
        
        new_replicas = pid.adjust_replicas(current_replicas, target_latency, simulated_latency)
        
        print(f"Step {i}: Latency={simulated_latency:.3f}s, Replicas: {current_replicas} -> {new_replicas}")
        
        current_replicas = new_replicas
        time.sleep(0.1)
