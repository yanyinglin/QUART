#!/usr/bin/env python3
"""
Quart Integrated Scaler
"""

import time
import asyncio
import pandas as pd
import subprocess
import math
import os
from typing import Dict, List, Optional
from prometheus_api_client import PrometheusConnect
from kubernetes import client, config
from kubernetes.client.rest import ApiException

# Import Quart components
from ReplicaCorrector import ReplicaCorrector
from PipelineSmoother import PipelineSmoother
from CPUCompensator import CPUCompensator

# Load the Kubernetes configuration
config.load_kube_config()

# Initialize the Kubernetes API client
kube_client = client.AppsV1Api()
autoscaling_client = client.AutoscalingV1Api()
v1 = client.CoreV1Api()

# Constants
TARGET_LATENCY = 0.5  # Target latency in seconds
MAX_REPLICAS = 8  # Maximum number of replicas for each deployment
MIN_REPLICAS = 1   # Minimum number of replicas
SCALE_COOLDOWN = 2 * 60  # 2 minutes cooldown between scales
DEFAULT_NUM_REPLICAS = 1

# Model execution times (from original DaShengScaler)
MODEL_EXEC_TIME = {
    'BERT': 0.127,
    'GPT': 0.095*3,
    'LLAMA': 0.7,
    'WIDERESNET': 0.649 + 0.6,
    'WHISPER': 0.073,
    'LLAMA2KK70': 1.5
}

scale_records = {}


class QuartScaler:
    """
    Integrated scaler using Quart's advanced components:
    - ReplicaCorrector for PID-based replica adjustment
    - PipelineSmoother for CV-based resource optimization
    - CPUCompensator for adaptive CPU allocation
    """

    def __init__(self, prometheus_url: str = "http://172.169.8.253:31113/",
                 namespace: str = "cdgp", target_latency: float = TARGET_LATENCY):
        """
        Initialize the Quart scaler

        Args:
            prometheus_url: URL of Prometheus server
            namespace: Kubernetes namespace to monitor
            target_latency: Target latency for replica correction
        """
        self.prometheus_client = PrometheusConnect(url=prometheus_url, disable_ssl=False)
        self.namespace = namespace
        self.target_latency = target_latency

        # Initialize Quart components
        self.replica_corrector = ReplicaCorrector(self.prometheus_client, target_latency=target_latency)
        self.pipeline_smoother = PipelineSmoother(self.prometheus_client)
        self.cpu_compensator = CPUCompensator(self.prometheus_client)

        print("QuartScaler initialized with integrated components:")
        print("- ReplicaCorrector (PID-based)")
        print("- PipelineSmoother (CV-based)")
        print("- CPUCompensator (Adaptive)")

    async def scale_pipeline(self, pipeline_stages: List[str], model_name: str) -> Dict[str, int]:
        """
        Scale a complete pipeline using Quart's integrated approach

        Args:
            pipeline_stages: List of pipeline stage names
            model_name: Name of the model being served

        Returns:
            Dictionary mapping stage names to recommended replica counts
        """
        print(f"\n=== Scaling pipeline: {model_name} ===")
        print(f"Stages: {pipeline_stages}")

        try:
            # Step 1: Replica Correction using PID control
            print("\n1. Applying Replica Correction (PID)...")
            corrected_replicas = self.replica_corrector.correct_pipeline_replicas(
                pipeline_stages, self.namespace
            )

            # Step 2: Pipeline Smoothing for resource optimization
            print("\n2. Applying Pipeline Smoothing (CV-based)...")
            smoothed_replicas = self.pipeline_smoother.apply_pipeline_smoothing(
                corrected_replicas, pipeline_stages, self.namespace
            )

            # Step 3: Calculate replica changes for CPU compensation
            replica_changes = {}
            stage_request_rates = {}

            for stage in pipeline_stages:
                # Get current replicas (mock implementation - in real scenario would query k8s)
                current_replicas = await self._get_current_replicas(stage)
                replica_changes[stage] = (current_replicas, smoothed_replicas[stage])

                # Get request rates for CPU compensation
                stage_request_rates[stage] = await self._get_stage_request_rate(stage)

            # Step 4: CPU Compensation for concentrated workloads
            print("\n3. Applying CPU Compensation...")
            cpu_compensations = self.cpu_compensator.compensate_after_smoothing(
                replica_changes, stage_request_rates
            )

            print(f"\n4. Final scaling decisions:")
            for stage in pipeline_stages:
                old_replicas = replica_changes[stage][0]
                new_replicas = smoothed_replicas[stage]
                cpu_boost = cpu_compensations.get(stage, 0)
                print(f"  {stage}: {old_replicas} -> {new_replicas} replicas, CPU boost: {cpu_boost}")

            return smoothed_replicas

        except Exception as e:
            print(f"Error in pipeline scaling: {e}")
            # Fallback to basic scaling
            return await self._fallback_scaling(pipeline_stages, model_name)

    async def _get_current_replicas(self, stage_name: str) -> int:
        """Get current replica count for a stage"""
        try:
            deployment = kube_client.read_namespaced_deployment(stage_name, self.namespace)
            return deployment.spec.replicas
        except ApiException:
            return 1  # Default to 1 if can't determine

    async def _get_stage_request_rate(self, stage_name: str) -> float:
        """Get request rate for a stage"""
        try:
            # Query Prometheus for request rate
            query = f'sum(irate(gateway_function_invocation_started{{function_name="{stage_name}.{self.namespace}"}}[1m]))'
            result = self.prometheus_client.custom_query(query=query)

            if result and result[0]['value']:
                return float(result[0]['value'][1])
            return 0.0
        except Exception:
            return 0.0

    async def _fallback_scaling(self, pipeline_stages: List[str], model_name: str) -> Dict[str, int]:
        """Fallback scaling using basic RPS-based approach"""
        print("Using fallback scaling approach...")

        scaling_decisions = {}
        for stage in pipeline_stages:
            request_rate = await self._get_stage_request_rate(stage)
            if request_rate > 0:
                # Simple RPS-based scaling
                model_exec_time = MODEL_EXEC_TIME.get(model_name, 0.1)
                desired_replicas = math.ceil(request_rate * model_exec_time)
                desired_replicas = max(MIN_REPLICAS, min(MAX_REPLICAS, desired_replicas))
            else:
                desired_replicas = DEFAULT_NUM_REPLICAS

            scaling_decisions[stage] = desired_replicas

        return scaling_decisions

    async def apply_scaling_decisions(self, scaling_decisions: Dict[str, int]) -> None:
        """
        Apply the scaling decisions to Kubernetes deployments

        Args:
            scaling_decisions: Dictionary mapping stage names to replica counts
        """
        print("\n=== Applying Scaling Decisions ===")

        for stage_name, desired_replicas in scaling_decisions.items():
            try:
                # Check cooldown period
                last_scale_time = scale_records.get(stage_name, 0)
                if time.time() - last_scale_time < SCALE_COOLDOWN:
                    print(f"Skipping {stage_name} due to cooldown")
                    continue

                # Get current replicas
                current_replicas = await self._get_current_replicas(stage_name)

                if current_replicas != desired_replicas:
                    # Apply scaling
                    await scale_deployment(stage_name, self.namespace, desired_replicas)
                    scale_records[stage_name] = time.time()
                    print(f"Scaled {stage_name}: {current_replicas} -> {desired_replicas}")
                else:
                    print(f"No scaling needed for {stage_name} (already at {desired_replicas})")

            except Exception as e:
                print(f"Error scaling {stage_name}: {e}")

    async def scale_all_pipelines(self) -> None:
        """Scale all pipelines in the namespace"""
        try:
            # Get all deployments
            deployments = kube_client.list_namespaced_deployment(self.namespace)

            # Group by pipeline (model)
            pipeline_groups = {}
            for deployment in deployments.items:
                deployment_name = deployment.metadata.name

                # Extract model name (assuming format: model-submod-X-...)
                if '-submod-' in deployment_name:
                    model_name = deployment_name.split('-submod-')[0]
                    if model_name not in pipeline_groups:
                        pipeline_groups[model_name] = []
                    pipeline_groups[model_name].append(deployment_name)

            # Scale each pipeline
            for model_name, stages in pipeline_groups.items():
                # Sort stages to ensure proper order
                stages.sort(key=lambda x: int(x.split('-submod-')[1].split('-')[0]))

                scaling_decisions = await self.scale_pipeline(stages, model_name.upper())
                await self.apply_scaling_decisions(scaling_decisions)

        except Exception as e:
            print(f"Error in scale_all_pipelines: {e}")


async def scale_deployment(deployment_name: str, namespace: str, num_replicas: int) -> None:
    """
    Scale a Kubernetes deployment to the specified number of replicas

    Args:
        deployment_name: Name of the deployment to scale
        namespace: Kubernetes namespace
        num_replicas: Desired number of replicas
    """
    try:
        # Use kubectl for scaling (could also use Kubernetes API directly)
        cmd = ["kubectl", "scale", "deployment", deployment_name,
               "--replicas", str(num_replicas), "--namespace", namespace]

        subprocess.check_output(cmd)
        print(f"Successfully scaled {deployment_name} to {num_replicas} replicas")

    except subprocess.CalledProcessError as e:
        print(f"Failed to scale deployment {deployment_name}: {e}")
    except Exception as e:
        print(f"Error scaling deployment {deployment_name}: {e}")


async def main():
    """Main scaling loop"""
    # Initialize Quart scaler
    scaler = QuartScaler()

    print("Starting Quart Integrated Scaler...")
    print(f"Target latency: {scaler.target_latency}s")
    print(f"Namespace: {scaler.namespace}")
    print("Scaling interval: 30 seconds")

    while True:
        try:
            await scaler.scale_all_pipelines()
            print(f"\nWaiting 30 seconds until next scaling cycle...")
            await asyncio.sleep(30)

        except KeyboardInterrupt:
            print("\nShutting down Quart Scaler...")
            break
        except Exception as e:
            print(f"Error in main scaling loop: {e}")
            await asyncio.sleep(30)


if __name__ == "__main__":
    # Run the integrated scaler
    asyncio.run(main())