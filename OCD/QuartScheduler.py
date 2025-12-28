#!/usr/bin/env python3
"""
Quart Integrated Scheduler
"""

import os
import sys
import time
import json
import ssl
import base64
import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Tuple, Optional
from flask import Flask, request
from kubernetes import client, config
from prometheus_api_client import PrometheusConnect
from datetime import datetime

# Import Quart components
from CacheAwareScheduler import CacheAwareScheduler
from ReplicaCorrector import ReplicaCorrector
from PipelineSmoother import PipelineSmoother
from CPUCompensator import CPUCompensator

# Import existing utilities
from perfering import GPUMetric
from Kuberinit import KubernetesInstance

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Setup SSL context for webhook
context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)

# Initialize Kubernetes and Prometheus clients
try:
    config.load_kube_config()
    api = client.CoreV1Api()
    v1 = client.AppsV1Api()
    k8s_available = True
except Exception as e:
    logging.warning(f"Kubernetes not available: {e}")
    api = None
    v1 = None
    k8s_available = False

# Configuration
CLUSTER_CONFIG = {
    "c108": {
        "master_ip": "172.169.8.253",
        "kubeconfig_path": "config_c1",
        "prometheus_port": 31113,
        "price": 0.019
    }
}

# Initialize Prometheus
try:
    prometheus = PrometheusConnect(url="http://172.169.8.253:30500")
    prometheus_available = True
except Exception as e:
    logging.warning(f"Prometheus not available: {e}")
    prometheus = None
    prometheus_available = False

# Initialize Kubernetes instance
try:
    kinstance = KubernetesInstance(
        CLUSTER_CONFIG['c108']['kubeconfig_path'],
        CLUSTER_CONFIG['c108']['master_ip']
    )
    kinstance_available = True
except Exception as e:
    logging.warning(f"Kubernetes instance not available: {e}")
    kinstance = None
    kinstance_available = False

# Initialize Quart components
try:
    cache_scheduler = CacheAwareScheduler(prometheus)
    cache_available = True
except Exception as e:
    logging.warning(f"CacheAwareScheduler not available: {e}")
    cache_scheduler = None
    cache_available = False

try:
    replica_corrector = ReplicaCorrector(prometheus, target_latency=0.5)
    replica_available = True
except Exception as e:
    logging.warning(f"ReplicaCorrector not available: {e}")
    replica_corrector = None
    replica_available = False

try:
    pipeline_smoother = PipelineSmoother(prometheus)
    smoother_available = True
except Exception as e:
    logging.warning(f"PipelineSmoother not available: {e}")
    pipeline_smoother = None
    smoother_available = False

try:
    cpu_compensator = CPUCompensator(prometheus)
    cpu_available = True
except Exception as e:
    logging.warning(f"CPUCompensator not available: {e}")
    cpu_compensator = None
    cpu_available = False

SCHEDULER_NAME = "quart_scheduler"
NAMESPACE = "dasheng"
GPU_QUOTA = 1.15
RESERVED_CPU = 0.3

# Node configuration
NODE_LIST = {
    "3090": ["cc225", "cc226", "cc231", "cc230"],
    "A100": ["cc234", "cc233", "cc245"],
    "V100": ["cc221", "cc222", "cc223", "cc224", "cc229", "cc230", "cc227", "cc228"],
    "A40": ["cc238", "cc239", "cc240", "cc241", "cc242"],
}

MEMORY_UNITS = {
    'K': 1024 ** 2,
    'M': 1024 ** 1,
    'G': 1,
}

# Model configurations
MODEL_CONFIGS = {
    'bert-21b': {'parameter_size': 8.6, 'critical_stages': ['bert-submod-0', 'bert-submod-20']},
    'gpt-66b': {'parameter_size': 66.0, 'critical_stages': ['gpt-submod-0', 'gpt-submod-65']},
    'llama-7b': {'parameter_size': 7.0, 'critical_stages': ['llama-submod-0', 'llama-submod-6']},
    'whisper-large': {'parameter_size': 1.5, 'critical_stages': ['whisper-submod-0']},
}

gpu_metric = GPUMetric()
priority_scores = {}
last_scheduled_time = {}
scheduling_history = []


def standar_mem(memory_str):
    """Standardize memory string to GB"""
    memory_match = re.match(r'(\d+)([KMGT]i?)?', memory_str)
    memory_value = int(memory_match.group(1))
    memory_unit = memory_match.group(2).replace('i', '') if memory_match.group(2) else None

    if memory_unit:
        memory_multiplier = MEMORY_UNITS.get(memory_unit, 1)
        memory_allocatable = memory_value / memory_multiplier
    else:
        memory_allocatable = memory_value / 1024 ** 3
    return memory_allocatable


def get_function_gpu_require(function):
    """Get GPU requirements for a function"""
    try:
        function_model_size_df = pd.read_csv('../benchmark/function_model_size.csv')
        if '-whole' not in function:
            gpu_quota = 1.4
        else:
            gpu_quota = GPU_QUOTA
        
        function_row = function_model_size_df[function_model_size_df['function'] == function]
        if not function_row.empty:
            model_size = function_row['model_size'].values[0]
            required_gpus = int(np.ceil(model_size / gpu_quota))
            return required_gpus, model_size
        else:
            logging.warning(f"Function {function} not found in model size CSV")
            return 1, 1.0
    except Exception as e:
        logging.error(f"Error getting GPU requirements: {e}")
        return 1, 1.0


def env_to_kv(env_list):
    """Convert environment variable list to key-value dict"""
    env_kv = {}
    for env_var in env_list:
        if 'name' in env_var and 'value' in env_var:
            env_kv[env_var['name']] = env_var['value']
    return env_kv


def get_kv_master_ip_with_retry(kv_master_function, max_retries=5, retry_delay=2):
    """Get KV master IP with retry logic"""
    for attempt in range(max_retries):
        try:
            pods = api.list_namespaced_pod(namespace=NAMESPACE).items
            for pod in pods:
                if pod.metadata.labels.get('faas_function') == kv_master_function:
                    if pod.status.phase == 'Running' and pod.status.pod_ip:
                        return pod.status.pod_ip, pod.metadata.name
            
            logging.warning(f"KV master {kv_master_function} not found, attempt {attempt + 1}/{max_retries}")
            time.sleep(retry_delay)
        except Exception as e:
            logging.error(f"Error getting KV master IP: {e}")
            time.sleep(retry_delay)
    
    raise Exception(f"Could not find KV master {kv_master_function} after {max_retries} attempts")


def create_resoure_patch(pod):
    """Create resource patch for pod"""
    container_env = pod['spec']['containers'][0].get('env', [])
    env_kv = env_to_kv(container_env)
    
    gpu_function_patch = []
    resources_value = {}
    
    # Add resource patches based on pod requirements
    if 'gpu_requirement' in env_kv:
        gpu_requirement = int(env_kv['gpu_requirement'])
        resources_value['nvidia.com/gpu'] = str(gpu_requirement)
    
    return gpu_function_patch, resources_value


def patch_pod(pod, node_name, device_uuid_list):
    """Patch pod with node assignment and GPU allocation"""
    container_env = pod['spec']['containers'][0].get('env', [])
    env_kv = env_to_kv(container_env)
    extend_patch = create_resoure_patch(pod)
    gpu_function_patch = []
    resources_value = {}
    function_name = pod['metadata']['labels'].get('faas_function')

    # Handle KV master configuration
    if "kv_master" in env_kv:
        kv_master_val = env_kv["kv_master"]
        if kv_master_val == function_name:
            kv_master_ip = "0.0.0.0"
        else:
            kv_master_ip, instance_name = get_kv_master_ip_with_retry(kv_master_val)

        # Replace placeholders in extra_args
        for env_var in container_env:
            if env_var.get('name') == 'extra_args' and 'value' in env_var:
                env_var['value'] = env_var['value'].replace(
                    "{kv_master_ip}", 
                    kv_master_ip
                )
    
    # Check hostIPC configuration
    has_host_ipc = pod.get('spec', {}).get('hostIPC', False)
    logging.info(f"Pod hostIPC status: {has_host_ipc}")
    
    # Create patch operations
    patch_operation = [
        {
            "op": "add",
            "path": "/spec/nodeSelector",
            "value": {
                "kubernetes.io/hostname": node_name
            }
        },
        {
            "op": "add" if not has_host_ipc else "replace",
            "path": "/spec/hostIPC",
            "value": True
        },
        {
            "op": "add",
            "path": "/spec/containers/0/env",
            "value": container_env + [
                {
                    "name": "CUDA_VISIBLE_DEVICES",
                    "value": ",".join([str(i) for i in range(len(device_uuid_list))])
                }
            ]
        }
    ]
    
    return patch_operation


def calculate_score(row, functions_on_gpu, model_require_gpu_memory_util, 
                   function_qps_map, target_function, cache_percent):
    """Calculate scheduling score for a GPU"""
    gpu_util = row['gpu_util']
    memory_util = row['memory_util']
    
    # Base score from resource utilization
    base_score = (1 - gpu_util) * 0.4 + (1 - memory_util) * 0.6
    
    # Cache bonus
    cache_bonus = cache_percent / 100.0 * 0.3
    
    # Function co-location penalty
    colocation_penalty = 0
    for func in functions_on_gpu:
        if func != target_function:
            colocation_penalty += 0.1
    
    final_score = base_score + cache_bonus - colocation_penalty
    return max(0, final_score)


def record_scheduling_history(pod_name, function, selected_node, device_uuid_list, 
                             scheduling_latency_ms, spare_GM, node_cache_percentages,
                             candidate_nodes, scheduling_reason="normal"):
    """Record scheduling decision for analysis"""
    current_time = time.time()
    
    # Calculate cluster statistics
    total_available_gpus = len(spare_GM)
    total_nodes_available = spare_GM['node_name'].nunique() if not spare_GM.empty else 0
    cluster_avg_gpu_util = spare_GM['gpu_util'].mean() if not spare_GM.empty else 0.0
    cluster_avg_memory_util = spare_GM['memory_util'].mean() if not spare_GM.empty else 0.0
    
    recently_scheduled_nodes = sum(1 for node, last_time in last_scheduled_time.items() 
                                 if (current_time - last_time) < 60)
    
    # Get selected node info
    selected_node_info = spare_GM[spare_GM['node_name'] == selected_node].iloc[0] if not spare_GM.empty else None
    selected_node_gpu_util = selected_node_info['gpu_util'] if selected_node_info is not None else 0.0
    selected_node_memory_util = selected_node_info['memory_util'] if selected_node_info is not None else 0.0
    
    cache_hit_rate = node_cache_percentages.get(selected_node, 0.0)
    
    # Record history
    history_entry = {
        'timestamp': datetime.now().isoformat(),
        'pod_name': pod_name,
        'function': function,
        'selected_node': selected_node,
        'gpu_count': len(device_uuid_list),
        'scheduling_latency_ms': scheduling_latency_ms,
        'cache_hit_rate': cache_hit_rate,
        'cluster_avg_gpu_util': cluster_avg_gpu_util,
        'cluster_avg_memory_util': cluster_avg_memory_util,
        'total_available_gpus': total_available_gpus,
        'scheduling_reason': scheduling_reason
    }
    
    scheduling_history.append(history_entry)
    logging.info(f"📝 Recorded scheduling history for pod {pod_name}")


class QuartScheduler:
    """
    Integrated scheduler combining cache-aware scheduling with Quart components
    """

    def __init__(self):
        self.cache_scheduler = cache_scheduler
        self.replica_corrector = replica_corrector
        self.pipeline_smoother = pipeline_smoother
        self.cpu_compensator = cpu_compensator
        self.gpu_metric = gpu_metric

        # Track available components
        self.available_components = {
            'cache': cache_available,
            'replica': replica_available,
            'smoother': smoother_available,
            'cpu': cpu_available
        }

        # Initialize mappings
        self.server_to_pods, self.server_gpu_map = self._init_mappings()
        
        # Initialize cluster state for cache-aware scheduling
        if cache_available:
            self._initialize_cluster()

        available_list = [k for k, v in self.available_components.items() if v]
        logging.info(f"QuartScheduler initialized with {len(available_list)} available components")

    def _init_mappings(self):
        """Initialize server to pods and GPU mappings"""
        server_to_pods = {}
        server_gpu_map = {}
        
        if not k8s_available:
            return server_to_pods, server_gpu_map
        
        try:
            node_list = api.list_node().items
            for node in node_list:
                node_name = node.metadata.name
                server_to_pods[node_name] = []
                gpu_allocatable = int(node.status.allocatable.get("nvidia.com/gpu", 0))
                server_gpu_map[node_name] = gpu_allocatable
                if gpu_allocatable > 0 and node_name not in priority_scores:
                    priority_scores[node_name] = 100
            
            pod_list = api.list_namespaced_pod(namespace=NAMESPACE).items
            for pod in pod_list:
                if pod.spec.scheduler_name != SCHEDULER_NAME:
                    continue
                node_name = pod.spec.node_name
                if node_name is None:
                    continue
                server_to_pods[node_name].append(pod.metadata.name)
                gpu_requested = int(pod.spec.containers[0].resources.requests.get("nvidia.com/gpu", 0))
                if gpu_requested > 0:
                    server_gpu_map[node_name] -= gpu_requested
        except Exception as e:
            logging.error(f"Error initializing mappings: {e}")
        
        return server_to_pods, server_gpu_map

    def _initialize_cluster(self):
        """Initialize cluster state for cache-aware scheduling"""
        if not k8s_available:
            logging.info("Kubernetes not available, using mock cluster configuration")
            mock_servers = [
                ("mock-node-1", 32, 4),
                ("mock-node-2", 64, 8),
                ("mock-node-3", 32, 4),
            ]
            self.cache_scheduler.initialize_cluster(mock_servers)
            return

        try:
            nodes = api.list_node()
            cluster_servers = []

            for node in nodes.items:
                node_name = node.metadata.name
                gpu_count = self._get_node_gpu_count(node)

                if gpu_count > 0:
                    memory_gb = 64 if 'A100' in str(node.metadata.labels) else 32
                    cluster_servers.append((node_name, memory_gb, gpu_count))

            self.cache_scheduler.initialize_cluster(cluster_servers)
            logging.info(f"Initialized cluster with {len(cluster_servers)} GPU servers")

        except Exception as e:
            logging.error(f"Error initializing cluster: {e}")

    def _get_node_gpu_count(self, node) -> int:
        """Get GPU count for a node"""
        node_name = node.metadata.name
        for gpu_type, nodes in NODE_LIST.items():
            if node_name in nodes:
                if gpu_type == "3090":
                    return 8
                elif gpu_type == "A100":
                    return 4
                elif gpu_type == "V100":
                    return 8
                elif gpu_type == "A40":
                    return 4
        return 0

    def schedule_pod(self, pod):
        """
        Main scheduling function combining cache-aware and Quart components
        """
        start_time = time.time()
        pod_name = pod['metadata']['name']
        function = pod['metadata']['labels'].get('faas_function')
        
        logging.info(f"\n{'='*80}")
        logging.info(f"🎯 Scheduling pod: {pod_name} (function: {function})")
        logging.info(f"{'='*80}")
        
        try:
            # Get GPU requirements
            required_gpus, model_size = get_function_gpu_require(function)
            logging.info(f"📊 GPU Requirements: {required_gpus} GPUs, Model Size: {model_size}GB")
            
            # Get GPU metrics
            spare_GM = self.gpu_metric.get_spare_gpu_metric()
            if spare_GM.empty:
                raise Exception("No spare GPUs available in cluster")
            
            logging.info(f"💾 Available GPUs: {len(spare_GM)} across {spare_GM['node_name'].nunique()} nodes")
            
            # Get cache information (if cache scheduler available)
            node_cache_percentages = {}
            if cache_available:
                try:
                    # Use cache scheduler to get cache information
                    for node in spare_GM['node_name'].unique():
                        cache_info = self.cache_scheduler.get_cache_info(node, function)
                        if cache_info:
                            node_cache_percentages[node] = cache_info.get('cache_percent', 0.0)
                except Exception as e:
                    logging.warning(f"Could not get cache information: {e}")
            
            # Build GPU to function mapping
            gpu_to_function_map = {}
            function_qps_map = {}
            
            # Calculate scores for each GPU
            def get_cache_percent(node_name):
                return node_cache_percentages.get(node_name, 0.0)
            
            spare_GM['score'] = spare_GM.apply(
                lambda row: calculate_score(
                    row,
                    gpu_to_function_map.get(row['device_uuid'], []),
                    0.8,  # model_require_gpu_memory_util
                    function_qps_map,
                    function,
                    get_cache_percent(row['node_name'])
                ),
                axis=1
            )
            
            # Find candidate nodes
            def find_candidate_nodes(gm_data):
                grouped = gm_data.groupby('node_name')
                candidate_nodes = {}
                current_time = time.time()
                
                for node, group in grouped:
                    if len(group) >= required_gpus:
                        sorted_group = group.sort_values(by='score', ascending=False)
                        candidate_score = sorted_group.head(required_gpus)['score'].sum()
                        
                        # Apply time-based penalty
                        if node in last_scheduled_time and (current_time - last_scheduled_time[node] < 60):
                            candidate_score *= 0.5
                        
                        # Apply cache bonus
                        cache_percent = node_cache_percentages.get(node, 0.0)
                        if cache_percent > 0:
                            candidate_score *= (1.0 + cache_percent / 100.0)
                        
                        candidate_nodes[node] = (candidate_score, sorted_group, cache_percent)
                
                return candidate_nodes
            
            candidate_nodes = find_candidate_nodes(spare_GM)
            
            if not candidate_nodes:
                raise Exception(f"No node available with {required_gpus} GPUs for {function}")
            
            # Sort candidates by cache percentage and score
            sorted_candidates = sorted(
                candidate_nodes.items(),
                key=lambda item: (item[1][2], item[1][0]),
                reverse=True
            )
            
            selected_node = sorted_candidates[0][0]
            selected_info = sorted_candidates[0][1]
            selected_group = selected_info[1]
            cache_percent = selected_info[2]
            
            selected_gpus = selected_group.head(required_gpus)
            device_uuid_list = list(selected_gpus['device_uuid'])
            
            # Update scheduling time
            current_time = time.time()
            last_scheduled_time[selected_node] = current_time
            
            # Calculate scheduling latency
            scheduling_latency_ms = (time.time() - start_time) * 1000
            
            # Record scheduling history
            record_scheduling_history(
                pod_name, function, selected_node, device_uuid_list,
                scheduling_latency_ms, spare_GM, node_cache_percentages,
                candidate_nodes, "cache_aware"
            )
            
            # Log scheduling decision
            logging.info(f"\n✅ Scheduling Decision:")
            logging.info(f"   ├─ Selected Node: {selected_node}")
            logging.info(f"   ├─ GPUs Allocated: {len(device_uuid_list)}")
            logging.info(f"   ├─ Cache Hit Rate: {cache_percent:.1f}%")
            logging.info(f"   ├─ Node Score: {selected_info[0]:.4f}")
            logging.info(f"   └─ Scheduling Latency: {scheduling_latency_ms:.2f}ms")
            
            return selected_node, device_uuid_list
            
        except Exception as e:
            logging.error(f"❌ Scheduling failed for {pod_name}: {e}")
            raise

    def mutate(self, pod_dict: Dict) -> Dict:
        """
        Mutate pod specification with scheduling decisions.
        This is the main interface for the admission webhook.
        
        Args:
            pod_dict: Pod specification as dictionary
            
        Returns:
            Dictionary containing patch operations for the pod
        """
        try:
            pod_name = pod_dict.get('metadata', {}).get('name', 'unknown')
            logging.info(f"🔧 Mutating pod: {pod_name}")
            
            # Schedule the pod to get node and device assignments
            selected_node, device_uuid_list = self.schedule_pod(pod_dict)
            
            # Generate patch operations
            patch_operations = patch_pod(pod_dict, selected_node, device_uuid_list)
            
            # Return the patch in the format expected by Kubernetes admission webhook
            response = {
                "allowed": True,
                "patchType": "JSONPatch",
                "patch": base64.b64encode(
                    json.dumps(patch_operations).encode()
                ).decode()
            }
            
            logging.info(f"✅ Successfully mutated pod {pod_name}")
            return response
            
        except Exception as e:
            logging.error(f"❌ Failed to mutate pod: {e}")
            # Return allowed=True but without patches to let pod schedule normally
            return {
                "allowed": True,
                "status": {
                    "message": f"Scheduling failed: {str(e)}"
                }
            }

    def optimize_pipeline_placement(self, pipeline_stages: List[str],
                                  model_name: str) -> Dict[str, Tuple[str, int]]:
        """
        Optimize placement for an entire pipeline using Quart components
        """
        logging.info(f"\n{'='*80}")
        logging.info(f"🔧 Optimizing placement for {model_name} pipeline")
        logging.info(f"{'='*80}")

        # Identify critical stages using replica corrector
        critical_stages = []
        if replica_available:
            try:
                critical_analysis = self.replica_corrector.identify_critical_stages(
                    pipeline_stages, NAMESPACE
                )
                critical_stages = [stage for stage, score in critical_analysis if score > 0.7]
                logging.info(f"🎯 Critical stages identified: {critical_stages}")
            except Exception as e:
                logging.warning(f"Could not identify critical stages: {e}")

        # Schedule each stage
        placement_decisions = {}
        for stage in pipeline_stages:
            is_critical = stage in critical_stages
            
            # Create mock pod for scheduling
            mock_pod = {
                'metadata': {
                    'name': f"{stage}-pod",
                    'labels': {'faas_function': stage}
                },
                'spec': {
                    'containers': [{'env': []}]
                }
            }
            
            try:
                server, device_list = self.schedule_pod(mock_pod)
                gpu_id = 0 if device_list else 0
                placement_decisions[stage] = (server, gpu_id)
            except Exception as e:
                logging.error(f"Failed to schedule {stage}: {e}")
                placement_decisions[stage] = ("default-node", 0)

        # Apply pipeline smoothing
        if smoother_available:
            try:
                logging.info("\n🔄 Applying pipeline smoothing...")
                current_replicas = {stage: 1 for stage in pipeline_stages}
                corrected_replicas = {stage: (2 if stage in critical_stages else 1) 
                                    for stage in pipeline_stages}
                
                smoothed_replicas = self.pipeline_smoother.apply_pipeline_smoothing(
                    corrected_replicas, pipeline_stages, NAMESPACE
                )
                logging.info(f"✅ Smoothed replica distribution: {smoothed_replicas}")
            except Exception as e:
                logging.warning(f"Pipeline smoothing failed: {e}")

        return placement_decisions

    def balance_workload(self, current_placements: Dict[str, Tuple[str, int]]) -> Dict[str, float]:
        """Balance workload using CPU compensation"""
        if not cpu_available:
            return {}
        
        try:
            logging.info("\n⚖️  Analyzing workload balance...")
            
            node_workloads = {}
            for stage, (server, gpu) in current_placements.items():
                if server not in node_workloads:
                    node_workloads[server] = []
                node_workloads[server].append(stage)
            
            compensations = {}
            for server, stages in node_workloads.items():
                if len(stages) > 1:
                    replica_changes = {stage: (1, 2) for stage in stages}
                    request_rates = {stage: 10.0 for stage in stages}
                    
                    server_compensations = self.cpu_compensator.compensate_after_smoothing(
                        replica_changes, request_rates
                    )
                    compensations[server] = server_compensations
            
            return compensations
            
        except Exception as e:
            logging.error(f"Workload balancing failed: {e}")
            return {}


# Flask app for webhook integration
app = Flask(__name__)
quart_scheduler = QuartScheduler()


@app.route('/mutate', methods=['POST'])
def mutate_endpoint():
    """
    Admission webhook endpoint for pod mutation.
    This is the standard Kubernetes mutating webhook interface.
    """
    try:
        # Parse admission review request
        admission_review = request.get_json()
        
        if not admission_review:
            return {"error": "No data provided"}, 400
        
        # Extract pod from admission request
        admission_request = admission_review.get('request', {})
        pod = admission_request.get('object')
        
        if not pod:
            return {"error": "No pod object in request"}, 400
        
        # Call mutate method
        mutation_response = quart_scheduler.mutate(pod)
        
        # Wrap in admission review response
        admission_response = {
            "apiVersion": "admission.k8s.io/v1",
            "kind": "AdmissionReview",
            "response": {
                "uid": admission_request.get('uid'),
                **mutation_response
            }
        }
        
        return admission_response, 200
        
    except Exception as e:
        logging.error(f"Mutation endpoint error: {e}")
        # Return error response in admission review format
        return {
            "apiVersion": "admission.k8s.io/v1",
            "kind": "AdmissionReview",
            "response": {
                "uid": request.get_json().get('request', {}).get('uid'),
                "allowed": False,
                "status": {
                    "message": str(e)
                }
            }
        }, 500


@app.route('/schedule', methods=['POST'])
def schedule_endpoint():
    """Webhook endpoint for scheduling requests"""
    try:
        data = request.get_json()
        if not data:
            return {"error": "No data provided"}, 400

        pod = data.get('pod')
        if not pod:
            return {"error": "pod data required"}, 400

        node, device_list = quart_scheduler.schedule_pod(pod)

        response = {
            "pod_name": pod['metadata']['name'],
            "node": node,
            "devices": device_list,
            "scheduler": SCHEDULER_NAME
        }

        return response, 200

    except Exception as e:
        logging.error(f"Scheduling endpoint error: {e}")
        return {"error": str(e)}, 500


@app.route('/optimize_pipeline', methods=['POST'])
def optimize_pipeline_endpoint():
    """Endpoint for optimizing entire pipeline placement"""
    try:
        data = request.get_json()
        if not data:
            return {"error": "No data provided"}, 400

        pipeline_stages = data.get('stages', [])
        model_name = data.get('model_name')

        if not pipeline_stages or not model_name:
            return {"error": "stages and model_name required"}, 400

        placements = quart_scheduler.optimize_pipeline_placement(pipeline_stages, model_name)
        compensations = quart_scheduler.balance_workload(placements)

        response = {
            "model_name": model_name,
            "placements": placements,
            "compensations": compensations,
            "scheduler": SCHEDULER_NAME
        }

        return response, 200

    except Exception as e:
        logging.error(f"Pipeline optimization error: {e}")
        return {"error": str(e)}, 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "scheduler": SCHEDULER_NAME,
        "components": {
            "cache_scheduler": cache_available,
            "replica_corrector": replica_available,
            "pipeline_smoother": smoother_available,
            "cpu_compensator": cpu_available
        }
    }, 200


def main():
    """Main scheduler loop"""
    logging.info("="*80)
    logging.info("🚀 Starting Quart Integrated Scheduler")
    logging.info("="*80)
    logging.info(f"Scheduler: {SCHEDULER_NAME}")
    logging.info(f"Namespace: {NAMESPACE}")
    logging.info(f"Components: Cache-Aware + Quart (Replica/Smoother/CPU)")

    try:
        app.run(
            host='0.0.0.0',
            port=5000,
            ssl_context=context if os.path.exists('hook/server.crt') else None,
            debug=False
        )
    except Exception as e:
        logging.error(f"Error starting scheduler: {e}")


if __name__ == "__main__":
    main()