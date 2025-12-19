"""
CV-Based Pipeline Stage Smoothing with Graph Attention Networks
Implements pipeline smoothing using CV propagation analysis and adaptive resource reallocation.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from prometheus_api_client import PrometheusConnect
import pandas as pd


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer for CV propagation modeling.
    Uses attention mechanisms to model burstiness propagation across pipeline stages.
    """
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1, 
                 alpha: float = 0.2, concat: bool = True):
        """
        Initialize GAT layer.
        
        Args:
            in_features: Number of input features per node
            out_features: Number of output features per node
            dropout: Dropout rate
            alpha: LeakyReLU negative slope
            concat: Whether to concatenate or average multi-head attention
        """
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # Learnable weight matrix W
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Attention weight vector a
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor, cv_values: torch.Tensor, 
                theta: float = 0.5) -> torch.Tensor:
        """
        Forward pass with CV-aware attention.
        
        Args:
            h: Node feature matrix (N x in_features)
            adj: Adjacency matrix (N x N)
            cv_values: CV values for each node (N,)
            theta: Weight for CV influence on attention
        
        Returns:
            Updated node features (N x out_features)
        """
        # Linear transformation: Wh_i
        Wh = torch.mm(h, self.W)  # (N, out_features)
        
        # Compute attention coefficients
        N = Wh.size()[0]
        
        # Concatenate features for all pairs
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        
        # Compute attention scores: LeakyReLU(a^T [Wh_i || Wh_j])
        e = self.leakyrelu(torch.matmul(all_combinations, self.a).squeeze(1))
        e = e.view(N, N)
        
        # Add CV influence: + theta * CV_i (Equation 4)
        cv_influence = theta * cv_values.unsqueeze(1).expand(-1, N)
        e = e + cv_influence
        
        # Mask attention scores where no edge exists
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Normalize attention coefficients using softmax (Equation 4)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Apply attention to aggregate neighbor features (Equation 5)
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class CVPropagationModel(nn.Module):
    """
    Graph Attention Network for CV propagation prediction.
    """
    
    def __init__(self, num_features: int = 8, hidden_dim: int = 16, 
                 num_heads: int = 4, dropout: float = 0.1):
        """
        Initialize CV propagation model.
        
        Args:
            num_features: Number of input features per stage
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(CVPropagationModel, self).__init__()
        
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Multi-head attention layers
        self.attention_heads = nn.ModuleList([
            GraphAttentionLayer(num_features, hidden_dim, dropout=dropout, concat=True)
            for _ in range(num_heads)
        ])
        
        # Output layer
        self.out_layer = GraphAttentionLayer(
            hidden_dim * num_heads, 1, dropout=dropout, concat=False
        )
        
        # Feature embedding for stages
        self.feature_embedding = nn.Linear(num_features, num_features)
    
    def forward(self, features: torch.Tensor, adj: torch.Tensor, 
                cv_values: torch.Tensor) -> torch.Tensor:
        """
        Predict CV values for pipeline stages.
        
        Args:
            features: Stage feature matrix (N x num_features)
            adj: Pipeline adjacency matrix (N x N)
            cv_values: Current CV values (N,)
        
        Returns:
            Predicted CV values (N,)
        """
        # Embed features
        x = F.dropout(features, self.dropout, training=self.training)
        x = F.elu(self.feature_embedding(x))
        
        # Multi-head attention
        x = torch.cat([
            att(x, adj, cv_values) for att in self.attention_heads
        ], dim=1)
        
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Output layer
        cv_pred = self.out_layer(x, adj, cv_values)
        cv_pred = cv_pred.squeeze(1)
        
        # Ensure non-negative CV values
        cv_pred = F.relu(cv_pred)
        
        return cv_pred


class PipelineSmoother:
    """
    Pipeline stage smoothing using CV propagation and adaptive resource reallocation.
    Optimizes resource distribution based on predicted burstiness patterns.
    """
    
    def __init__(self, prometheus_client: PrometheusConnect,
                 num_features: int = 8, hidden_dim: int = 16,
                 learning_rate: float = 0.001, device: str = 'cpu'):
        """
        Initialize pipeline smoother.
        
        Args:
            prometheus_client: Prometheus client for metrics
            num_features: Number of features per stage
            hidden_dim: Hidden dimension for GAT
            learning_rate: Learning rate for model training
            device: Device for PyTorch ('cpu' or 'cuda')
        """
        self.prometheus = prometheus_client
        self.device = torch.device(device)
        
        # CV propagation model
        self.cv_model = CVPropagationModel(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_heads=4,
            dropout=0.1
        ).to(self.device)
        
        # Optimizer for online learning
        self.optimizer = torch.optim.Adam(
            self.cv_model.parameters(),
            lr=learning_rate
        )
        
        # Historical data for training
        self.history: List[Dict] = []
        
        # Model training state
        self.is_trained = False
    
    def compute_cv(self, values: List[float]) -> float:
        """
        Compute coefficient of variation.
        
        Args:
            values: List of observations
        
        Returns:
            CV = std / mean
        """
        if len(values) < 2:
            return 0.0
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if mean_val == 0:
            return 0.0
        
        return std_val / mean_val
    
    def get_stage_cv(self, stage_name: str, namespace: str = "cdgp",
                     time_window: int = 60) -> float:
        """
        Get CV of request arrival for a stage.
        
        Args:
            stage_name: Pipeline stage name
            namespace: Kubernetes namespace
            time_window: Time window in seconds
        
        Returns:
            Coefficient of variation
        """
        # Query request arrival rate over time
        query = f'irate(gateway_function_invocation_started{{function_name="{stage_name}.{namespace}"}}[{time_window}s])'
        result = self.prometheus.custom_query(query)
        
        if not result or len(result) == 0:
            return 0.0
        
        # Extract time series data
        values = []
        for r in result:
            if 'values' in r:
                for val in r['values']:
                    values.append(float(val[1]))
            elif 'value' in r:
                values.append(float(r['value'][1]))
        
        return self.compute_cv(values)
    
    def extract_stage_features(self, stage_name: str, namespace: str = "cdgp") -> np.ndarray:
        """
        Extract feature vector for a pipeline stage.
        
        Args:
            stage_name: Pipeline stage name
            namespace: Kubernetes namespace
        
        Returns:
            Feature vector (num_features,)
        """
        features = []
        
        # Feature 1: Request rate (RPS)
        rps_query = f'rate(gateway_function_invocation_started{{function_name="{stage_name}.{namespace}"}}[1m])'
        rps_result = self.prometheus.custom_query(query=rps_query)
        rps = float(rps_result[0]['value'][1]) if rps_result else 0.0
        features.append(rps)
        
        # Feature 2: Average latency
        latency_query = f'rate(gateway_functions_seconds_sum{{function_name="{stage_name}.{namespace}"}}[1m]) / rate(gateway_functions_seconds_count{{function_name="{stage_name}.{namespace}"}}[1m])'
        latency_result = self.prometheus.custom_query(query=latency_query)
        latency = float(latency_result[0]['value'][1]) if latency_result else 0.0
        features.append(latency)
        
        # Feature 3: Queue depth (approximate from latency and RPS)
        queue_depth = rps * latency
        features.append(queue_depth)
        
        # Feature 4: Current replicas
        replicas_query = f'kube_deployment_spec_replicas{{deployment="{stage_name}",namespace="{namespace}"}}'
        replicas_result = self.prometheus.custom_query(query=replicas_query)
        replicas = float(replicas_result[0]['value'][1]) if replicas_result else 1.0
        features.append(replicas)
        
        # Feature 5-8: Padding/derived features
        features.extend([
            np.log1p(rps),  # Log-transformed RPS
            np.log1p(latency),  # Log-transformed latency
            rps / max(replicas, 1),  # Per-replica throughput
            latency * replicas  # Aggregate processing capacity
        ])
        
        return np.array(features[:8])  # Ensure exactly 8 features
    
    def build_pipeline_graph(self, stages: List[str]) -> np.ndarray:
        """
        Build adjacency matrix for pipeline stages.
        
        Args:
            stages: List of stage names in order
        
        Returns:
            Adjacency matrix (N x N)
        """
        n = len(stages)
        adj = np.eye(n)  # Self-loops
        
        # Sequential connections (stage i -> stage i+1)
        for i in range(n - 1):
            adj[i, i + 1] = 1
            adj[i + 1, i] = 1  # Bidirectional for GAT
        
        return adj
    
    def predict_cv_propagation(self, stages: List[str], namespace: str = "cdgp") -> Dict[str, float]:
        """
        Predict CV values for all stages using GAT model.
        
        Args:
            stages: List of stage names in pipeline order
            namespace: Kubernetes namespace
        
        Returns:
            Dictionary mapping stage names to predicted CV values
        """
        n = len(stages)
        
        # Extract features for all stages
        features = np.zeros((n, 8))
        cv_values = np.zeros(n)
        
        for i, stage in enumerate(stages):
            features[i] = self.extract_stage_features(stage, namespace)
            cv_values[i] = self.get_stage_cv(stage, namespace)
        
        # Build adjacency matrix
        adj = self.build_pipeline_graph(stages)
        
        # Convert to tensors
        features_tensor = torch.FloatTensor(features).to(self.device)
        adj_tensor = torch.FloatTensor(adj).to(self.device)
        cv_tensor = torch.FloatTensor(cv_values).to(self.device)
        
        # Predict CV values
        self.cv_model.eval()
        with torch.no_grad():
            predicted_cv = self.cv_model(features_tensor, adj_tensor, cv_tensor)
        
        # Convert back to dictionary
        cv_predictions = {}
        for i, stage in enumerate(stages):
            cv_predictions[stage] = predicted_cv[i].item()
        
        return cv_predictions
    
    def smoothing_coefficient(self, cv: float, alpha: float = 0.5, 
                             cv_threshold: float = 1.0) -> float:
        """
        Compute smoothing coefficient beta(CV).
        
        High CV -> high beta (preserve replicas)
        Low CV -> low beta (allow smoothing)
        
        Args:
            cv: Coefficient of variation
            alpha: Base smoothing factor
            cv_threshold: CV threshold for full preservation
        
        Returns:
            Smoothing coefficient beta ∈ [0, 1]
        """
        # Sigmoid-based smoothing: beta = alpha + (1-alpha) * sigmoid(cv/threshold - 1)
        x = (cv / cv_threshold) - 1.0
        sigmoid = 1.0 / (1.0 + np.exp(-x))
        
        beta = alpha + (1 - alpha) * sigmoid
        
        return np.clip(beta, 0.0, 1.0)
    
    def smooth_replicas(self, corrected_replicas: Dict[str, int],
                       cv_predictions: Dict[str, float],
                       stages: List[str],
                       min_replicas: int = 1) -> Dict[str, int]:
        """
        Apply adaptive smoothing to replica allocation.
        
        Args:
            corrected_replicas: Replica counts after correction
            cv_predictions: Predicted CV values
            stages: List of stage names in pipeline order
            min_replicas: Minimum replicas per stage
        
        Returns:
            Smoothed replica allocation
        """
        smoothed = {}
        n = len(stages)
        
        for i, stage in enumerate(stages):
            r_corrected = corrected_replicas.get(stage, min_replicas)
            cv = cv_predictions.get(stage, 0.0)
            
            # Compute neighborhood average (previous and next stages)
            neighbors = []
            if i > 0:
                neighbors.append(corrected_replicas.get(stages[i-1], min_replicas))
            if i < n - 1:
                neighbors.append(corrected_replicas.get(stages[i+1], min_replicas))
            
            r_neighbors = np.mean(neighbors) if neighbors else r_corrected
            
            # Compute smoothing coefficient
            beta = self.smoothing_coefficient(cv)
            
            # Apply smoothing: r_smooth = beta * r_corrected + (1-beta) * r_neighbors
            r_smooth = beta * r_corrected + (1 - beta) * r_neighbors
            
            # Round and ensure minimum
            smoothed[stage] = max(min_replicas, int(round(r_smooth)))
            
            print(f"Stage {stage}: CV={cv:.3f}, beta={beta:.3f}, "
                  f"{r_corrected} -> {smoothed[stage]} replicas")
        
        return smoothed
    
    def apply_pipeline_smoothing(self, corrected_replicas: Dict[str, int],
                                 stages: List[str], namespace: str = "cdgp",
                                 min_replicas: int = 1) -> Dict[str, int]:
        """
        Complete pipeline smoothing process.
        
        Args:
            corrected_replicas: Replica counts after PID correction
            stages: List of stage names in pipeline order
            namespace: Kubernetes namespace
            min_replicas: Minimum replicas per stage
        
        Returns:
            Final smoothed replica allocation
        """
        print("=" * 60)
        print("Pipeline Smoothing")
        print("=" * 60)
        
        # Predict CV propagation
        cv_predictions = self.predict_cv_propagation(stages, namespace)
        
        print("\nCV Propagation Predictions:")
        for stage, cv in cv_predictions.items():
            print(f"  {stage}: CV = {cv:.3f}")
        
        # Apply smoothing
        print("\nApplying Adaptive Smoothing:")
        smoothed_replicas = self.smooth_replicas(
            corrected_replicas, cv_predictions, stages, min_replicas
        )
        
        # Calculate resource savings
        total_before = sum(corrected_replicas.values())
        total_after = sum(smoothed_replicas.values())
        savings = total_before - total_after
        
        print(f"\nTotal Replicas: {total_before} -> {total_after} (saved {savings})")
        
        return smoothed_replicas


# Example usage and testing
if __name__ == "__main__":
    print("Testing Pipeline Smoother")
    print("=" * 60)
    
    # Simulate pipeline stages
    stages = ["bert-submod-0", "bert-submod-1", "bert-submod-2", 
              "bert-submod-3", "bert-submod-4"]
    
    # Simulated corrected replicas (from PID)
    corrected_replicas = {
        "bert-submod-0": 5,
        "bert-submod-1": 6,
        "bert-submod-2": 4,
        "bert-submod-3": 7,
        "bert-submod-4": 3
    }
    
    # Simulated CV values (from observation)
    cv_values = {
        "bert-submod-0": 0.8,  # High CV - keep replicas
        "bert-submod-1": 1.2,  # Very high CV - preserve
        "bert-submod-2": 0.3,  # Low CV - can smooth
        "bert-submod-3": 1.5,  # Critical - preserve
        "bert-submod-4": 0.2   # Very low CV - smooth aggressively
    }
    
    print("Corrected Replicas (from PID):")
    for stage, replicas in corrected_replicas.items():
        print(f"  {stage}: {replicas} replicas")
    
    print("\nObserved CV Values:")
    for stage, cv in cv_values.items():
        print(f"  {stage}: CV = {cv:.3f}")
    
    print("\n" + "=" * 60)
    print("Applying Smoothing Algorithm")
    print("=" * 60)
    
    # Create smoother (without Prometheus for testing)
    class MockPrometheus:
        def custom_query(self, query):
            return []
    
    smoother = PipelineSmoother(MockPrometheus())
    
    # Apply smoothing manually with simulated CV
    smoothed = smoother.smooth_replicas(corrected_replicas, cv_values, stages)
    
    print("\nFinal Allocation:")
    for stage in stages:
        print(f"  {stage}: {corrected_replicas[stage]} -> {smoothed[stage]} replicas")
    
    total_before = sum(corrected_replicas.values())
    total_after = sum(smoothed.values())
    print(f"\nTotal: {total_before} -> {total_after} replicas (saved {total_before - total_after})")
