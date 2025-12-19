"""
Unit tests for PipelineSmoother module
"""

import unittest
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from OCD.PipelineSmoother import PipelineSmoother


class TestPipelineSmoother(unittest.TestCase):
    """Test Pipeline Smoother"""
    
    def setUp(self):
        """Set up test fixtures"""
        class MockPrometheus:
            def custom_query(self, query):
                return []
        
        self.smoother = PipelineSmoother(MockPrometheus())
    
    def test_compute_cv_empty(self):
        """Test CV computation with empty list"""
        cv = self.smoother.compute_cv([])
        self.assertEqual(cv, 0.0)
    
    def test_compute_cv_single_value(self):
        """Test CV computation with single value"""
        cv = self.smoother.compute_cv([5.0])
        self.assertEqual(cv, 0.0)
    
    def test_compute_cv_no_variation(self):
        """Test CV computation with no variation"""
        cv = self.smoother.compute_cv([5.0, 5.0, 5.0, 5.0])
        self.assertEqual(cv, 0.0)
    
    def test_compute_cv_with_variation(self):
        """Test CV computation with variation"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        cv = self.smoother.compute_cv(values)
        
        mean = np.mean(values)
        std = np.std(values)
        expected_cv = std / mean
        
        self.assertAlmostEqual(cv, expected_cv, places=5)
    
    def test_compute_cv_zero_mean(self):
        """Test CV computation with zero mean"""
        cv = self.smoother.compute_cv([0.0, 0.0, 0.0])
        self.assertEqual(cv, 0.0)
    
    def test_build_pipeline_graph(self):
        """Test pipeline graph construction"""
        stages = ["stage-0", "stage-1", "stage-2", "stage-3"]
        adj = self.smoother.build_pipeline_graph(stages)
        
        # Check shape
        self.assertEqual(adj.shape, (4, 4))
        
        # Check self-loops
        for i in range(4):
            self.assertEqual(adj[i, i], 1)
        
        # Check sequential connections
        for i in range(3):
            self.assertEqual(adj[i, i+1], 1)
            self.assertEqual(adj[i+1, i], 1)
    
    def test_smoothing_coefficient(self):
        """Test smoothing coefficient calculation"""
        # Low CV should give low beta (more smoothing)
        beta_low = self.smoother.smoothing_coefficient(0.1)
        
        # High CV should give high beta (preserve replicas)
        beta_high = self.smoother.smoothing_coefficient(2.0)
        
        self.assertLess(beta_low, beta_high)
        self.assertGreaterEqual(beta_low, 0.0)
        self.assertLessEqual(beta_high, 1.0)
    
    def test_smooth_replicas_preserves_critical(self):
        """Test that smoothing preserves high-CV stages"""
        corrected = {
            "stage-0": 5,
            "stage-1": 6,
            "stage-2": 4
        }
        
        cv_predictions = {
            "stage-0": 0.2,  # Low CV - can smooth
            "stage-1": 1.5,  # High CV - preserve
            "stage-2": 0.3   # Low CV - can smooth
        }
        
        stages = ["stage-0", "stage-1", "stage-2"]
        
        smoothed = self.smoother.smooth_replicas(corrected, cv_predictions, stages)
        
        # High CV stage should be close to original
        self.assertGreaterEqual(smoothed["stage-1"], corrected["stage-1"] - 1)
    
    def test_smooth_replicas_respects_minimum(self):
        """Test that smoothing respects minimum replicas"""
        corrected = {"stage-0": 2, "stage-1": 1}
        cv_predictions = {"stage-0": 0.1, "stage-1": 0.1}
        stages = ["stage-0", "stage-1"]
        
        smoothed = self.smoother.smooth_replicas(corrected, cv_predictions, stages, min_replicas=1)
        
        for stage in stages:
            self.assertGreaterEqual(smoothed[stage], 1)


class TestGraphAttentionLayer(unittest.TestCase):
    """Test Graph Attention Layer"""
    
    def test_gat_layer_forward(self):
        """Test GAT layer forward pass"""
        from OCD.PipelineSmoother import GraphAttentionLayer
        
        layer = GraphAttentionLayer(in_features=8, out_features=16)
        
        # Create dummy input
        h = torch.randn(4, 8)  # 4 nodes, 8 features
        adj = torch.eye(4)  # Self-loops only
        cv_values = torch.tensor([0.5, 1.0, 0.3, 0.8])
        
        # Forward pass
        output = layer(h, adj, cv_values)
        
        # Check output shape
        self.assertEqual(output.shape, (4, 16))


if __name__ == '__main__':
    unittest.main()
