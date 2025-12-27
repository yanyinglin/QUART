"""
Unit tests for CacheAwareScheduler module
"""

import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quart.core.cache_scheduler import (
    CachedParameter, ServerCache, KeysManager,
    KLDivergenceOptimizer, CacheAwareScheduler
)


class TestServerCache(unittest.TestCase):
    """Test ServerCache"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cache = ServerCache(
            server_name="test-server",
            total_memory=256.0,
            gpu_count=4
        )
    
    def test_initialization(self):
        """Test server cache initialization"""
        self.assertEqual(self.cache.server_name, "test-server")
        self.assertEqual(self.cache.total_memory, 256.0)
        self.assertEqual(self.cache.gpu_count, 4)
        self.assertEqual(self.cache.used_memory, 0.0)
        self.assertEqual(len(self.cache.cached_parameters), 0)
    
    def test_memory_utilization(self):
        """Test memory utilization calculation"""
        self.cache.used_memory = 128.0
        util = self.cache.memory_utilization()
        self.assertAlmostEqual(util, 50.0, places=1)
    
    def test_add_cached_success(self):
        """Test adding cached parameter"""
        param = CachedParameter(
            stage_name="test-stage",
            model_name="test-model",
            parameter_size=10.0,
            cached_time=0.0,
            last_access_time=0.0
        )
        
        success = self.cache.add_cached(param)
        self.assertTrue(success)
        self.assertTrue(self.cache.has_cached("test-stage"))
        self.assertEqual(self.cache.used_memory, 10.0)
    
    def test_add_cached_exceeds_threshold(self):
        """Test adding parameter that exceeds memory threshold"""
        # Fill cache to near capacity
        self.cache.used_memory = 220.0  # 85% of 256 is 217.6
        
        param = CachedParameter(
            stage_name="test-stage",
            model_name="test-model",
            parameter_size=50.0,  # Would exceed threshold
            cached_time=0.0,
            last_access_time=0.0
        )
        
        success = self.cache.add_cached(param)
        self.assertFalse(success)
    
    def test_evict_lru(self):
        """Test LRU eviction"""
        # Add multiple parameters
        for i in range(3):
            param = CachedParameter(
                stage_name=f"stage-{i}",
                model_name="model",
                parameter_size=10.0,
                cached_time=float(i),
                last_access_time=float(i)
            )
            self.cache.add_cached(param)
        
        # Evict LRU
        evicted = self.cache.evict_lru()
        
        self.assertIsNotNone(evicted)
        self.assertEqual(evicted.stage_name, "stage-0")  # Oldest
        self.assertFalse(self.cache.has_cached("stage-0"))


class TestKeysManager(unittest.TestCase):
    """Test KeysManager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.manager = KeysManager()
        self.manager.initialize_server_cache("server-1", 256.0, 4)
        self.manager.initialize_server_cache("server-2", 256.0, 4)
    
    def test_initialization(self):
        """Test KeysManager initialization"""
        self.assertEqual(len(self.manager.server_caches), 2)
        self.assertIn("server-1", self.manager.server_caches)
        self.assertIn("server-2", self.manager.server_caches)
    
    def test_cache_parameters(self):
        """Test caching parameters"""
        success = self.manager.cache_parameters_cow(
            stage_name="test-stage",
            model_name="test-model",
            parameter_size=10.0,
            server_name="server-1"
        )
        
        self.assertTrue(success)
        
        # Check it's cached
        servers = self.manager.get_servers_with_cached_stage("test-stage")
        self.assertIn("server-1", servers)
    
    def test_cache_hit(self):
        """Test cache hit scenario"""
        # Cache first time
        self.manager.cache_parameters_cow(
            "test-stage", "test-model", 10.0, "server-1"
        )
        
        # Cache again (should be hit)
        success = self.manager.cache_parameters_cow(
            "test-stage", "test-model", 10.0, "server-1"
        )
        
        self.assertTrue(success)
        
        # Check hit count increased
        cached = self.manager.server_caches["server-1"].cached_parameters["test-stage"]
        self.assertGreater(cached.hit_count, 0)
    
    def test_cache_eviction_on_full(self):
        """Test cache eviction when memory is full"""
        # Fill cache
        for i in range(20):
            self.manager.cache_parameters_cow(
                f"stage-{i}", "model", 12.0, "server-1"
            )
        
        # Memory should be managed (evictions occurred)
        cache = self.manager.server_caches["server-1"]
        self.assertLess(cache.memory_utilization(), 90.0)


class TestKLDivergenceOptimizer(unittest.TestCase):
    """Test KL Divergence Optimizer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = KLDivergenceOptimizer(
            learning_rate=0.1,
            max_iterations=50,
            convergence_threshold=0.01
        )
    
    def test_compute_kl_divergence_identical(self):
        """Test KL divergence for identical distributions"""
        P = np.array([[0.25, 0.25, 0.25, 0.25]])
        Q = np.array([[0.25, 0.25, 0.25, 0.25]])
        
        kl = self.optimizer.compute_kl_divergence(P, Q)
        self.assertAlmostEqual(kl, 0.0, places=5)
    
    def test_compute_kl_divergence_different(self):
        """Test KL divergence for different distributions"""
        P = np.array([[0.7, 0.2, 0.05, 0.05]])
        Q = np.array([[0.25, 0.25, 0.25, 0.25]])
        
        kl = self.optimizer.compute_kl_divergence(P, Q)
        self.assertGreater(kl, 0.0)
    
    def test_compute_gradient_shape(self):
        """Test gradient computation shape"""
        P = np.random.rand(4, 3)
        P = P / P.sum(axis=1, keepdims=True)
        Q = np.ones((4, 3)) / 3
        
        gradient = self.optimizer.compute_gradient(P, Q)
        self.assertEqual(gradient.shape, P.shape)
    
    def test_optimize_placement_converges(self):
        """Test that optimization converges"""
        num_stages = 5
        num_servers = 3
        critical_stages = {0, 2}
        server_capacities = np.array([4.0, 4.0, 4.0])
        
        placement = self.optimizer.optimize_placement(
            num_stages, num_servers, critical_stages, server_capacities
        )
        
        # Check shape
        self.assertEqual(placement.shape, (num_stages, num_servers))
        
        # Check each row sums to 1 (probability distribution)
        row_sums = placement.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(num_stages), decimal=5)
    
    def test_select_server(self):
        """Test server selection"""
        placement_probs = np.array([
            [0.5, 0.3, 0.2],
            [0.2, 0.5, 0.3]
        ])
        
        available_servers = [0, 1, 2]
        
        # Select multiple times to test randomness
        selections = []
        for _ in range(10):
            selected = self.optimizer.select_server(placement_probs, 0, available_servers)
            selections.append(selected)
            self.assertIn(selected, available_servers)
        
        # Should have some variation
        self.assertGreater(len(set(selections)), 1)


class TestCacheAwareSchedulerIntegration(unittest.TestCase):
    """Integration tests for CacheAwareScheduler"""
    
    def test_scheduler_initialization(self):
        """Test scheduler initialization"""
        class MockPrometheus:
            def custom_query(self, query):
                return []
        
        try:
            scheduler = CacheAwareScheduler(MockPrometheus())
            
            servers = [
                ("server-1", 256, 4),
                ("server-2", 256, 4)
            ]
            
            scheduler.initialize_cluster(servers)
            
            self.assertEqual(len(scheduler.keys_manager.server_caches), 2)
        except Exception as e:
            if "kube" in str(e).lower():
                self.skipTest("Kubernetes not available")
            raise


if __name__ == '__main__':
    unittest.main()
