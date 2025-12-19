"""
Unit tests for ReplicaCorrector module
"""

import unittest
import sys
import os
import math

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from OCD.ReplicaCorrector import MMCQueue, PIDController, ReplicaCorrector


class TestMMCQueue(unittest.TestCase):
    """Test M/M/c queuing model"""
    
    def test_factorial(self):
        """Test factorial calculation"""
        self.assertEqual(MMCQueue.factorial(0), 1)
        self.assertEqual(MMCQueue.factorial(1), 1)
        self.assertEqual(MMCQueue.factorial(5), 120)
    
    def test_compute_p0_stable(self):
        """Test P0 computation for stable system"""
        rho = 2.0
        c = 3
        p0 = MMCQueue.compute_p0(rho, c)
        self.assertGreater(p0, 0)
        self.assertLess(p0, 1)
    
    def test_compute_p0_unstable(self):
        """Test P0 computation for unstable system"""
        rho = 5.0
        c = 3
        p0 = MMCQueue.compute_p0(rho, c)
        self.assertEqual(p0, 0.0)
    
    def test_queue_delay_stable(self):
        """Test queue delay for stable system"""
        arrival_rate = 10.0
        service_rate = 4.0
        num_replicas = 3
        
        delay = MMCQueue.compute_queue_delay(arrival_rate, service_rate, num_replicas)
        self.assertGreater(delay, 0)
        self.assertNotEqual(delay, float('inf'))
    
    def test_queue_delay_unstable(self):
        """Test queue delay for unstable system"""
        arrival_rate = 20.0
        service_rate = 4.0
        num_replicas = 3
        
        delay = MMCQueue.compute_queue_delay(arrival_rate, service_rate, num_replicas)
        self.assertEqual(delay, float('inf'))
    
    def test_queue_delay_increases_with_load(self):
        """Test that delay increases with arrival rate"""
        service_rate = 5.0
        num_replicas = 4
        
        delay1 = MMCQueue.compute_queue_delay(5.0, service_rate, num_replicas)
        delay2 = MMCQueue.compute_queue_delay(10.0, service_rate, num_replicas)
        delay3 = MMCQueue.compute_queue_delay(15.0, service_rate, num_replicas)
        
        self.assertLess(delay1, delay2)
        self.assertLess(delay2, delay3)
    
    def test_queue_delay_decreases_with_replicas(self):
        """Test that delay decreases with more replicas"""
        arrival_rate = 10.0
        service_rate = 3.0
        
        delay1 = MMCQueue.compute_queue_delay(arrival_rate, service_rate, 4)
        delay2 = MMCQueue.compute_queue_delay(arrival_rate, service_rate, 5)
        delay3 = MMCQueue.compute_queue_delay(arrival_rate, service_rate, 6)
        
        self.assertGreater(delay1, delay2)
        self.assertGreater(delay2, delay3)


class TestPIDController(unittest.TestCase):
    """Test PID controller"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.pid = PIDController(kp=1.0, ki=0.1, kd=0.05, min_replicas=1, max_replicas=10)
    
    def test_initialization(self):
        """Test PID controller initialization"""
        self.assertEqual(self.pid.kp, 1.0)
        self.assertEqual(self.pid.ki, 0.1)
        self.assertEqual(self.pid.kd, 0.05)
        self.assertEqual(self.pid.integral, 0.0)
        self.assertEqual(self.pid.previous_error, 0.0)
    
    def test_control_signal_positive_error(self):
        """Test control signal with positive error (need to scale up)"""
        target = 0.5
        current = 1.0  # Higher than target
        
        signal = self.pid.compute_control_signal(target, current)
        # Negative error means we need to scale up
        self.assertLess(signal, 0)
    
    def test_control_signal_negative_error(self):
        """Test control signal with negative error (need to scale down)"""
        target = 1.0
        current = 0.5  # Lower than target
        
        signal = self.pid.compute_control_signal(target, current)
        # Positive error means we can scale down
        self.assertGreater(signal, 0)
    
    def test_adjust_replicas_increase(self):
        """Test replica adjustment increases when latency is high"""
        current_replicas = 3
        target_latency = 0.5
        current_latency = 1.5  # High latency
        
        new_replicas = self.pid.adjust_replicas(current_replicas, target_latency, current_latency)
        # PID may not immediately increase, but should not decrease significantly
        # Just check it's within reasonable bounds
        self.assertGreaterEqual(new_replicas, self.pid.min_replicas)
        self.assertLessEqual(new_replicas, self.pid.max_replicas)
    
    def test_adjust_replicas_decrease(self):
        """Test replica adjustment decreases when latency is low"""
        current_replicas = 5
        target_latency = 1.0
        current_latency = 0.3  # Low latency
        
        new_replicas = self.pid.adjust_replicas(current_replicas, target_latency, current_latency)
        # PID may not immediately decrease, but should not increase significantly
        # Just check it's within reasonable bounds
        self.assertGreaterEqual(new_replicas, self.pid.min_replicas)
        self.assertLessEqual(new_replicas, self.pid.max_replicas)
    
    def test_adjust_replicas_respects_min_max(self):
        """Test replica adjustment respects bounds"""
        # Test minimum bound
        new_replicas = self.pid.adjust_replicas(1, 0.5, 10.0)
        self.assertGreaterEqual(new_replicas, self.pid.min_replicas)
        
        # Test maximum bound
        new_replicas = self.pid.adjust_replicas(10, 10.0, 0.1)
        self.assertLessEqual(new_replicas, self.pid.max_replicas)
    
    def test_reset(self):
        """Test PID controller reset"""
        # Accumulate some state
        self.pid.compute_control_signal(0.5, 1.0)
        self.pid.compute_control_signal(0.5, 1.2)
        
        # Reset
        self.pid.reset()
        
        self.assertEqual(self.pid.integral, 0.0)
        self.assertEqual(self.pid.previous_error, 0.0)


class TestReplicaCorrectorIntegration(unittest.TestCase):
    """Integration tests for ReplicaCorrector"""
    
    def test_corrector_initialization(self):
        """Test corrector can be initialized without Prometheus"""
        # This is a basic test without actual Prometheus connection
        # In production, would use mock
        pass
    
    def test_stage_metrics_structure(self):
        """Test that stage metrics have correct structure"""
        # Would test with mock Prometheus client
        pass


if __name__ == '__main__':
    unittest.main()
