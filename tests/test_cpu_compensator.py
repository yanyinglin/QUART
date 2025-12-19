"""
Unit tests for CPUCompensator module
"""

import unittest
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from OCD.CPUCompensator import CPUDemandModel, CPUCompensator


class TestCPUDemandModel(unittest.TestCase):
    """Test CPU Demand Model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = CPUDemandModel()
    
    def test_phi_concurrency_increases(self):
        """Test that concurrency overhead increases with concentration"""
        phi1 = self.model.phi_concurrency(1.0)
        phi2 = self.model.phi_concurrency(2.0)
        phi3 = self.model.phi_concurrency(4.0)
        
        self.assertLess(phi1, phi2)
        self.assertLess(phi2, phi3)
    
    def test_phi_concurrency_positive(self):
        """Test that concurrency overhead is always positive"""
        for lambda_conc in [1.0, 2.0, 4.0, 8.0]:
            phi = self.model.phi_concurrency(lambda_conc)
            self.assertGreater(phi, 0)
    
    def test_psi_network_increases_with_size(self):
        """Test that network overhead increases with tensor size"""
        psi1 = self.model.psi_network(1.0)
        psi2 = self.model.psi_network(2.0)
        psi3 = self.model.psi_network(4.0)
        
        self.assertLess(psi1, psi2)
        self.assertLess(psi2, psi3)
    
    def test_psi_network_with_complexity(self):
        """Test network overhead with different complexities"""
        psi_simple = self.model.psi_network(1.0, activation_complexity=0.5)
        psi_complex = self.model.psi_network(1.0, activation_complexity=2.0)
        
        self.assertLess(psi_simple, psi_complex)
    
    def test_xi_memory_components(self):
        """Test memory overhead components"""
        # Test fragmentation effect
        xi_low_frag = self.model.xi_memory(0.1, 0.5)
        xi_high_frag = self.model.xi_memory(0.9, 0.5)
        self.assertLess(xi_low_frag, xi_high_frag)
        
        # Test GC frequency effect
        xi_low_gc = self.model.xi_memory(0.3, 0.1)
        xi_high_gc = self.model.xi_memory(0.3, 1.0)
        self.assertLess(xi_low_gc, xi_high_gc)
    
    def test_predict_cpu_requirement_base(self):
        """Test CPU requirement prediction with no load"""
        base_cpu = 2.0
        required = self.model.predict_cpu_requirement(
            base_cpu=base_cpu,
            lambda_conc=1.0,
            tensor_size=0.0,
            fragmentation_index=0.0,
            gc_frequency=0.0
        )
        
        # Should be at least base CPU
        self.assertGreaterEqual(required, base_cpu)
    
    def test_predict_cpu_requirement_increases_with_concentration(self):
        """Test CPU requirement increases with workload concentration"""
        base_cpu = 2.0
        
        req1 = self.model.predict_cpu_requirement(base_cpu, lambda_conc=1.0)
        req2 = self.model.predict_cpu_requirement(base_cpu, lambda_conc=2.0)
        req4 = self.model.predict_cpu_requirement(base_cpu, lambda_conc=4.0)
        
        self.assertLess(req1, req2)
        self.assertLess(req2, req4)
    
    def test_predict_cpu_requirement_all_factors(self):
        """Test CPU requirement with all factors"""
        required = self.model.predict_cpu_requirement(
            base_cpu=2.0,
            lambda_conc=4.0,
            tensor_size=2.0,
            activation_complexity=1.5,
            fragmentation_index=0.5,
            gc_frequency=0.8
        )
        
        # Should be significantly higher than base
        self.assertGreater(required, 2.0)


class TestCPUCompensatorIntegration(unittest.TestCase):
    """Integration tests for CPUCompensator"""
    
    def test_calculate_concentration_factor(self):
        """Test concentration factor calculation"""
        class MockPrometheus:
            def custom_query(self, query):
                return []
        
        try:
            compensator = CPUCompensator(MockPrometheus())
            
            # Test normal concentration
            factor = compensator.calculate_concentration_factor(
                prev_request_rate=100.0,
                new_request_rate=100.0,
                prev_replicas=5,
                new_replicas=3
            )
            
            # Concentration should be > 1 when replicas decrease
            self.assertGreater(factor, 1.0)
            
            # Test no concentration
            factor = compensator.calculate_concentration_factor(
                prev_request_rate=100.0,
                new_request_rate=100.0,
                prev_replicas=5,
                new_replicas=5
            )
            
            self.assertEqual(factor, 1.0)
        except Exception as e:
            # If Kubernetes not available, skip
            if "kube" in str(e).lower():
                self.skipTest("Kubernetes not available")
            raise


if __name__ == '__main__':
    unittest.main()
