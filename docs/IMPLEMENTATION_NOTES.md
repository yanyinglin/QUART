# Quart Implementation Notes

## Overview

This document provides implementation details for the Quart system, including the newly implemented components and their integration with the existing codebase.

## Implemented Components

### 1. ReplicaCorrector.py

**Location**: `OCD/ReplicaCorrector.py`

**Description**: Implements the Pipeline-Aware Replica Correction mechanism with PID control and M/M/c queuing theory.

**Key Classes**:
- `MMCQueue`: M/M/c queuing model for stage delay prediction (Equation 1)
- `PIDController`: PID controller for dynamic replica adjustment (Equations 2-3)
- `ReplicaCorrector`: Main corrector orchestrating critical stage identification and replica adjustment

**Usage Example**:
```python
from OCD.ReplicaCorrector import ReplicaCorrector

corrector = ReplicaCorrector(prometheus_client, target_latency=0.5)
recommendations = corrector.correct_pipeline_replicas(
    pipeline_stages=["bert-submod-0", "bert-submod-1", ...],
    namespace="cdgp"
)
```

**Integration**: This can be called from `DaShengScaler.py` before the scaling decision to get PID-based recommendations.

### 2. PipelineSmoother.py

**Location**: `OCD/PipelineSmoother.py`

**Description**: Implements CV-Based Pipeline Stage Smoothing using Graph Attention Networks.

**Key Classes**:
- `GraphAttentionLayer`: GAT layer for CV propagation (Equations 4-5)
- `CVPropagationModel`: Complete GAT model for CV prediction
- `PipelineSmoother`: Main smoother applying adaptive resource reallocation (Equation 6)

**Usage Example**:
```python
from OCD.PipelineSmoother import PipelineSmoother

smoother = PipelineSmoother(prometheus_client)
smoothed_replicas = smoother.apply_pipeline_smoothing(
    corrected_replicas=recommendations,
    stages=pipeline_stages,
    namespace="cdgp"
)
```

**Integration**: This should be called after replica correction to optimize resource distribution.

### 3. CPUCompensator.py

**Location**: `OCD/CPUCompensator.py`

**Description**: Implements Adaptive CPU Compensation for concentrated workloads.

**Key Classes**:
- `CPUDemandModel`: Predictive model for CPU requirements (Equations 7-10)
- `CPUCompensator`: Adaptive allocation using Algorithm 2

**Usage Example**:
```python
from OCD.CPUCompensator import CPUCompensator

compensator = CPUCompensator(prometheus_client)
cpu_allocations = compensator.compensate_after_smoothing(
    stage_replicas_changes={
        "bert-submod-0": (5, 3),  # old_replicas, new_replicas
    },
    stage_request_rates={"bert-submod-0": 10.5}
)
```

**Integration**: This should be called after smoothing when replicas are reduced.

### 4. CacheAwareScheduler.py

**Location**: `OCD/CacheAwareScheduler.py`

**Description**: Implements Cache-Aware Scheduling with KeysManager and KL divergence optimization.

**Key Classes**:
- `KeysManager`: Hierarchical parameter caching with COW (Section 7.1)
- `KLDivergenceOptimizer`: Dispersed placement using KL divergence (Equations 11-13)
- `CacheAwareScheduler`: Complete scheduling system

**Usage Example**:
```python
from OCD.CacheAwareScheduler import CacheAwareScheduler

scheduler = CacheAwareScheduler(prometheus_client)
scheduler.initialize_cluster([
    ("server-1", 256, 4),  # name, memory_gb, gpu_count
    ...
])

server, gpu = scheduler.schedule_critical_stage(
    stage_name="bert-submod-0",
    model_name="bert-21b",
    parameter_size=8.6,
    is_critical=True
)
```

**Integration**: This can replace or augment the scheduling logic in `DaShengScheduler.py`.

## Integration Strategy

### Option 1: Incremental Integration

Gradually integrate components into existing `DaShengScaler.py` and `DaShengScheduler.py`:

1. **Phase 1**: Add replica correction to `DaShengScaler.py`
   ```python
   from ReplicaCorrector import ReplicaCorrector
   
   corrector = ReplicaCorrector(prometheus_client)
   corrected = corrector.correct_pipeline_replicas(stages)
   # Use corrected instead of current calculate_replicas_by_model()
   ```

2. **Phase 2**: Add pipeline smoothing
   ```python
   from PipelineSmoother import PipelineSmoother
   
   smoother = PipelineSmoother(prometheus_client)
   smoothed = smoother.apply_pipeline_smoothing(corrected, stages)
   ```

3. **Phase 3**: Add CPU compensation
   ```python
   from CPUCompensator import CPUCompensator
   
   compensator = CPUCompensator(prometheus_client)
   compensator.compensate_after_smoothing(replica_changes, rates)
   ```

4. **Phase 4**: Integrate cache-aware scheduling in `DaShengScheduler.py`
   ```python
   from CacheAwareScheduler import CacheAwareScheduler
   
   scheduler = CacheAwareScheduler(prometheus_client)
   # Use in schedule() function
   ```

### Option 2: Complete Replacement

Create new `QuartScaler.py` and `QuartScheduler.py` that use all new components:

```python
# QuartScaler.py
from ReplicaCorrector import ReplicaCorrector
from PipelineSmoother import PipelineSmoother
from CPUCompensator import CPUCompensator

class QuartScaler:
    def __init__(self, prometheus_client):
        self.corrector = ReplicaCorrector(prometheus_client)
        self.smoother = PipelineSmoother(prometheus_client)
        self.compensator = CPUCompensator(prometheus_client)
    
    async def scale_pipeline(self, stages, namespace):
        # Step 1: Correct replicas using PID
        corrected = self.corrector.correct_pipeline_replicas(stages, namespace)
        
        # Step 2: Smooth resource allocation
        smoothed = self.smoother.apply_pipeline_smoothing(
            corrected, stages, namespace
        )
        
        # Step 3: Compensate CPU if needed
        replica_changes = {
            stage: (corrected[stage], smoothed[stage])
            for stage in stages
        }
        self.compensator.compensate_after_smoothing(
            replica_changes, stage_request_rates
        )
        
        return smoothed
```

## Testing

Each component includes a `__main__` section with unit tests:

```bash
# Test ReplicaCorrector
python OCD/ReplicaCorrector.py

# Test PipelineSmoother
python OCD/PipelineSmoother.py

# Test CPUCompensator
python OCD/CPUCompensator.py

# Test CacheAwareScheduler
python OCD/CacheAwareScheduler.py
```

## Performance Tuning

### PID Tuning

For more responsive scaling:
```python
corrector = ReplicaCorrector(prometheus, kp=3.0, ki=0.5, kd=0.2)
```

For more stable scaling:
```python
corrector = ReplicaCorrector(prometheus, kp=1.0, ki=0.1, kd=0.05)
```

### CV Propagation Model Training

The GAT model can be trained on historical data:
```python
smoother.cv_model.train()
# Implement training loop with historical CV data
```

### CPU Compensation Parameters

Adjust based on your hardware:
```python
compensator = CPUCompensator(
    prometheus,
    min_cpu=2.0,      # Minimum for your workload
    max_cpu=32.0,     # Based on server capacity
    increment_ratio=0.3  # Larger for faster adaptation
)
```

## File Organization

```
OCD/
├── ReplicaCorrector.py       
├── PipelineSmoother.py       
├── CPUCompensator.py         
├── CacheAwareScheduler.py     
├── DaShengScaler.py           
├── DaShengScheduler.py        
├── Metrics.py                 
└── perfering.py               
```

## Known Limitations

1. **GAT Model**: Currently uses random weights; should be trained on historical data
2. **Performance**: First iteration may be slower due to cold start; improves with caching

