# Quart

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)

## 🎯 Key Features

- **Pipeline-Aware Resource Management**: Dynamically identifies and scales critical pipeline stages using CV-based burst propagation analysis
- **PID-Controlled Replica Allocation**: Automatically adjusts replicas for congested stages using queuing theory and feedback control
- **CV-Based Pipeline Smoothing**: Optimizes resource distribution across pipeline stages through Graph Attention Networks
- **Adaptive CPU Compensation**: Dynamically allocates CPU resources when GPU replicas are consolidated
- **Hierarchical Parameter Caching**: Enables sub-second scaling through copy-on-write memory caching (KeysManager)
- **Cache-Aware Scheduling**: Optimizes stage placement using KL divergence for maximum cache utilization


## 🏗️ System Architecture

Quart consists of four coordinated components:

```
┌─────────────────────────────────────────────────────────────┐
│                     Quart System                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │   Replica      │  │  Pipeline    │  │      CPU      │  │
│  │  Corrector     │→ │  Smoother    │→ │  Compensator  │  │
│  │  (PID+M/M/c)   │  │  (GAT+CV)    │  │  (Adaptive)   │  │
│  └────────────────┘  └──────────────┘  └───────────────┘  │
│                            ↓                                │
│                   ┌─────────────────┐                       │
│                   │  Cache-Aware    │                       │
│                   │   Scheduler     │                       │
│                   │ (KeysManager+KL)│                       │
│                   └─────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

### Component Details

1. **Replica Corrector** (`OCD/ReplicaCorrector.py`)
   - M/M/c queuing model for stage delay prediction
   - PID controller for dynamic replica adjustment
   - Critical stage identification based on queue depth

2. **Pipeline Smoother** (`OCD/PipelineSmoother.py`)
   - Graph Attention Network for CV propagation modeling
   - Adaptive smoothing strategy based on predicted burstiness
   - Resource reallocation from over-provisioned to critical stages

3. **CPU Compensator** (`OCD/CPUCompensator.py`)
   - Multi-factor CPU demand prediction model
   - Incremental allocation with performance monitoring
   - cgroup-based bandwidth control

4. **Cache-Aware Scheduler** (`OCD/CacheAwareScheduler.py`)
   - KeysManager for hierarchical parameter caching
   - KL divergence optimization for dispersed placement
   - Copy-on-write fork mechanisms for sub-second scaling

## 🚀 Quick Start

### Prerequisites

- **Cluster**: 12+ GPU servers (A40/V100/3090 or similar)
- **Kubernetes**: v1.24 or higher
- **OpenFaaS**: Latest version with faasd
- **Python**: 3.8+
- **Dependencies**: PyTorch, Kubernetes Python client, Prometheus API client

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-org/quart.git
cd quart
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Configure Kubernetes access:
```bash
export KUBECONFIG=/path/to/kubeconfig
kubectl cluster-info
```

4. Deploy Prometheus for metrics:

5. Set up OpenFaaS:
```bash
# Follow OpenFaaS installation guide
# Deploy gateway and configure namespaces
```

### Basic Usage

1. **Deploy Model Pipeline**:
```bash
cd benchmark/BERT/ME-21
./action_deploy.sh
```

2. **Start Replica Corrector**:
```bash
cd OCD
python DaShengScaler.py
```

3. **Start Cache-Aware Scheduler**:
```bash
cd OCD
python DaShengScheduler.py
```

4. **Monitor System**:
```bash
# Access Prometheus dashboard
kubectl port-forward -n monitoring svc/prometheus 9090:9090

# View scheduler logs
tail -f OCD/scheduler_record.csv
```


## 📁 Project Structure

```
quart/
├── README.md                      # This file
├── OCD/                           # Core implementation
│   ├── ReplicaCorrector.py        # PID-based replica correction
│   ├── PipelineSmoother.py        # CV-based pipeline smoothing
│   ├── CPUCompensator.py          # Adaptive CPU compensation
│   ├── CacheAwareScheduler.py     # Cache-aware scheduling
│   ├── DaShengScaler.py           # Main scaler (legacy + integration)
│   ├── DaShengScheduler.py        # Main scheduler (legacy + integration)
│   ├── Metrics.py                 # Prometheus metrics collector
│   ├── perfering.py               # GPU/CPU metrics
│   └── hook/                      # Kubernetes webhook configs
├── benchmark/                     # Model deployment configurations
│   ├── BERT/                      # BERT-21B pipeline configurations
│   ├── LLAMA/                     # LLAMA-7B configurations
│   ├── GPT/                       # OPT-66B configurations
│   ├── WHISPER/                   # Whisper-9B configurations
│   └── function_template/         # OpenFaaS templates
```


## 📈 Performance Tuning

### PID Controller Tuning

Adjust gains in `ReplicaCorrector.py`:
```python
corrector = ReplicaCorrector(
    prometheus,
    kp=2.0,  # Proportional gain (responsiveness)
    ki=0.3,  # Integral gain (steady-state error)
    kd=0.1   # Derivative gain (damping)
)
```

### Cache Memory Threshold

Configure in `CacheAwareScheduler.py`:
```python
keys_manager = KeysManager(
    memory_threshold=0.65  # Use up to 85% of server memory
)
```

### CPU Compensation Parameters

Tune in `CPUCompensator.py`:
```python
compensator = CPUCompensator(
    prometheus,
    min_cpu=1.0,
    max_cpu=16.0,
    increment_ratio=0.25  # Allocate 25% of predicted at a time
)
```

## 🐛 Troubleshooting

### Common Issues

1. **High latency despite scaling**:
   - Check PID gains (may be too conservative)
   - Verify target_latency is appropriate for your SLO
   - Ensure network bandwidth is sufficient for inter-stage communication

2. **Cache misses**:
   - Increase memory_threshold in KeysManager
   - Check server memory capacity
   - Verify COW fork is working correctly

3. **Uneven stage distribution**:
   - Reduce KL divergence learning_rate for slower convergence
   - Check server capacity constraints
   - Ensure critical stages are properly identified

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

View Prometheus metrics:
```bash
# Check stage metrics
curl "http://prometheus:9090/api/v1/query?query=gateway_function_invocation_started"

# Check GPU utilization
curl "http://prometheus:9090/api/v1/query?query=DCGM_FI_DEV_GPU_UTIL"
```

## 📝 Citation

If you use Quart in your research, please cite our paper:

```bibtex
@inproceedings{lin2024quart,
  title = {Quart: Latency-Aware FaaS System for Pipelining Large Model Inference},
  author = {Lin, Yanying and Li, Yanbo and Peng, Shijie and Tang, Yingfei and Luo, Shutian and Shen, Haiying and Xu, Chengzhong and Ye, Kejiang},
  booktitle = {Proceedings of the 44th IEEE International Conference on Distributed Computing Systems},
  year = {2024},
}
```

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

See `CONTRIBUTING.md` for detailed guidelines.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We thank the open-source community for:
- Kubernetes and OpenFaaS for serverless infrastructure
- PyTorch for deep learning framework
- Prometheus for monitoring and metrics