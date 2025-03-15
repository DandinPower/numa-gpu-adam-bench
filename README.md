# numa-gpu-adam-bench

## Introduction

This repository provides a benchmark for the GPU Adam optimizer, built upon the DeepSpeed library. It is designed to measure the performance of the optimizer on Linux systems with NUMA (Non-Uniform Memory Access) architectures and multiple NVIDIA GPUs. Specifically, this benchmark aims to assess the performance when tensor data—such as parameters, gradients, and optimizer states—is placed in CPU memory, while the computations are performed on the GPU. This setup helps reveal the actual performance implications of data transfer latencies between CPU and GPU and GPU computation. The benchmark supports `fp16`, `bf16`, and `fp32` data types, but the current implementation hardcodes the data type to `float32`. It uses `numactl` to manage memory allocation policies for CPU tensors across NUMA nodes and employs multi-processing to simulate concurrent optimizer updates on multiple GPUs.

## Prerequisites

To build and run this benchmark, ensure the following requirements are met:

- **System**: A machine with multiple NUMA nodes and 1 or more NVIDIA GPUs.
- **CUDA**: Installed to support GPU computations.
- **numactl**: Installed to control NUMA memory policies.

On Ubuntu, install `numactl` with:

```bash
sudo apt-get install numactl
```

Verify your NUMA configuration with:

```bash
numactl --hardware
```

### Optional: Weighted Interleave

The weighted interleave feature requires additional setup:

1. **Kernel Requirement**: Linux kernel ≥ 6.9 to support `/sys/kernel/mm/mempolicy/weighted_interleave/node*`. Check your kernel version with:
   ```bash
   uname -r
   ```
   Update your kernel if necessary.

2. **Latest `numactl`**: Weighted interleave requires `numactl` ≥ 2.0.19 (APT provides 2.0.18), so compile it manually:
   ```bash
   sudo apt remove numactl libnuma-dev
   git clone https://github.com/numactl/numactl.git
   cd numactl
   git checkout v2.0.19
   ./autogen.sh
   ./configure
   make
   make test
   sudo make install
   ```
   Verify the version with:
   ```bash
   numactl --version
   ```

**Note**: The weighted interleave feature is optional. If you do not plan to use it, the standard `numactl` package and an older kernel version suffice.

## Building or Installation

Clone the repository:

```bash
git clone <repository-url>
cd numa-gpu-adam-bench
```

Install the required Python packages (ensure `requirements.txt` is present; if not, include `torch`):

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, install PyTorch manually:

```bash
pip install torch
```

Build the CUDA extension for the fused Adam optimizer:

```bash
pip install .
```

This compiles the `fused_adam` module, which includes the GPU-accelerated Adam optimizer.

**Note**: Ensure that the CUDA toolkit is installed and properly configured on your system.

## Usage and Example

Run the benchmark using the `mp_bench.py` script with `numactl` to specify NUMA memory policies for CPU tensors. The script uses multi-processing to simulate concurrent optimizer updates on multiple GPUs and reports the average latencies for host-to-device (H2D) transfer, GPU computation, and device-to-host (D2H) transfer per optimization step.

### Example Command

```bash
numactl --interleave=0,1 python mp_bench.py --nprocess 2 --param_size 1048576 --num_bench 100
```

This command:
- Interleaves memory for CPU tensors across NUMA nodes 0 and 1.
- Uses 2 processes (`--nprocess 2`), each assigned to a separate GPU.
- Sets the parameter tensor to 1,048,576 elements (`--param_size 1048576`).
- Runs 100 benchmark iterations (`--num_bench 100`).

### Arguments

- **`--nprocess`**: Number of processes to simulate concurrent updates (e.g., `2`).
- **`--param_size`**: Number of elements in the parameter tensor (e.g., `1048576` for 1M elements).
- **`--num_bench`**: Number of benchmark iterations (e.g., `100`).

### Output

The script outputs the average latencies per optimization step in milliseconds, e.g.:

```
Benchmarking with benchmark steps=100, dtype=torch.float32, param_size=1048576, nprocess=2, partitioned_param_size=524288
Average H2D Latency per step: 1.234567 ms
Average Computation Latency per step: 0.123456 ms
Average D2H Latency per step: 1.234567 ms
```

**Note**: The benchmark partitions the `param_size` across processes and measures the maximum latencies for H2D, computation, and D2H across all processes, then averages over iterations, simulating distributed training workloads. The current implementation hardcodes the data type to `float32`.

## Benchmark Scripts

Two Bash scripts are provided to automate benchmarking across different configurations and generate CSV-formatted results.

### Standard Interleave Script

The `interleave_scripts.sh` script tests standard interleave configurations:

```bash
bash interleave_scripts.sh > example/normal_interleave.csv
```

#### Script Details

- **Configurations Tested**:
  - Number of processes (`ngpus`): `1`, `2`
  - NUMA nodes (`numa_nodes`): `"0"`, `"3"` (customize based on your system)
  - Parameter sizes (`benchmark_element_items`): Predefined sizes (e.g., for Mistral Nemo 12B: `5242880`, `20971520`, `73400320`, `671088640`)
- **Fixed Parameters**:
  - Iterations per benchmark: `10` (adjustable via `ITERATION_PER_BENCH`)
  - Data type: `fp32` (hardcoded in `mp_bench.py`)
- **Output Format**:
  CSV columns:
  - `ngpus`: Number of processes
  - `numa_nodes`: Comma-separated list of NUMA nodes
  - `update_element`: Number of elements in the parameter tensor
  - `avg_h2d_latency(ms)`: Average host-to-device latency per step
  - `avg_compute_latency(ms)`: Average GPU computation latency per step
  - `avg_d2h_latency(ms)`: Average device-to-host latency per step

#### Customization

Edit `interleave_scripts.sh` to adjust:
- `ngpus_items`: Number of processes
- `numa_nodes_items`: NUMA nodes (check with `numactl --hardware`)
- `benchmark_element_items`: Parameter sizes

### Weighted Interleave Script

The `weighted_scripts.sh` script tests weighted interleave configurations (requires root access):

```bash
bash weighted_scripts.sh > example/weighted_interleave.csv
```

#### Script Details

- **Configurations Tested**:
  - Number of processes (`ngpus`): `1`, `2`
  - NUMA nodes (`numa_nodes`): `"0,3"` (customize via `LOCAL_NODE` and `CXL_NODE`)
  - Weight configurations (`numa_weight_items`): `"1,1"`, `"1,2"`, `"1,3"`, `"1,4"` (weights for nodes 0 and 3)
  - Parameter sizes (`benchmark_element_items`): Predefined sizes (e.g., for Qwen2.5 7B: `1835008`, `12845056`, `67895296`, `544997376`)
- **Fixed Parameters**:
  - Iterations per benchmark: `10` (adjustable via `ITERATION_PER_BENCH`)
  - Data type: `fp32` (hardcoded in `mp_bench.py`)
- **Output Format**:
  CSV columns:
  - `ngpus`: Number of processes
  - `numa_nodes`: Comma-separated list of NUMA nodes
  - `numa_weight`: Comma-separated weights
  - `update_element`: Number of elements
  - `avg_h2d_latency(ms)`: Average host-to-device latency per step
  - `avg_compute_latency(ms)`: Average GPU computation latency per step
  - `avg_d2h_latency(ms)`: Average device-to-host latency per step

#### Customization

Edit `weighted_scripts.sh` to adjust:
- `ngpus_items`: Number of processes
- `LOCAL_NODE` and `CXL_NODE`: NUMA nodes
- `numa_weight_items`: Weight configurations
- `benchmark_element_items`: Parameter sizes

## Acknowledge and License

The fused Adam optimizer implementation is sourced from the [DeepSpeed library](https://github.com/microsoft/DeepSpeed), which is licensed under the Apache License 2.0. This repository, `numa-gpu-adam-bench`, is also licensed under the Apache License 2.0. We acknowledge the DeepSpeed team for their foundational work on the fused Adam optimizer.