import torch
import torch.multiprocessing as mp

from argparse import ArgumentParser
from gpu_adam_bench import GPUAdam_Benchmark

NUM_WARMUP=10

def worker_run_benchmark(rank, dtype, param_size, num_bench, shared_latency_variable, barrier, lock, return_queue):  
    # print(f"Process {rank}| Benchmarking with dtype={dtype}, param_size={param_size}")
    # print(f"Process {rank}| Running benchmark: {NUM_WARMUP} warm-up steps, {num_bench} benchmark steps")
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        device = torch.device(f'cuda:{rank % num_gpus}')  # Assign device based on rank
        torch.cuda.set_device(device)
        # print(f"Process {rank}| Assigned to {device}")
    else:
        device = torch.device('cpu')
        # print(f"Process {rank}| No GPUs available, using CPU")
    
    benchmark_instance = GPUAdam_Benchmark(dtype, param_size, device)
    total_h2d_latency_ns = 0.0
    total_d_compute_latency_ns = 0.0
    total_d2h_latency_ns = 0.0
    
    
    for _ in range(NUM_WARMUP):
        benchmark_instance.step()

    for _ in range(num_bench):
        barrier.wait()  # make sure all process is ready to start
        h2d_latency_ns, d_compute_latency_ns, d2h_latency_ns = benchmark_instance.step()
        
        # print(f"Process {rank}| h2d_latency_ns: {h2d_latency_ns}")
        # print(f"Process {rank}| d_compute_latency_ns: {d_compute_latency_ns}")
        # print(f"Process {rank}| d2h_latency_ns: {d2h_latency_ns}")
        
        with lock:
            if h2d_latency_ns > shared_latency_variable[0]:
                shared_latency_variable[0] = h2d_latency_ns
            if d_compute_latency_ns > shared_latency_variable[1]:
                shared_latency_variable[1] = d_compute_latency_ns
            if d2h_latency_ns > shared_latency_variable[2]:
                shared_latency_variable[2] = d2h_latency_ns

        barrier.wait()  # make sure all process finish checking maximum value
        total_h2d_latency_ns += shared_latency_variable[0]
        total_d_compute_latency_ns += shared_latency_variable[1]
        total_d2h_latency_ns += shared_latency_variable[2]
        
        barrier.wait()  # make sure all process finish accumulation
        if rank == 0:
            shared_latency_variable[0] = -1
            shared_latency_variable[1] = -1
            shared_latency_variable[2] = -1
                   
    avg_h2d_latency_per_step = total_h2d_latency_ns / num_bench
    avg_d_compute_latency_per_step = total_d_compute_latency_ns / num_bench
    avg_d2h_latency_per_step = total_d2h_latency_ns / num_bench
    # print(f"Process {rank}| Average H2D Latency per step: {avg_h2d_latency_per_step:.6f} ns")
    # print(f"Process {rank}| Average Computation Latency per step: {avg_d_compute_latency_per_step:.6f} ns")
    # print(f"Process {rank}| Average D2H Latency per step: {avg_d2h_latency_per_step:.6f} ns")
    return_queue.put((rank, avg_h2d_latency_per_step, avg_d_compute_latency_per_step, avg_d2h_latency_per_step))
  
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--nprocess",
        type=int,
        required=True
    )
    parser.add_argument(
        "--param_size",
        type=int,
        required=True
    )
    parser.add_argument(
        "--num_bench",
        type=int,
        required=True
    )
    args = parser.parse_args()
    
    nprocess = args.nprocess
    param_size = args.param_size
    num_bench = args.num_bench
    dtype = torch.float32
    
    aligned_nprocess_param_size = ((param_size + nprocess - 1) / nprocess) * nprocess
    partitioned_param_size = int(aligned_nprocess_param_size // nprocess)
    
    mp.set_start_method('spawn')
    shared_latency_variable = mp.Array('l', [0, 0, 0])
    lock = mp.Lock()
    barrier = mp.Barrier(nprocess)
    return_queue = mp.Queue()

    processes = []
    for rank in range(nprocess):
        p = mp.Process(target=worker_run_benchmark, args=(rank, dtype, partitioned_param_size, num_bench, shared_latency_variable, barrier, lock, return_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    _, avg_h2d, avg_d_compute, avg_d2h = return_queue.get()
    print(f"Benchmarking with benchmark steps={num_bench}, dtype={dtype}, param_size={param_size}, nprocess={nprocess}, partitioned_param_size={partitioned_param_size}")
    print(f"Average H2D Latency per step: {avg_h2d / 1e6:.6f} ms")
    print(f"Average Computation Latency per step: {avg_d_compute / 1e6:.6f} ms")
    print(f"Average D2H Latency per step: {avg_d2h / 1e6:.6f} ms")