import time
import torch
from fused_adam import multi_tensor_adam
from fused_adam import MultiTensorApply

# Hyperparameters 
lr = 1e-3
bias_correction = True
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 0
adamw_mode = True
    
class GPUAdam_Benchmark:
    def __init__(self, dtype: torch.dtype, param_size: int, device: torch.device) -> None:
        """Initialize the benchmark with optimizer and tensors."""
        self.optimizer_id = 0
        self.dtype = dtype
        self.param_size = param_size
        self.step_id = 0
        self.adam_w_mode = 1 if adamw_mode else 0
        self.bias_correction = 1 if bias_correction else 0
        self.device = device

        # Initialize tensors
        self.param_cpu = torch.zeros((param_size,), dtype=dtype, device="cpu", pin_memory=True)
        self.grad_cpu = torch.randn((param_size,), dtype=dtype, device="cpu", pin_memory=True)
        self.exp_avg_cpu = torch.zeros((param_size,), dtype=dtype, device="cpu", pin_memory=True)
        self.exp_avg_sq_cpu = torch.zeros((param_size,), dtype=dtype, device="cpu", pin_memory=True)
        self.param = torch.empty((param_size,), dtype=dtype, device=self.device)
        self.grad = torch.empty((param_size,), dtype=dtype, device=self.device)
        self.exp_avg = torch.empty((param_size,), dtype=dtype, device=self.device)
        self.exp_avg_sq = torch.empty((param_size,), dtype=dtype, device=self.device)
        
        self._dummy_overflow_buf = torch.zeros((1,), dtype=torch.int, device=self.device)
        self.multi_tensor_adam = multi_tensor_adam
        self.multi_tensor_applier = MultiTensorApply(2048 * 32)

    @torch.no_grad()
    def step(self) -> tuple[float, float, float]:
        self.step_id += 1
        beta1, beta2 = betas
        
        # create lists for multi-tensor apply
        g_16, p_16, m_16, v_16 = [], [], [], []
        g_bf, p_bf, m_bf, v_bf = [], [], [], []
        g_32, p_32, m_32, v_32 = [], [], [], []
        
        start_host_to_device = time.perf_counter_ns()
        
        self.param.copy_(self.param_cpu, non_blocking=True)
        self.grad.copy_(self.grad_cpu, non_blocking=True)
        self.exp_avg.copy_(self.exp_avg_cpu, non_blocking=True)
        self.exp_avg_sq.copy_(self.exp_avg_sq_cpu, non_blocking=True)
        
        if self.dtype == torch.float16:
            g_16.append(self.grad)
            p_16.append(self.param)
            m_16.append(self.exp_avg)
            v_16.append(self.exp_avg_sq)  
        elif self.dtype == torch.bfloat16:
            g_bf.append(self.grad)
            p_bf.append(self.param)
            m_bf.append(self.exp_avg)
            v_bf.append(self.exp_avg_sq)
        elif self.dtype == torch.float32:
            g_32.append(self.grad)
            p_32.append(self.param)
            m_32.append(self.exp_avg)
            v_32.append(self.exp_avg_sq)
        else:
            raise RuntimeError('FusedAdam only support fp16, bf16 and fp32.')

        torch.cuda.synchronize()
        
        end_host_to_device = time.perf_counter_ns()
        
        start_device_computation = time.perf_counter_ns()
        
        if len(g_16) > 0:
            self.multi_tensor_applier(self.multi_tensor_adam, self._dummy_overflow_buf, [g_16, p_16, m_16, v_16],
                                    lr, beta1, beta2, eps, self.step_id, self.adam_w_mode,
                                    bias_correction, weight_decay)

        if len(g_bf) > 0:
            self.multi_tensor_applier(self.multi_tensor_adam, self._dummy_overflow_buf, [g_bf, p_bf, m_bf, v_bf],
                                    lr, beta1, beta2, eps, self.step_id, self.adam_w_mode,
                                    bias_correction, weight_decay)

        if len(g_32) > 0:
            self.multi_tensor_applier(self.multi_tensor_adam, self._dummy_overflow_buf, [g_32, p_32, m_32, v_32],
                                    lr, beta1, beta2, eps, self.step_id, self.adam_w_mode,
                                    bias_correction, weight_decay)
            
        end_device_computation = time.perf_counter_ns()
        
        start_device_to_host = time.perf_counter_ns()
        
        self.param_cpu.copy_(self.param, non_blocking=True)
        self.grad_cpu.copy_(self.grad, non_blocking=True)
        self.exp_avg_cpu.copy_(self.exp_avg, non_blocking=True)
        self.exp_avg_sq_cpu.copy_(self.exp_avg_sq, non_blocking=True)
        torch.cuda.synchronize()
        
        end_device_to_host = time.perf_counter_ns()
        return end_host_to_device-start_host_to_device, end_device_computation-start_device_computation, end_device_to_host-start_device_to_host
        
if __name__ == "__main__":
    device=torch.device("cuda")      
    dtype=torch.float32
    param_size = 1024768
    benchmark_instance = GPUAdam_Benchmark(dtype, param_size,device)

    h2d, d_compute, d2h = benchmark_instance.step()
    print(f"Host to Device latency: {h2d/1000000:.2f} ms")
    print(f"Device Computation latency: {d_compute/1000000:.2f} ms")
    print(f"Device to Host latency: {d2h/1000000:.2f} ms")