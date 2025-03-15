import torch
import torch.nn as nn
from fused_adam import FusedAdam

# Function to set gradients manually to a constant value (e.g., 1.0)
def set_gradients(model):
    for param in model.parameters():
        param.grad = torch.ones_like(param, device='cuda')

# Function to compare two optimizers
def compare_optimizers(optimizer1, optimizer2, model1, model2, num_steps=5):
    """
    Compares two optimizers by running optimization steps and checking parameter and state consistency.
    
    Args:
        optimizer1: First optimizer (e.g., torch.optim.Adam or AdamW)
        optimizer2: Second optimizer (FusedAdam)
        model1: Model optimized by optimizer1
        model2: Model optimized by optimizer2
        num_steps: Number of optimization steps to perform
    """
    for step in range(num_steps):
        # Set identical gradients for both models
        set_gradients(model1)
        set_gradients(model2)

        # Perform optimization step
        optimizer1.step()
        optimizer2.step()

        # Compare parameter values
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6), f"Parameters differ at step {step}"

        # Compare optimizer states (exp_avg, exp_avg_sq, step)
        for group1, group2 in zip(optimizer1.param_groups, optimizer2.param_groups):
            for p1, p2 in zip(group1['params'], group2['params']):
                state1 = optimizer1.state[p1]
                state2 = optimizer2.state[p2]
                assert torch.allclose(state1['exp_avg'], state2['exp_avg'], atol=1e-6), \
                    f"exp_avg differs at step {step}"
                assert torch.allclose(state1['exp_avg_sq'], state2['exp_avg_sq'], atol=1e-6), \
                    f"exp_avg_sq differs at step {step}"
                print(p1, p2)

    print(f"Optimizers match for {num_steps} steps")

# Common hyperparameters
lr = 0.001
betas = (0.9, 0.999)
eps = 1e-8
num_steps = 5

# Test 2: FusedAdam with adam_w_mode=True vs torch.optim.AdamW (decoupled weight decay)
print("\nRunning Test 2: FusedAdam (adam_w_mode=True) vs torch.optim.AdamW")
model1 = nn.Linear(2, 5).to('cuda')
model2 = nn.Linear(2, 5).to('cuda')

# Ensure models start with identical parameters
model2.weight.data.copy_(model1.weight.data)
model2.bias.data.copy_(model1.bias.data)

# Set weight_decay to test decoupled weight decay
weight_decay = 0.01

optimizer1 = torch.optim.AdamW(
    model1.parameters(),
    lr=lr,
    betas=betas,
    eps=eps,
    weight_decay=weight_decay
)
optimizer2 = FusedAdam(
    model2.parameters(),
    lr=lr,
    betas=betas,
    eps=eps,
    weight_decay=weight_decay,
    adam_w_mode=True  # Matches torch.optim.AdamW behavior
)

compare_optimizers(optimizer1, optimizer2, model1, model2, num_steps)

print("\nAll tests passed successfully!") 