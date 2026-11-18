import torch
from rmsnorm_reference import rmsnorm_reference
from run_rmsnorm import module
torch.manual_seed(42)
eps = 1e-6
shapes = [
    (1,1),
    (1,7),
    (3,5),
    (8,128),
    (4,257),
    (130,257),
    (32,1024),
    (16,4096),
]
for rows,cols in shapes:
    x = torch.randn(
        rows,cols,
        dtype=torch.float32,
        device="cuda"
    )
    weight = torch.randn(
        cols,
        dtype=torch.float32,
        device="cuda",
    )
    actual = module.rmsnorm(x,weight,eps)
    expected = rmsnorm_reference(x,weight,eps)
    torch.testing.assert_close(
        actual,expected,rtol=1e-4,atol=1e-5,
    )
    max_error = (actual - expected).abs().max().item()
    print(
        f"通过:shapes=({rows},{cols}),"
        f"最大绝对误差={max_error:.3e}"
    )
x_zeros = torch.zeros(3,257,dtype=torch.float32,device="cuda",)
weight = torch.ones(257,dtype=torch.float32,device="cuda",)
actual_zero = module.rmsnorm(x_zeros,weight,eps)
torch.testing.assert_close(actual_zero,torch.zeros_like(x_zeros),rtol=0,atol=0,)
print("通过:全零输入")
print("CUDA RMSNorm正确性测试全部通过")