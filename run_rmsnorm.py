from pathlib import Path
import torch
from torch.utils.cpp_extension import load
project_dir = Path(__file__).resolve().parent
module = load(
    name="rmsnorm_ext",
    sources=[
        str(project_dir / "rmsnorm_binding.cpp"),
        str(project_dir / "rmsnorm_kernel.cu"),
    ],
    extra_cflags=["-O2"],
    extra_cuda_cflags=["-O2"],
    verbose=True,
)
x = torch.tensor(
    [
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 0.0, 0.0, 0.0],
    ],
    dtype=torch.float32,
    device="cuda",
)

weight = torch.tensor(
    [1.0, 2.0, 1.0, 0.5],
    dtype=torch.float32,
    device="cuda",
)

# 调用 CUDA 算子。
y = module.rmsnorm(x, weight, 1e-6)

print("输出所在设备：", y.device)
print("输出结果：")
print(y.cpu())