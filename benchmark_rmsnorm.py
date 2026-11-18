import statistics
import torch
import torch.nn.functional as F
from rmsnorm_reference import rmsnorm_reference
from run_rmsnorm import module
def benchmark(fn,warmup=20,iterations=100,repeats=5):
    #预热,不计入时间
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    #多轮测量减少误差
    for _ in range(repeats):
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        average_us = start.elapsed_time(end)*1000/iterations
        samples.append(average_us)
    return statistics.median(samples)
def main():
    torch.manual_seed(42)

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)
    print("dtype: float32")
    print("计时单位：微秒 us；越小越快")

    shapes = [
        (1, 4096),
        (32, 4096),
        (256, 4096),
        (130, 257),
    ]

    eps = 1e-6

    with torch.no_grad():
        for rows, cols in shapes:
            x = torch.randn(
                rows, cols,
                dtype=torch.float32,
                device="cuda",
            )

            weight = torch.randn(
                cols,
                dtype=torch.float32,
                device="cuda",
            )

            def run_reference():
                return rmsnorm_reference(x, weight, eps)

            def run_pytorch():
                return F.rms_norm(
                    x,
                    normalized_shape=(cols,),
                    weight=weight,
                    eps=eps,
                )

            def run_custom_cuda():
                return module.rmsnorm(x, weight, eps)

            # 先检查当前测量形状的结果。
            expected = run_reference()

            torch.testing.assert_close(
                run_custom_cuda(),
                expected,
                rtol=1e-4,
                atol=1e-5,
            )

            torch.testing.assert_close(
                run_pytorch(),
                expected,
                rtol=1e-4,
                atol=1e-5,
            )

            reference_us = benchmark(run_reference)
            pytorch_us = benchmark(run_pytorch)
            custom_us = benchmark(run_custom_cuda)

            speedup = pytorch_us / custom_us

            print(f"\nshape=({rows}, {cols})")
            print(f"  参考公式：       {reference_us:.3f} us")
            print(f"  PyTorch 官方：   {pytorch_us:.3f} us")
            print(f"  自定义 CUDA：    {custom_us:.3f} us")
            print(f"  相对官方加速比： {speedup:.3f}x")


if __name__ == "__main__":
    main()