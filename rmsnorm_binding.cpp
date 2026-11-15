#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
void launch_rmsnorm(
    const float*x,
    const float*weight,
    float*y,
    int rows,
    int cols,
    float eps,
    cudaStream_t stream
);
torch::Tensor rmsnorm(
    torch::Tensor x,
    torch::Tensor weight,
    double eps
){
    c10::cuda::CUDAGuard devcie_guard(x.device());
    auto y = torch::empty_like(x);
    int rows = static_cast<int>(x.size(0));
    int cols = static_cast<int>(x.size(1));
    cudaStream_t stream =
    c10::cuda::getCurrentCUDAStream().stream();
    launch_rmsnorm(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        y.data_ptr<float>(),
        rows,
        cols,
        static_cast<float>(eps),
        stream
    );
    return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("rmsnorm",&rmsnorm);
}