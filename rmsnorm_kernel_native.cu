#include <cuda_runtime.h>
__global__ void rmsnorm_native_kernel(
    const float*x,
    const float*weight,
    float*y,
    int rows,
    int cols,
    float eps
){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows){
        return;
    }
    float sum_sq = 0.0f;
    for(int col = 0;col<cols;++col){
        float value = x[row * cols + col];
        sum_sq += value * value;
    }
    float scale = rsqrtf(sum_sq /cols +eps);
    for (int col = 0;col < cols;++col){
        int index = row *cols + col;
        y[index] = x[index] * scale *weight[col];
    }
}
void launch_rmsnorm(
    const float*x,
    const float*weight,
    float*y,
    int rows,
    int cols,
    float eps,
    cudaStream_t stream
){
    int threads = 128;
    int blocks = (rows + threads -1)/threads;
    rmsnorm_native_kernel<<<blocks,threads,0,stream>>>(
        x,weight,y,rows,cols,eps
    );
}