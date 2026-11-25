#include <cuda_runtime.h>
constexpr int BLOCK_SIZE = 128;
__global__ void rmsnorm_block_kernel(
    const float*x,
    const float*weight,
    float*y,
    int rows,
    int cols,
    float eps
){
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows){
        return;
    }
    __shared__ float
    partial_sums[BLOCK_SIZE];
    __shared__ float scale;
    float local_sum = 0.0f;
    for(int col = tid;col<cols;col+=blockDim.x){
        int index = row*cols +col;
        float value = x[index];
        local_sum += value*value;
    }
    partial_sums[tid] = local_sum;
    __syncthreads();
    for(int stride = BLOCK_SIZE / 2;stride > 0;stride /=2){
        if(tid<stride){
            partial_sums[tid]+=partial_sums[tid + stride];
        }
        __syncthreads();
    }
    if(tid==0){
        float mean_square = partial_sums[0] / cols;
        scale = rsqrtf(mean_square + eps);
    }
    __syncthreads();
    for(int col = tid;col<cols;col+=blockDim.x){
        int index = row*cols + col;
        y[index] = x[index] *scale *weight[col];
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
    rmsnorm_block_kernel<<<rows,BLOCK_SIZE,0,stream>>>(x,weight,y,rows,cols,eps);
}