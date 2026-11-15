#include <cuda_runtime.h>
#include <cstdio>
__global__ void rmsnorm_native_kernel(const float*x,const float*weight,float*y,int rows,int cols,float eps){
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if(row >= rows){
        return;
    }
    float sum_sq = 0.0f;
    for(int col = 0;col<cols;++col){
        float value = x[row*cols + col];
        sum_sq += value * value;
    }
    float scale = rsqrtf(sum_sq / cols + eps);
    for (int col = 0;col<cols;++col){
        int index = row*cols+col;
        y[index] = x[index]*scale*weight[col];
    }
}
int main(){
    const int rows = 2;
    const int cols = 4;
    const float eps = 1e-6f;
    float h_x[rows*cols] = {
        1.0f,2.0f,3.0f,4.0f,
        2.0f,0.0f,0.0f,0.0f
    };
    float h_weight[cols] = {
        1.0f,2.0f,1.0f,0.5f
    };
    float h_y[rows * cols] = {};
    float*d_x = nullptr;
    float*d_weight = nullptr;
    float*d_y = nullptr;
    cudaMalloc(&d_x,sizeof(h_x));
    cudaMalloc(&d_weight,sizeof(h_weight));
    cudaMalloc(&d_y,sizeof(h_y));
    cudaMemcpy(d_x,h_x,sizeof(h_x),
    cudaMemcpyHostToDevice
);
    cudaMemcpy(
        d_weight,h_weight,sizeof(h_weight),
        cudaMemcpyHostToDevice
    );
    int threads = 128;
    int blocks = (rows + threads-1)/ threads;
    rmsnorm_native_kernel<<<blocks,threads>>>(
        d_x,d_weight,d_y,rows,cols,eps
    );
    cudaDeviceSynchronize();
    cudaMemcpy(
        h_y,d_y,sizeof(h_y),
        cudaMemcpyDeviceToHost
    );
    std::printf("cuda rmsnorm output:\n");
    for (int row = 0;row<rows;++row){
        for(int col = 0;col<cols;++col){
            std::printf("%.6f ", h_y[row*cols+col]);
        }
        std::printf("\n");
    }
    cudaFree(d_x);
    cudaFree(d_weight);
    cudaFree(d_y);
    return 0;
}