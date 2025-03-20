#include <torch/extension.h>

#include <ATen/cuda/Exceptions.h> // for CUDNN_CHECK
#include <ATen/cudnn/Descriptors.h> // for TensorDescriptor
#include <ATen/cudnn/Handle.h> // for getCudnnHandle

template <typename T>
__global__ void cuda_add_kernel(T* x_ptr, T* y_ptr, T* output_ptr, int n_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_elements) {
    output_ptr[idx] = x_ptr[idx] + y_ptr[idx];
  }
}

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
template <typename T>
__global__ void cuda_add_packed_kernel(T* x_ptr, T* y_ptr, T* output_ptr, int n_elements) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < n_elements) {
    float4 x4 = FETCH_FLOAT4(x_ptr[idx]);
    float4 y4 = FETCH_FLOAT4(y_ptr[idx]);
    float4 output4;
    output4.x = x4.x + y4.x;
    output4.y = x4.y + y4.y;
    output4.z = x4.z + y4.z;
    output4.w = x4.w + y4.w;
    FETCH_FLOAT4(output_ptr[idx]) = output4;
  }
}

template <typename T>
__global__ void cuda_add_coarsened_kernel(T* x_ptr, T* y_ptr, T* output_ptr, int n_elements, int BLOCK_SIZE, int factor) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * factor;
  if (idx + factor * BLOCK_SIZE < n_elements) {
    for(int i= 0; i < factor * BLOCK_SIZE; i += BLOCK_SIZE) {
      output_ptr[idx + i] = x_ptr[idx+i] + y_ptr[idx + i];
    }
  }
}

torch::Tensor cuda_add_naive(torch::Tensor x, torch::Tensor y) {
  const int BLOCK_SIZE = 1024;
  torch::Tensor output = torch::zeros_like(x);
  int n_elements = x.numel();
  int grid_size = (n_elements + BLOCK_SIZE -1) / BLOCK_SIZE;
  cuda_add_kernel<float><<<grid_size, BLOCK_SIZE>>>(x.data_ptr<float>(), y.data_ptr<float>(), output.data_ptr<float>(), n_elements);
  return output;
}

torch::Tensor cuda_add_packed(torch::Tensor x, torch::Tensor y) {
  const int BLOCK_SIZE = 1024;
  torch::Tensor output = torch::zeros_like(x);
  int n_elements = x.numel();
  int grid_size = (n_elements + BLOCK_SIZE * 4 -1) / BLOCK_SIZE / 4;
  cuda_add_packed_kernel<float><<<grid_size, BLOCK_SIZE>>>(x.data_ptr<float>(), y.data_ptr<float>(), output.data_ptr<float>(), n_elements);
  return output;
}

torch::Tensor cuda_add_coarsened(torch::Tensor x, torch::Tensor y) {
  const int BLOCK_SIZE = 1024;
  torch::Tensor output = torch::zeros_like(x);
  int n_elements = x.numel();
  int factor = 8;
  int grid_size = (n_elements + BLOCK_SIZE * factor -1) / BLOCK_SIZE / factor;
  cuda_add_coarsened_kernel<float><<<grid_size, BLOCK_SIZE>>>(x.data_ptr<float>(), y.data_ptr<float>(), output.data_ptr<float>(), n_elements, BLOCK_SIZE, factor);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Use the same name as the check functions so error messages make sense
  m.def("cuda_add_coarsened", &cuda_add_coarsened, "cuda_add_coarsened");
  m.def("cuda_add_packed", &cuda_add_packed, "cuda_add_packed");
  m.def("cuda_add_naive", &cuda_add_naive, "cuda_add_naive");
}