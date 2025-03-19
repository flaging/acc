import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline

# https://github.com/triton-lang/triton/issues/5388
DEVICE = torch.device("cuda:0")

def torch_add(x : torch.Tensor,
              y : torch.Tensor):
  return x + y

@triton.jit
def triton_add_kernel(x_ptr,
               y_ptr,
               output_ptr,
               n_elements,
               BLOCK_SIZE: tl.constexpr):
  pid = tl.program_id(axis=0)
  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  mask = offsets < n_elements
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = x + y
  tl.store(output_ptr + offsets, output, mask=mask)

def triton_add(x : torch.Tensor,
        y : torch.Tensor):
  output = torch.empty_like(x)
  assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
  n_elements = output.numel()
  grid = lambda meta : (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
  triton_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
  return output

cuda_source = """
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
"""

cpp_source = """
torch::Tensor cuda_add_naive(torch::Tensor x, torch::Tensor y);
torch::Tensor cuda_add_packed(torch::Tensor x, torch::Tensor y);
"""
module = torch.utils.cpp_extension.load_inline(
             name="cuda_add",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=["cuda_add_naive", "cuda_add_packed"],
            # verbose=True,
        )
def cuda_add_naive(x : torch.Tensor,
             y : torch.Tensor):
  return module.cuda_add_naive(x, y)
def cuda_add_packed(x, y):
  return module.cuda_add_packed(x, y)

def check():
  torch.manual_seed(0)
  size = 98432
  x = torch.rand(size, device = DEVICE)
  y = torch.rand(size, device = DEVICE)
  output_torch = torch_add(x, y)
  output_triton = triton_add(x, y)
  output_cuda = cuda_add_naive(x, y)
  print(output_torch)
  print(output_triton)
  print(output_cuda)
  print("Max differences = {}, {}".format(
    torch.max(torch.abs(output_torch - output_triton)),
    torch.max(torch.abs(output_torch - output_cuda))))

check()

@triton.testing.perf_report(
  triton.testing.Benchmark(
    x_names = ['size'],
    x_vals = [2**i for i in range(12, 28, 1)],
    x_log = True,
    line_arg = 'provider',
    line_vals = ['triton', 'torch', 'cuda_naive', 'cuda_packed'],
    line_names = ['Triton', 'Torch', 'Cuda_naive', 'Cuda_packed'],
    styles = [('blue', '-'), ('green', '-'), ('red', '-'), ('blue', '--')],
    ylabel = 'GB/s',
    plot_name = 'vector_add_perf',
    args = {},
  )
)
def benchmark(size, provider):
  x = torch.rand(size, device=DEVICE, dtype=torch.float32)
  y = torch.rand(size, device=DEVICE, dtype=torch.float32)
  quantiles = [0.5, 0.2, 0.8]
  if provider=='torch':
    ms, min_ms, max_ms = triton.testing.do_bench(lambda : torch_add(x, y), quantiles=quantiles)
  if provider == 'triton':
    ms, min_ms, max_ms = triton.testing.do_bench(lambda : triton_add(x, y), quantiles=quantiles)
  if provider == 'cuda_naive':
    ms, min_ms, max_ms = triton.testing.do_bench(lambda : cuda_add_naive(x, y), quantiles = quantiles)
  if provider == 'cuda_packed':
    ms, min_ms, max_ms = triton.testing.do_bench(lambda : cuda_add_packed(x, y), quantiles = quantiles)
  gbps = lambda ms :  3* x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
  return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data = True, show_plots=True)