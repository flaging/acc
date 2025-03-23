import os
import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline

# remove warning on arch not set
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'

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
  compiled_kernel = triton_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
  # print(compiled_kernel.asm.keys())
  # print(compiled_kernel.asm["ptx"])
  return output

# https://github.com/pytorch/pytorch/blob/8bece886552e58b75a066226c1c7da7975d68ba6/test/test_cpp_extensions_jit.py#L293
module_load = torch.utils.cpp_extension.load(
            name="cuda_add",
            sources= [ '01.vector-add.cu'],
            #verbose=True,
            with_cuda = True,
            extra_cuda_cflags=["-O2"],
        )
def cuda_add_naive(x : torch.Tensor,
             y : torch.Tensor):
  return module_load.cuda_add_naive(x, y)
def cuda_add_packed(x, y):
  return module_load.cuda_add_packed(x, y)
def cuda_add_coarsened(x, y):
  return module_load.cuda_add_coarsened(x, y)

def check():
  torch.manual_seed(0)
  size = 98432
  x = torch.rand(size, device = DEVICE)
  y = torch.rand(size, device = DEVICE)
  results = {}
  reference = torch_add(x, y)
  results['triton'] = triton_add(x, y)
  results['cuda_naive'] = cuda_add_naive(x, y)
  results['cuda_packed'] = cuda_add_packed(x, y)
  results['cuda_coarsened'] = cuda_add_coarsened(x,y)

  for k, v in results.items():
    print("diff between torch and {} = {}".format(k,
      torch.max(torch.abs(reference - v))))

check()

@triton.testing.perf_report(
  triton.testing.Benchmark(
    x_names = ['size'],
    x_vals = [2**i for i in range(12, 28, 2)],
    x_log = True,
    line_arg = 'provider',
    line_vals = ['triton', 'torch', 'cuda_naive', 'cuda_packed', 'cuda_coarsed'],
    line_names = ['Triton', 'Torch', 'Cuda_naive', 'Cuda_packed', 'Cuda_coarsed'],
    styles = [('blue', '-'), ('green', '-'), ('red', '-'), ('red', '--'), ('red', ':')],
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
  if provider == 'cuda_coarsed':
    ms, min_ms, max_ms = triton.testing.do_bench(lambda : cuda_add_coarsened(x, y), quantiles = quantiles)
  gbps = lambda ms :  3* x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
  return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data = True, show_plots=True)