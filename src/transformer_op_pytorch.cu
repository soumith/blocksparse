#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h> 

#include <vector>

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// typedef struct __align__(2) ehalf {
//   __device__ __forceinline__ ehalf() {}
//   __device__ __forceinline__ ehalf(const unsigned short val) : x(val) {}
//   unsigned short x;
// } ehalf;

#include "ew_op_gpu.h"

template <typename TL>
bool SoftmaxCrossEntropy(CUstream stream, ehalf* grad, float* loss, const ehalf* logits, const TL* labels, uint N, uint K);
bool SoftmaxCrossEntropyGrad(CUstream stream, uint SMs, ehalf* dx, const float* dy, const ehalf* y, uint NK, uint K);

bool SoftmaxCrossEntropy_forward(
    torch::Tensor grad,
    torch::Tensor loss,
    torch::Tensor logits,
    torch::Tensor labels,
    uint N,
    uint K) {
  CHECK_INPUT(grad);
  CHECK_INPUT(loss);
  CHECK_INPUT(logits);
  CHECK_INPUT(labels);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // at::Half can freely be converted to ehalf because they're both backed by the struct __half
  ehalf* grad_data = reinterpret_cast<ehalf*>(grad.data<at::Half>());
  float* loss_data = loss.data<float>();
  ehalf* logits_data = reinterpret_cast<ehalf*>(logits.data<at::Half>());
  int* labels_data = labels.data<int>();
  // bool SoftmaxCrossEntropy(CUstream stream, ehalf* grad, float* loss, const ehalf* logits, const TL* labels, uint N, uint K)
  auto errcode = SoftmaxCrossEntropy<int>(stream, grad_data, loss_data, logits_data, labels_data, N, K);
  THCudaCheck(cudaGetLastError()); // catch launch errors
  return errcode;
}

bool SoftmaxCrossEntropy_backward(
    torch::Tensor dx,
    torch::Tensor dy,
    torch::Tensor y,
    uint NK,
    uint K) {
  CHECK_INPUT(dx);
  CHECK_INPUT(dy);
  CHECK_INPUT(y);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  uint SMs = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  ehalf* dx_data = reinterpret_cast<ehalf*>(dx.data<at::Half>());
  float* dy_data = dy.data<float>();
  ehalf* y_data = reinterpret_cast<ehalf*>(y.data<at::Half>());
  // bool SoftmaxCrossEntropyGrad(CUstream stream, uint SMs, ehalf* dx, const float* dy, const ehalf* y, uint NK, uint K);
  auto errcode = SoftmaxCrossEntropyGrad(stream, SMs, dx_data, dy_data, y_data, NK, K);
  THCudaCheck(cudaGetLastError()); // catch launch errors
  return errcode;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &SoftmaxCrossEntropy_forward, "SOFTMAXCROSSENTROPY forward (CUDA)");
  m.def("backward", &SoftmaxCrossEntropy_backward, "SOFTMAXCROSSENTROPY backward (CUDA)");
}
