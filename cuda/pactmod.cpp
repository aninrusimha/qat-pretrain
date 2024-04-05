#include <cuda_fp16.h>
#include <torch/extension.h>
#include <vector>
#include <cassert>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_CHECK
#endif

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor fwd_cuda(
    torch::Tensor const& input, 
    const float clip_val,
    const float clip_valn,
    const int num_bits);

std::vector<torch::Tensor> bwd_cuda(
    torch::Tensor const& output_grads, 
    torch::Tensor const& input,
    torch::Tensor const& clip_val,
    torch::Tensor const& clip_valn,
    const int num_bits);

torch::Tensor fwd(
    torch::Tensor const& input, 
    torch::Tensor const& clip_val,
    torch::Tensor const& clip_valn,
    const int num_bits){
  
  CHECK_INPUT(input);
  CHECK_INPUT(clip_val);
  CHECK_INPUT(clip_valn);

  AT_ASSERTM(clip_val.item<float>() > clip_valn.item<float>(), "expected clip_val > clip_valn");
  AT_ASSERTM(input.dim() == 3, "expected 3D input");
  AT_ASSERTM(input.dim() == 3, "expected 3D input");
  AT_ASSERTM(clip_val.dim() == 0, "expected scalar clip_val");
  AT_ASSERTM(clip_valn.dim() == 0, "expected scalar clip_valn");
  AT_ASSERTM((input.scalar_type() == at::ScalarType::Half) ||
	     (input.scalar_type() == at::ScalarType::BFloat16), 
      "Only fp16 and bf16 are supported");

  return fwd_cuda(input,  clip_val.item<float>(), clip_valn.item<float>(), num_bits);
}

std::vector<torch::Tensor> bwd(
    torch::Tensor const& output_grads, 
    torch::Tensor const& input,
    torch::Tensor const& clip_val,
    torch::Tensor const& clip_valn,
    const int num_bits) {

  CHECK_INPUT(output_grads);
  CHECK_INPUT(input);
  CHECK_INPUT(clip_val);
  CHECK_INPUT(clip_valn);
  
  AT_ASSERTM(clip_val.item<float>() > clip_valn.item<float>(), "expected clip_val > clip_valn");
  AT_ASSERTM(output_grads.dim() == 3, "expected 3D output grad");
  AT_ASSERTM(input.dim() == 3, "expected 3D input in bwd");
  AT_ASSERTM(clip_val.dim() == 0, "expected scalar clip_val");
  AT_ASSERTM(clip_valn.dim() == 0, "expected scalar clip_valn");

  AT_ASSERTM((output_grads.scalar_type() == at::ScalarType::Half) ||
	     (output_grads.scalar_type() == at::ScalarType::BFloat16), 
      "Only fp16 and bf16 are supported");
  AT_ASSERTM((input.scalar_type() == at::ScalarType::Half) ||
	     (input.scalar_type() == at::ScalarType::BFloat16), 
      "Only fp16 and bf16 are supported");

  return bwd_cuda(output_grads, input, clip_val, clip_valn, num_bits);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_quant", 
        &fwd,
	"Forward PACT+ Quantization (CUDA)");
  m.def("backward_quant", 
        &bwd,
	"Backward PACT+ Gradient Quantization (CUDA)");
}

