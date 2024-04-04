#include "ATen/ATen.h"
#include "ATen/AccumulateType.h"
#include "ATen/cuda/Atomic.cuh"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/DeviceUtils.cuh"
#include <torch/extension.h>
#include <stdio.h>

#include "type_shim.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <c10/macros/Macros.h>


template <typename Datatype, int ELEMENTS_PER_LDG>
__device__ __inline__ void copy_vector(Datatype *dst, const Datatype *src);

template <>
__device__ __inline__ void copy_vector<c10::BFloat16, 1>(c10::BFloat16 *dst, const c10::BFloat16 *src) { *dst = *src; }

template <>
__device__ __inline__ void copy_vector<c10::BFloat16, 4>(c10::BFloat16 *dst, const c10::BFloat16 *src) { *((float2*) dst) = *((float2*) src); }

template <> __device__ __inline__ void copy_vector<c10::Half, 1>(c10::Half *dst, const c10::Half *src) { *dst = *src; }

template <>
__device__ __inline__ void copy_vector<c10::Half, 4>(c10::Half *dst, const c10::Half *src) { *((float2*) dst) = *((float2*) src); }

template <>
__device__ __inline__ void copy_vector<uint8_t, 1>(uint8_t *dst, const uint8_t *src) { *dst = *src; }

template <>
__device__ __inline__ void copy_vector<uint8_t, 4>(uint8_t *dst, const uint8_t *src) {*((half2*) dst) = *((half2*) src); }

template <typename T>
__device__ __inline__ T clamp(T val, T clip_val, T clip_valn) {
    return val > clip_valn ? (val < clip_val ? val : clip_val) : clip_valn;
}

int log2_ceil(int value) {
    int log2_value = 0;
    while ((1 << log2_value) < value) ++log2_value;
    return log2_value;
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN_NATIVE(T value, int laneMask, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
    return __shfl_down_sync(mask, value, laneMask, width);
#else
    return __shfl_down(value, laneMask, width);
#endif
}
__device__ __forceinline__ c10::Half WARP_SHFL_DOWN_NATIVE(c10::Half value, int laneMask, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
    return __shfl_down_sync(mask, *reinterpret_cast<__half *>(&value), laneMask, width);
#else
    return __shfl_down(*reinterpret_cast<__half *>(&value), laneMask, width);
#endif
}

//Goal: make this function only execute on smaller lanes not larger lanes
//Do this by replacing xor shuffle with down shuffle
template <typename acc_t, int WARP_SIZE>
__device__ __forceinline__ void warp_reduce_double_sum(acc_t* sums) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sums[0] += WARP_SHFL_DOWN_NATIVE(sums[0], offset, WARP_SIZE);
        sums[1] += WARP_SHFL_DOWN_NATIVE(sums[1], offset, WARP_SIZE);
    }
}

//https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
template <typename acc_t, int WARP_SIZE, int WARPS_PER_BLOCK>
__device__ __forceinline__ void block_reduce_double_sum(acc_t* sums){
    static __shared__ int shared[WARPS_PER_BLOCK * 2];
    int tid = threadIdx.x; //thread id in warp
    int wid = threadIdx.y; //warp id in block
    warp_reduce_double_sum<acc_t, WARP_SIZE>(sums);
    if (tid == 0){
        shared[wid * 2] = sums[0];
        shared[wid * 2 + 1] = sums[1];
    }
    __syncthreads();
    if (wid * WARPS_PER_BLOCK + tid < WARPS_PER_BLOCK){
        sums[0] = shared[tid * 2];
        sums[1] = shared[tid * 2 + 1];
    }
    else{
        sums[0] = 0;
        sums[1] = 0;
    }
    // sums[0] = (wid * WARP_SIZE + tid < WARPS_PER_BLOCK) ? shared[tid] : 0;
    // sums[1] = (wid * WARP_SIZE + tid < WARPS_PER_BLOCK) ? shared[tid + WARP_SIZE] : 0;
    if (wid == 0){
        warp_reduce_double_sum<acc_t, WARP_SIZE>(sums);
    }
}

template<typename T, int log2_elements>
__global__ void quant_warp_forward(
    T *out,
    const T *inp,
    const T cv,
    const T cvn,
    const T scale,
    const T descale,
    const T zp,
    const int batches,
    const int seq_len,
    const int dim){
        /* template for this function:
         * figure out which token you need to quantize
         * load
         * quantize that token, using clip_val and clip_valn
         * dequantize that token
         * store
         * return
        */
        constexpr int next_power_of_two = 1 << log2_elements;
        constexpr int WARP_SIZE = (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
        constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
        // constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;
        constexpr int ELEMENTS_PER_LDG_STG = (WARP_ITERATIONS < 4) ? 1 : 4; // Should always be four, given our dim sizes
        // constexpr int warps_per_block = blockDim.y;

        // blockDim/threadIdx = (WARP_SIZE, WARPS_PER_BLOCK,1 )
        // gridDim/blockIdx = (batches, seq_len / warps_per_block, 1) 
        // each warp takes a token
        //const int first_token = blockDim.y * (blockIdx.x + gridDim.y * (blockIdx.y + gridDim.x)) + threadIdx.y;
        const int first_token = blockDim.y * (blockIdx.y + gridDim.y * blockIdx.x ) + threadIdx.y;
        const int local_idx = threadIdx.x;
        // we have no real notion of batches beyond organization
        inp += first_token * dim + ELEMENTS_PER_LDG_STG * local_idx;
        out += first_token * dim + ELEMENTS_PER_LDG_STG * local_idx;

        // printf("%d, %d\n", first_token, local_idx);

        T temp_data[ELEMENTS_PER_LDG_STG];
        #pragma unroll
        for (int i = 0; i < WARP_ITERATIONS; i+=ELEMENTS_PER_LDG_STG) {
            int elem_idx = ELEMENTS_PER_LDG_STG * local_idx + i * WARP_SIZE;
            if (elem_idx < dim){
                int itr_offset = i*WARP_SIZE;
                copy_vector<T, ELEMENTS_PER_LDG_STG>(temp_data, inp + itr_offset);
                copy_vector<T, ELEMENTS_PER_LDG_STG>(out + itr_offset, temp_data);
                #pragma unroll
                // quantize and dequantize
                for (int elem = 0; elem < ELEMENTS_PER_LDG_STG; ++elem){
                    T prequant = clamp<T>(temp_data[elem], cv, cvn) * scale - zp ;
                    T quant = hrint(prequant);
                    T dequant = (quant + zp) * descale;
                    temp_data[elem] = dequant;
                }
                copy_vector<T, ELEMENTS_PER_LDG_STG>(out + itr_offset, temp_data);
            }
        }
    }

template<typename T, int log2_elements>
__global__ void quant_warp_backward(
    const T *grad_output,
    const T *inp,
    const T *clip_val,
    const T *clip_valn,
    const T n_levels,
    const T inv_levels,
    T *input_grad,
    T *cv_grad,
    T *cvn_grad,
    const int seq_len,
    const int dim){
        /* template for this function:
         * figure out which token you need to compute grads for
         * load grad_output, inp, and cv
         * get grad_input for that token
         * dequantize that token
         * store input_grad
         * aggregate g_cv and g_cvn
         * write
         * aggregate across SM
         * atomc add g_cv, g_cvn
         * return grads
         */
        
        constexpr int next_power_of_two = 1 << log2_elements;
        constexpr int WARP_SIZE = (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
        constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
        // constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;
        constexpr int ELEMENTS_PER_LDG_STG = (WARP_ITERATIONS < 4) ? 1 : 4; // Should always be four, given our dim sizes
        constexpr int WARPS_PER_BLOCK = (128 / WARP_SIZE);

        // blockDim/threadIdx = (WARP_SIZE, WARPS_PER_BLOCK,1 )
        // gridDim/blockIdx = (batches, seq_len / warps_per_block, 1) 
        // max 4 * (23 + 256 * ())
        const int first_token = blockDim.y * (blockIdx.y + gridDim.y * blockIdx.x ) + threadIdx.y;
        const int local_idx = threadIdx.x;
        // we have no real notion of batches beyond organization
        grad_output += first_token * dim + ELEMENTS_PER_LDG_STG * local_idx;
        inp += first_token * dim + ELEMENTS_PER_LDG_STG * local_idx;
        input_grad += first_token * dim + ELEMENTS_PER_LDG_STG * local_idx;
        T cv = *clip_val;
        T cvn = *clip_valn;
        // printf("clip_val%f\n", __half2float(cv));
        // printf("clip_valn%f\n", __half2float(cvn));
        T scale = n_levels / (cv -cvn);
        T inp_buffer[ELEMENTS_PER_LDG_STG];
        T g_o_buffer[ELEMENTS_PER_LDG_STG];
        T g_i_buffer[ELEMENTS_PER_LDG_STG];
        T g_cvs[2] {0.0f};

        #pragma unroll
        for (int i = 0; i < WARP_ITERATIONS; i+=ELEMENTS_PER_LDG_STG) {
            int elem_idx = ELEMENTS_PER_LDG_STG * local_idx + i * WARP_SIZE;
            if (elem_idx < dim){
                int itr_offset = i*WARP_SIZE;
                copy_vector<T, ELEMENTS_PER_LDG_STG>(inp_buffer, inp + itr_offset);
                copy_vector<T, ELEMENTS_PER_LDG_STG>(g_o_buffer, grad_output + itr_offset);
                #pragma unroll
                for (int elem = 0; elem < ELEMENTS_PER_LDG_STG; ++elem){
                    T inp_val = inp_buffer[elem];
                    // if beyond cv bounds, apply gradient to clip_values
                    if (inp_val >= cv){
                        g_cvs[0] += g_o_buffer[elem];
                        g_i_buffer[elem] = CUDART_ZERO_FP16;
                    }
                    else if (inp_val <= cvn){
                        g_cvs[1] += g_o_buffer[elem];
                        g_i_buffer[elem] = CUDART_ZERO_FP16;
                    }
                    else{
                        // otherwise, pass gradient straight through
                        T out = g_o_buffer[elem];
                        g_i_buffer[elem] = out;
                        T prequant = clamp<T>(inp_val - cvn, cv, cvn) * scale;
                        // in addition, apply rounding error gradient to clip_values
                        T quant = hrint(prequant); 
                        T delz = (prequant - quant) * inv_levels;
                        g_cvs[0] += -delz * out;
                        g_cvs[1] += delz * out;
                    }
                    // temp_data[elem] = (quant i zp)
                }
                copy_vector<T, ELEMENTS_PER_LDG_STG>(input_grad + itr_offset, g_i_buffer);
            }
        }
        // per warp accumulation (slower)

        /* warp_reduce_double_sum<T, WARP_SIZE>(g_cvs);
        *  if (local_idx == 0 && threadIdx.y == 0){
        *      gpuAtomicAdd(cv_grad, g_cvs[0]);
        *      gpuAtomicAdd(cvn_grad, g_cvs[1]);
        *  }
        *  warp_reduce_double_sum<T, WARP_SIZE>(g_cvs);
        *  if (local_idx == 0){
        */

        // per block / SM accumulation

        block_reduce_double_sum<T, WARP_SIZE, WARPS_PER_BLOCK>(g_cvs);
        if (local_idx == 0 && threadIdx.y == 0){
            
            gpuAtomicAdd(cv_grad, g_cvs[0]);
            gpuAtomicAdd(cvn_grad, g_cvs[1]);
        }
    }

template<typename T>
void dispatch_quant_forward(
    T *out,
    const T *inp,
    const float clip_val,
    const float clip_valn,
    const int batch_size,
    const int seq_len,
    const int dim,
    const int num_bits){
        /* template for this function:
         * asserts and checks
         * divide the work so that each warp handles quantizing a single token
        */
        // moved below assert to fwd, as can't do pointer stuff here
        // TORCH_INTERNAL_ASSERT(clip_val > clip_valn);
        TORCH_INTERNAL_ASSERT(num_bits > 0);
        // This number can be much higher. TODO: test the limit 
        TORCH_INTERNAL_ASSERT(dim <= 16384);

        // int warp_size = C10_WARP_SIZE;
        // int threads_per_block = 128;
        int threads_per_block = 128;

        // int batches_per_warp = warps_per_block;
        
        const int n_levels = pow(2, num_bits) - 1;

        const T cv = clip_val; //clip_val[0];
        // const T cvn = clip_valn[0];
        const T cvn = clip_valn; //clip_val[0];
        const T scale = n_levels / (cv - cvn);
        const T descale = 1 / scale;
        const T zp = scale * cvn;
        // printf("%f, %f, %f, %f, %f\n", __half2float(cv), clip_val, __half2float(scale), __half2float(zp), __half2float(descale));
        // warps_per_block is batches per block
        // const int batches = batch_size * seq_len / warps_per_block;
        const int log2_elements = log2_ceil(dim);
        const int next_power_of_two = 1 << log2_elements;
        const int warp_size = (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
        int warps_per_block = (threads_per_block / warp_size);
        // printf("%d, %d, %d\n", warp_size, next_power_of_two, warps_per_block);
        
        dim3 blocks(1, batch_size* seq_len / warps_per_block, 1);
        dim3 threads(warp_size, warps_per_block, 1);


        switch (log2_elements) {
            case 1: // 2
                quant_warp_forward<T, 1>
                  <<<blocks,  threads, 0, at::cuda::getCurrentCUDAStream()>>>(out, inp, cv, cvn, scale, descale, zp, batch_size, seq_len, dim);
                  break;
            case 2: // 4
                quant_warp_forward<T, 2>
                  <<<blocks,  threads, 0, at::cuda::getCurrentCUDAStream()>>>(out, inp, cv, cvn, scale, descale, zp, batch_size, seq_len, dim);
                  break;
            case 3: // 8
                quant_warp_forward<T, 3>
                  <<<blocks,  threads, 0, at::cuda::getCurrentCUDAStream()>>>(out, inp, cv, cvn, scale, descale, zp, batch_size, seq_len, dim);
                  break;
            case 4: // 16
                quant_warp_forward<T, 4>
                  <<<blocks,  threads, 0, at::cuda::getCurrentCUDAStream()>>>(out, inp, cv, cvn, scale, descale, zp, batch_size, seq_len, dim);
                  break;
            case 5: // 32
                quant_warp_forward<T, 5>
                  <<<blocks,  threads, 0, at::cuda::getCurrentCUDAStream()>>>(out, inp, cv, cvn, scale, descale, zp, batch_size, seq_len, dim);
                  break;
            case 6: // 64
                quant_warp_forward<T, 6>
                  <<<blocks,  threads, 0, at::cuda::getCurrentCUDAStream()>>>(out, inp, cv, cvn, scale, descale, zp, batch_size, seq_len, dim);
                  break;
            case 7: // 128
                quant_warp_forward<T, 7>
                  <<<blocks,  threads, 0, at::cuda::getCurrentCUDAStream()>>>(out, inp, cv, cvn, scale, descale, zp, batch_size, seq_len, dim);
                  break;
            case 8: // 256
                quant_warp_forward<T, 8>
                  <<<blocks,  threads, 0, at::cuda::getCurrentCUDAStream()>>>(out, inp, cv, cvn, scale, descale, zp, batch_size, seq_len, dim);
                  break;
            case 9: // 512
                quant_warp_forward<T, 9>
                  <<<blocks,  threads, 0, at::cuda::getCurrentCUDAStream()>>>(out, inp, cv, cvn, scale, descale, zp, batch_size, seq_len, dim);
                  break;
            case 10: // 1024
                quant_warp_forward<T, 10>
                  <<<blocks,  threads, 0, at::cuda::getCurrentCUDAStream()>>>(out, inp, cv, cvn, scale, descale, zp, batch_size, seq_len, dim);
                  break;
            case 11: // 2048
                quant_warp_forward<T, 11>
                  <<<blocks,  threads, 0, at::cuda::getCurrentCUDAStream()>>>(out, inp, cv, cvn, scale, descale, zp, batch_size, seq_len, dim);
                  break;
            case 12: // 4096
                quant_warp_forward<T, 12>
                  <<<blocks,  threads, 0, at::cuda::getCurrentCUDAStream()>>>(out, inp, cv, cvn, scale, descale, zp, batch_size, seq_len, dim);
                  break;
            case 13: // 8192
                quant_warp_forward<T, 13>
                  <<<blocks,  threads, 0, at::cuda::getCurrentCUDAStream()>>>(out, inp, cv, cvn, scale, descale, zp, batch_size, seq_len, dim);
                  break;
            case 14: // 16384
                quant_warp_forward<T, 14>
                  <<<blocks,  threads, 0, at::cuda::getCurrentCUDAStream()>>>(out, inp, cv, cvn, scale, descale, zp, batch_size, seq_len, dim);
                  break;
            default:
                break;
        }  
    }

template<typename T>
void dispatch_quant_backward (
    const T *output_grad,
    const T *input,
    const T *clip_val,
    const T *clip_valn,
    T *input_grad,
    T *clip_val_grad,
    T *clip_valn_grad,
    const int batch_size,
    const int seq_len,
    const int dim,
    const int num_bits
    ){
        /* template for this function
         * asserts and checks
        */
        TORCH_INTERNAL_ASSERT(num_bits > 0);

        int warp_size = C10_WARP_SIZE;
        int threads_per_block = 128;
        // int threads_per_block = 128;

        int warps_per_block = (threads_per_block / warp_size);
        // int batches_per_warp = warps_per_block;
        
        const int n_levels = pow(2, num_bits) - 1;
        // const T cv = *clip_val;
        // const T cvn = *clip_valn;
        // const T scale = n_levels / (cv - cvn);
        const T inv_levels = 1.0 / n_levels;
        // warps_per_block is batches per block
        // const int batches = batch_size * seq_len / warps_per_block;
        
        dim3 blocks(1 , batch_size*  seq_len / warps_per_block, 1);
        dim3 threads(warp_size, warps_per_block, 1);

        const int log2_elements = log2_ceil(dim);
        switch (log2_elements) {
            case 1: // 2
                quant_warp_backward<T, 1>
                  <<<blocks,  threads, 0, at::cuda::getCurrentCUDAStream()>>>(output_grad, input, clip_val, clip_valn, n_levels, inv_levels, input_grad, clip_val_grad, clip_valn_grad, seq_len, dim);
                  break;
            case 2: // 4
                quant_warp_backward<T, 2>
                  <<<blocks,  threads, 0, at::cuda::getCurrentCUDAStream()>>>(output_grad, input, clip_val, clip_valn, n_levels, inv_levels, input_grad, clip_val_grad, clip_valn_grad, seq_len, dim);
                  break;
            case 3: // 8
                quant_warp_backward<T, 3>
                  <<<blocks,  threads, 0, at::cuda::getCurrentCUDAStream()>>>(output_grad, input, clip_val, clip_valn, n_levels, inv_levels, input_grad, clip_val_grad, clip_valn_grad, seq_len, dim);
                  break;
            case 4: // 16
                quant_warp_backward<T, 4>
                  <<<blocks,  threads, 0, at::cuda::getCurrentCUDAStream()>>>(output_grad, input, clip_val, clip_valn, n_levels, inv_levels, input_grad, clip_val_grad, clip_valn_grad, seq_len, dim);
                  break;
            case 5: // 32
                quant_warp_backward<T, 5>
                  <<<blocks,  threads, 0, at::cuda::getCurrentCUDAStream()>>>(output_grad, input, clip_val, clip_valn, n_levels, inv_levels, input_grad, clip_val_grad, clip_valn_grad, seq_len, dim);
                  break;
            case 6: // 64
                quant_warp_backward<T, 6>
                  <<<blocks,  threads, 0, at::cuda::getCurrentCUDAStream()>>>(output_grad, input, clip_val, clip_valn, n_levels, inv_levels, input_grad, clip_val_grad, clip_valn_grad, seq_len, dim);
                  break;
            case 7: // 128
                quant_warp_backward<T, 7>
                  <<<blocks,  threads, 0, at::cuda::getCurrentCUDAStream()>>>(output_grad, input, clip_val, clip_valn, n_levels, inv_levels, input_grad, clip_val_grad, clip_valn_grad, seq_len, dim);
                  break;
            case 8: // 256
                quant_warp_backward<T, 8>
                  <<<blocks,  threads, 0, at::cuda::getCurrentCUDAStream()>>>(output_grad, input, clip_val, clip_valn, n_levels, inv_levels, input_grad, clip_val_grad, clip_valn_grad, seq_len, dim);
                  break;
            case 9: // 512
                quant_warp_backward<T, 9>
                  <<<blocks,  threads, 0, at::cuda::getCurrentCUDAStream()>>>(output_grad, input, clip_val, clip_valn, n_levels, inv_levels, input_grad, clip_val_grad, clip_valn_grad, seq_len, dim);
                  break;
            case 10: // 1024
                quant_warp_backward<T, 10>
                  <<<blocks,  threads, 0, at::cuda::getCurrentCUDAStream()>>>(output_grad, input, clip_val, clip_valn, n_levels, inv_levels, input_grad, clip_val_grad, clip_valn_grad, seq_len, dim);
                  break;
            case 11: // 2048
                quant_warp_backward<T, 11>
                  <<<blocks,  threads, 0, at::cuda::getCurrentCUDAStream()>>>(output_grad, input, clip_val, clip_valn, n_levels, inv_levels, input_grad, clip_val_grad, clip_valn_grad, seq_len, dim);
                  break;
            case 12: // 4096
                quant_warp_backward<T, 12>
                  <<<blocks,  threads, 0, at::cuda::getCurrentCUDAStream()>>>(output_grad, input, clip_val, clip_valn, n_levels, inv_levels, input_grad, clip_val_grad, clip_valn_grad, seq_len, dim);
                  break;
            case 13: // 8192
                quant_warp_backward<T, 13>
                  <<<blocks,  threads, 0, at::cuda::getCurrentCUDAStream()>>>(output_grad, input, clip_val, clip_valn, n_levels, inv_levels, input_grad, clip_val_grad, clip_valn_grad, seq_len, dim);
                  break;
            case 14: // 8192
                quant_warp_backward<T, 14>
                  <<<blocks,  threads, 0, at::cuda::getCurrentCUDAStream()>>>(output_grad, input, clip_val, clip_valn, n_levels, inv_levels, input_grad, clip_val_grad, clip_valn_grad, seq_len, dim);
                  break;
            default:
                break;  
        }
        // cudaFree(write_array);
    }

torch::Tensor fwd_cuda(
    torch::Tensor const& input, 
    const float clip_val,
    const float clip_valn,
    const int num_bits){
        const int batch_size = input.size(0);
        const int seq_len = input.size(1);
        const int dim = input.size(2);
        auto output = torch::zeros_like(input);

        void* output_ptr = static_cast<void*>(output.data_ptr());
        void* input_ptr = static_cast<void*>(input.data_ptr());

        DISPATCH_HALF(input.scalar_type(), "dispatch_quant_forward_cuda", 
              dispatch_quant_forward<scalar_t> (
                reinterpret_cast<scalar_t*>(output_ptr),
                reinterpret_cast<const scalar_t*>(input_ptr),
                clip_val,
                clip_valn,
                batch_size,
                seq_len,
                dim,
                num_bits);
          );

        return output; 
    }
    // live dequantize, no 

std::vector<torch::Tensor> bwd_cuda(
    torch::Tensor const& output_grads, 
    torch::Tensor const& input,
    // const float clip_val,
    // const float clip_valn,
    torch::Tensor const& clip_val,
    torch::Tensor const& clip_valn,
    const int num_bits){
        const int batch_size = input.size(0);
        const int seq_len = input.size(1);
        const int dim = input.size(2);
        // split work such that 

        // ciel division
        // const unsigned int num_blocks  = (size + threads - 1) / threads;

        void* output_grad_ptr = static_cast<void*>(output_grads.data_ptr());
        void* input_ptr = static_cast<void*>(input.data_ptr());
        void* clip_val_ptr = static_cast<void*>(clip_val.data_ptr());
        void* clip_valn_ptr = static_cast<void*>(clip_valn.data_ptr());
        
        auto grad_input = torch::zeros_like(input);
        auto grad_clip_val = torch::zeros_like(clip_val);
        auto grad_clip_valn = torch::zeros_like(clip_valn);

        void* grad_input_ptr = static_cast<void*>(grad_input.data_ptr());
        void* clip_val_grad_ptr = static_cast<void*>(grad_clip_val.data_ptr());
        void* clip_valn_grad_ptr = static_cast<void*>(grad_clip_valn.data_ptr());

        DISPATCH_HALF(input.scalar_type(), "dispatch_quant_backward_cuda", 
              dispatch_quant_backward<scalar_t> (
                reinterpret_cast<const scalar_t*>(output_grad_ptr),
                reinterpret_cast<const scalar_t*>(input_ptr),
                reinterpret_cast<const scalar_t*>(clip_val_ptr),
                reinterpret_cast<const scalar_t*>(clip_valn_ptr),
                reinterpret_cast<scalar_t*>(grad_input_ptr),
                reinterpret_cast<scalar_t*>(clip_val_grad_ptr),
                reinterpret_cast<scalar_t*>(clip_valn_grad_ptr),
                batch_size,
                seq_len,
                dim,
                num_bits
               ) ;
          );
        return {grad_input, grad_clip_val, grad_clip_valn}; 
    }