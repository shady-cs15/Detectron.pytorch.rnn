//#include "THCUNN.h"
//#include "THCHalf.h"
//#include "THCHalfAutoNumerics.cuh"
//#include <THC/THC.h>
#include "common.h"
#include "stdio.h"
#include "assemble_kernel.h"

__global__ void assemble_kernel(
            int n, int H, int W, int D, int pad,
            float *cur_prev_aff,
            float *feat,
            float *output,
            float *masked_cpa) 
  {
    // n = D*H*W
    CUDA_KERNEL_LOOP(index, n) {
      int HW = H*W;
      int d = index / HW;
      int loc = index % HW;
      int y = loc / W;
      int x = loc % W;
      float bound = 1e-7;

      // Init a mass counter
      float mass = 0.0;
      for (int i = -pad; i <= pad; i++){
        for (int j = -pad; j <= pad; j++){
          int prev_y = y + i;
          int prev_x = x + j;
          if (prev_y >= 0 && prev_y < H && prev_x >= 0 && prev_x < W)
          {
            int flat_idx = y * W * HW + x * HW + prev_y * W + prev_x; 
            float coef = cur_prev_aff[flat_idx];
            // mass += coef * coef;
            if (coef > 0) {
              mass += coef;
            }
          }
        }
      }
      // mass = sqrt(mass);

      float val = 0.0;
      if (mass > -bound && mass < bound) {
          // Avoid divided-by-0
          //mass = bound;
          int flat_idx = y * W * HW + x * HW + y * W + x; 
          int feat_flat_idx = d * HW + y * W + x;
	      val = feat[feat_flat_idx];
	      if (d == 0) {
          	masked_cpa[flat_idx] += 1.0;
          }
      } else {
	      // Looping the local region
	      for (int i = -pad; i <= pad; i++){
	        for (int j = -pad; j <= pad; j++){
	          int prev_y = y + i;
	          int prev_x = x + j;
	          if (prev_y >= 0 && prev_y < H && prev_x >= 0 && prev_x < W)
	          {
	            // Update output
	            int flat_idx = y * W * HW + x * HW + prev_y * W + prev_x; 
	            float a = cur_prev_aff[flat_idx];
	            if (a > 0) {
	              a = a / mass;
	              int feat_flat_idx = d * HW + prev_y * W + prev_x;
	              float fc = feat[feat_flat_idx];
	              val += a * fc;
	              // Update gradient
	              if (d == 0) { // The thread for the first dim is responsible for this
	                masked_cpa[flat_idx] += a;
	              }  
	            }
	          }
	        }
	      }
	  }


      // Get the right cell in the output
      int output_idx = d * HW + y * W + x;
      output[output_idx] = val;
    }
}

void assemble_engine_launcher(float* cur_prev_aff, float* feat, float* output, 
                      float* masked_cpa, int pad, cudaStream_t stream,
                      int H, int W, long D, long N) {
  
  // launch kernel
  int count;
  cudaError_t err;
  
  count = H * W * D;
  assemble_kernel<<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>
    (count, H, W, D, pad, cur_prev_aff, feat, output, masked_cpa);

  // check error
  err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }
}
