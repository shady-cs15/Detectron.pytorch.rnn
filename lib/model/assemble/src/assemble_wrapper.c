#include <THC/THC.h>
#include <stdio.h>
#include "common.h"
#include "assemble_kernel.h"

extern THCState* state;

int gpu_assemble(THFloatTensor* cur_prev_aff,
		            THFloatTensor* feat,
		            THFloatTensor* output,
		            THFloatTensor* masked_cpa,
		            int pad)
{
  //bool DEBUG = false;

  
  float * data_cpa = THFloatTensor_data(cur_prev_aff);
  float * data_feat = THFloatTensor_data(feat);
  float * data_output = THFloatTensor_data(output);
  float * data_masked_cpa = THFloatTensor_data(masked_cpa);

  int H = THFloatTensor_size(cur_prev_aff, 0);
  int W = THFloatTensor_size(cur_prev_aff, 1);
  int D = THFloatTensor_size(feat, 0);
  int N = THFloatTensor_size(feat, 1);
  
  output = THFloatTensor_newContiguous(output);
  masked_cpa = THFloatTensor_newContiguous(masked_cpa);

  THFloatTensor_zero(output);
  THFloatTensor_zero(masked_cpa);
  
  cudaStream_t stream = THCState_getCurrentStream(state);

  assemble_engine_launcher(data_cpa, data_feat, data_output, data_masked_cpa, pad,
                            stream, H, W, D, N);

  return 1;
}

