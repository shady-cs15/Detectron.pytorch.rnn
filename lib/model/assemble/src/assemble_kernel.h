#ifdef __cplusplus
extern "C" {
#endif

void assemble_engine_launcher(float* cur_prev_aff, float* feat, float* output, 
            float* masked_cpa, int pad, cudaStream_t stream,
            int H, int W, long D, long N);

#ifdef __cplusplus
}
#endif