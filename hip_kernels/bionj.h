#ifndef HIP_KERNELS_BIONJ_H
#define HIP_KERNELS_BIONJ_H

/* Declared (with C linkage) in bionj_gpu.h, which is the only consumer.
 * Note: this header previously used the guard BIONJ_GPU_H, the same macro
 * bionj_gpu.h defines before including it, so its contents were always
 * skipped - and it would not have compiled if they had not been, because
 * GPUMergeEvent is declared in bionj_gpu.h. */

struct GPUMergeEvent;

extern "C"
void launch_bionj_gpu(const float* h_dist, const float* h_var, int N,
                      GPUMergeEvent* h_history,
                      float* final_d3, int* final_map3);

#endif /* HIP_KERNELS_BIONJ_H */
