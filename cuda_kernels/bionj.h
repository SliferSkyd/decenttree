#ifndef BIONJ_GPU_H
#define BIONJ_GPU_H

void launch_bionj_gpu(const float* h_dist, int N, 
                      GPUMergeEvent* h_history, 
                      float* final_dist, int* final_pair);

#endif /* BIONJ_GPU_H */