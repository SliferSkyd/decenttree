#include "hip/hip_runtime.h"
#include <cfloat>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <utils/timeutil.h>

#define BLOCK_SIZE 256
#define REDUCE_THREADS 256

#define CHECK_HIP(call) { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "CUDA Error: %s (Line %d)\n", hipGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

struct GPUMergeEvent {
    int node_i;
    int node_j;
    float branch_i;
    float branch_j;
};

struct UpdateParams {
    int i_real;
    int j_real;
    int i_curr;
    int j_curr;
    float lambda;
    float mu;
    float d_corr;
    float v_corr;
};

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float shared[64]; // Hỗ trợ tối đa 64 warps (2048 threads)
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;

    if (wid == 0) val = warpReduceSum(val);

    return val;
}

__device__ __forceinline__ void warpReduceMinIdx(float &val, size_t &idx) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down(val, offset);
        size_t other_idx = (size_t)__shfl_down((unsigned long long)idx, offset);
        if (other_val < val) {
            val = other_val;
            idx = other_idx;
        }
    }
}

__device__ __forceinline__ void blockReduceMinIdx(float &val, size_t &idx) {
    __shared__ float shared_val[64];
    __shared__ size_t shared_idx[64];
    
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    warpReduceMinIdx(val, idx);

    if (lane == 0) {
        shared_val[wid] = val;
        shared_idx[wid] = idx;
    }
    __syncthreads();

    float read_val = FLT_MAX;
    size_t read_idx = SIZE_MAX;
    if (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) {
        read_val = shared_val[lane];
        read_idx = shared_idx[lane];
    }

    if (wid == 0) warpReduceMinIdx(read_val, read_idx);

    if (threadIdx.x == 0) {
        val = read_val;
        idx = read_idx;
    }
}

__global__ void k_calc_sums_initial(const float* D, float* R, int N) {
    int row = blockIdx.x; 
    if (row >= N) return;

    int tid = threadIdx.x;
    float local_sum = 0.0f;

    for (int k = tid; k < N; k += blockDim.x) {
        if (k != row) {
            local_sum += D[(size_t)row * N + k];
        }
    }

    local_sum = blockReduceSum(local_sum);
    
    if (tid == 0) R[row] = local_sum;
}

__global__ void k_find_min_compact(const float* D, const float* R, int count, int N_orig, 
                                   float* row_mins, size_t* row_idxs) {
    int row = blockIdx.x;
    if (row >= count) return;

    int tid = threadIdx.x;
    float r_i = R[row];
    float my_min_val = FLT_MAX;
    size_t my_min_idx = SIZE_MAX;

    for (int k = row + 1 + tid; k < count; k += blockDim.x) {
        float d_ik = D[(size_t)row * N_orig + k];
        float q = (count - 2) * d_ik - r_i - R[k];

        if (q < my_min_val) {
            my_min_val = q;
            my_min_idx = (size_t)row * N_orig + k;
        }
    }

    blockReduceMinIdx(my_min_val, my_min_idx);
    
    if (tid == 0) {
        row_mins[row] = my_min_val;
        row_idxs[row] = my_min_idx;
    }
}

__global__ void k_reduce_generic(const float* in_vals, const size_t* in_idxs, int n, float* out_vals, size_t* out_idxs) {
    int tid = threadIdx.x; 
    int gid = blockIdx.x * blockDim.x + tid;
    
    float local_val = FLT_MAX; 
    size_t local_idx = SIZE_MAX;
    
    for (int i = gid; i < n; i += gridDim.x * blockDim.x) {
        float v = in_vals[i]; 
        if (v < local_val) { 
            local_val = v; 
            local_idx = in_idxs[i]; 
        }
    }
    
    blockReduceMinIdx(local_val, local_idx);
    
    if (tid == 0) { 
        out_vals[blockIdx.x] = local_val; 
        out_idxs[blockIdx.x] = local_idx; 
    }
}

__global__ void k_reduce_final_pointer(const float* in_vals, const size_t* in_idxs, int n, float* d_global_min, size_t* d_global_idx) {
    int tid = threadIdx.x; 
    float local_val = FLT_MAX; 
    size_t local_idx = SIZE_MAX;
    
    for (int i = tid; i < n; i += blockDim.x) {
        float v = in_vals[i]; 
        if (v < local_val) { 
            local_val = v; 
            local_idx = in_idxs[i]; 
        }
    }
    
    blockReduceMinIdx(local_val, local_idx);
    
    if (tid == 0) { 
        *d_global_min = local_val; 
        *d_global_idx = local_idx; 
    }
}

__global__ void k_calc_update_params_compact(const float* D, const float* V, float* R, 
                                             const int* map, int count, 
                                             const size_t* best_idx_ptr, UpdateParams* params, 
                                             GPUMergeEvent* history, int step, int N_orig) {
    int tid = threadIdx.x;

    __shared__ int i_curr, j_curr;
    if (tid == 0) {
        size_t best = *best_idx_ptr;
        i_curr = (int)(best / N_orig);
        j_curr = (int)(best % N_orig);
        if (i_curr > j_curr) { int t = i_curr; i_curr = j_curr; j_curr = t; }
    }
    __syncthreads();

    float local_diff = 0.0f;
    for (int k = tid; k < count; k += blockDim.x) {
        if (k != i_curr && k != j_curr) {
            local_diff += (V[(size_t)j_curr * N_orig + k] - V[(size_t)i_curr * N_orig + k]);
        }
    }

    local_diff = blockReduceSum(local_diff);

    if (tid == 0) {
        float d_ij = D[(size_t)i_curr * N_orig + j_curr];
        float v_ij = V[(size_t)i_curr * N_orig + j_curr];
        float r_i = R[i_curr];
        float r_j = R[j_curr];

        float lambda = 0.5f;
        if (count > 2 && v_ij > 1e-12f) {
             lambda = 0.5f + local_diff / (2.0f * (count - 2) * v_ij);
        }
        if (lambda < 0.0f) lambda = 0.0f; else if (lambda > 1.0f) lambda = 1.0f;
        
        float mu = 1.0f - lambda;
        float median = 0.5f * d_ij;
        float fudge = (count > 2) ? (r_i - r_j) / (2.0f * (count - 2)) : 0.0f;
        float bi = median + fudge;
        float bj = median - fudge;

        int i_real = map[i_curr];
        int j_real = map[j_curr];
        history[step].node_i = i_real;
        history[step].node_j = j_real;
        history[step].branch_i = bi;
        history[step].branch_j = bj;

        params->i_curr = i_curr;
        params->j_curr = j_curr;
        params->i_real = i_real;
        params->j_real = j_real;
        params->lambda = lambda;
        params->mu = mu;
        params->d_corr = -lambda * bi - mu * bj;
        params->v_corr = -lambda * mu * v_ij;

        R[i_curr] = 0.0f; 
    }
}

__global__ void k_apply_update_incremental(float* D, float* V, float* R, int count, 
                                           const UpdateParams* params, int N_orig) {
    int tid = threadIdx.x;
    
    __shared__ UpdateParams p;
    if (tid == 0) p = *params;
    __syncthreads();

    float local_u_sum = 0.0f;

    for (int k = blockIdx.x * blockDim.x + tid; k < count; k += gridDim.x * blockDim.x) {
        if (k != p.i_curr && k != p.j_curr) {
            size_t idx_ik = (size_t)p.i_curr * N_orig + k;
            size_t idx_jk = (size_t)p.j_curr * N_orig + k;
            size_t idx_ki = (size_t)k * N_orig + p.i_curr;
            
            float d_ik = D[idx_ik];
            float d_jk = D[idx_jk];

            float d_new = p.lambda * d_ik + p.mu * d_jk + p.d_corr;
            float v_new = p.lambda * V[idx_ik] + p.mu * V[idx_jk] + p.v_corr;

            D[idx_ik] = d_new;
            D[idx_ki] = d_new;
            V[idx_ik] = v_new;
            V[idx_ki] = v_new;

            float r_k = R[k];
            r_k = r_k - d_ik - d_jk + d_new;
            R[k] = r_k;

            local_u_sum += d_new;
        }
    }

    if (local_u_sum != 0.0f) {
        atomicAdd(&R[p.i_curr], local_u_sum);
    }
}

__global__ void k_compact_matrix(float* D, float* V, float* R, int* map, 
                                 const UpdateParams* params, int old_count, int N_orig) {
    int dst = params->j_curr; 
    int src = old_count - 1;  

    if (dst == src) return;

    int tid = threadIdx.x;
    
    if (tid == 0 && blockIdx.x == 0) {
        R[dst] = R[src];
        map[dst] = map[src];
    }

    for (int k = blockIdx.x * blockDim.x + tid; k < old_count; k += gridDim.x * blockDim.x) {
        size_t src_row_idx = (size_t)src * N_orig + k;
        size_t src_col_idx = (size_t)k * N_orig + src;
        size_t dst_row_idx = (size_t)dst * N_orig + k;
        size_t dst_col_idx = (size_t)k * N_orig + dst;
        
        float d_val = D[src_row_idx];
        float v_val = V[src_row_idx];
        
        D[dst_row_idx] = d_val;
        V[dst_row_idx] = v_val;

        D[dst_col_idx] = D[src_col_idx];
        V[dst_col_idx] = V[src_col_idx];
    }
}

__global__ void k_get_final_dist_compact(const float* D, int* map, int N_orig, 
                                         float* out_dist, int* out_pair) {
    if (threadIdx.x == 0) {
        *out_dist = D[0 * (size_t)N_orig + 1];
        out_pair[0] = map[0];
        out_pair[1] = map[1];
    }
}

extern "C" 
void launch_bionj_gpu(const float* h_dist, int N, 
                      GPUMergeEvent* h_history, 
                      float* final_dist, int* final_pair) {
    auto startTime = getRealTime();
    
    float *d_D, *d_V, *d_R;
    int *d_map;
    GPUMergeEvent *d_history;
    UpdateParams *d_params;
    float *d_fin_dist; int *d_fin_pair;
    
    float *d_row_mins, *d_blk_mins, *d_gl_min;
    size_t *d_row_idxs, *d_blk_idxs, *d_gl_idx;

    size_t mat_sz = (size_t)N * (size_t)N * sizeof(float);

    CHECK_HIP(hipMalloc(&d_D, mat_sz));
    CHECK_HIP(hipMalloc(&d_V, mat_sz));
    CHECK_HIP(hipMalloc(&d_R, N * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_map, N * sizeof(int)));
    CHECK_HIP(hipMalloc(&d_history, (N-2) * sizeof(GPUMergeEvent)));
    CHECK_HIP(hipMalloc(&d_params, sizeof(UpdateParams)));
    CHECK_HIP(hipMalloc(&d_fin_dist, sizeof(float)));
    CHECK_HIP(hipMalloc(&d_fin_pair, 2 * sizeof(int)));

    CHECK_HIP(hipMalloc(&d_row_mins, N * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_row_idxs, N * sizeof(size_t)));
    int reduce_blocks = 256;
    CHECK_HIP(hipMalloc(&d_blk_mins, reduce_blocks * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_blk_idxs, reduce_blocks * sizeof(size_t)));
    CHECK_HIP(hipMalloc(&d_gl_min, sizeof(float)));
    CHECK_HIP(hipMalloc(&d_gl_idx, sizeof(size_t)));

    CHECK_HIP(hipMemcpy(d_D, h_dist, mat_sz, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_V, h_dist, mat_sz, hipMemcpyHostToDevice));
    
    std::vector<int> h_map(N);
    for(int i = 0; i < N; ++i) h_map[i] = i;
    CHECK_HIP(hipMemcpy(d_map, h_map.data(), N * sizeof(int), hipMemcpyHostToDevice));

    int threads = 256;
    int fixed_large_grid = 256;

    k_calc_sums_initial<<<N, threads>>>(d_D, d_R, N);
    
    CHECK_HIP(hipDeviceSynchronize());

    for (int step = 0; step < N - 2; ++step) {
        int count = N - step;
        
        // Find Min (Row-based + Parallel Reduction)
        k_find_min_compact<<<count, threads>>>(d_D, d_R, count, N, d_row_mins, d_row_idxs);

        k_reduce_generic<<<reduce_blocks, threads>>>(d_row_mins, d_row_idxs, count, d_blk_mins, d_blk_idxs);
        k_reduce_final_pointer<<<1, threads>>>(d_blk_mins, d_blk_idxs, reduce_blocks, d_gl_min, d_gl_idx);

        // Calculate Constants & Reset R[u]
        k_calc_update_params_compact<<<1, threads>>>(d_D, d_V, d_R, d_map, count, d_gl_idx, d_params, d_history, step, N);

        // Apply Update
        k_apply_update_incremental<<<fixed_large_grid, threads>>>(d_D, d_V, d_R, count, d_params, N);

        // Compact Matrix
        k_compact_matrix<<<fixed_large_grid, threads>>>(d_D, d_V, d_R, d_map, d_params, count, N);
    }

    CHECK_HIP(hipDeviceSynchronize());
    
    CHECK_HIP(hipMemcpy(h_history, d_history, (N - 2) * sizeof(GPUMergeEvent), hipMemcpyDeviceToHost));
    
    k_get_final_dist_compact<<<1, 1>>>(d_D, d_map, N, d_fin_dist, d_fin_pair);
    CHECK_HIP(hipMemcpy(final_dist, d_fin_dist, sizeof(float), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(final_pair, d_fin_pair, 2 * sizeof(int), hipMemcpyDeviceToHost));

    hipFree(d_D); hipFree(d_V); hipFree(d_R); hipFree(d_map);
    hipFree(d_history); hipFree(d_params); hipFree(d_fin_dist); hipFree(d_fin_pair);
    hipFree(d_row_mins); hipFree(d_row_idxs); hipFree(d_blk_mins); hipFree(d_blk_idxs); 
    hipFree(d_gl_min); hipFree(d_gl_idx);
}