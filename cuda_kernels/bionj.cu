#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <utils/timeutil.h>

#define BLOCK_SIZE 256
#define REDUCE_THREADS 256

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (Line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

struct GPUMergeEvent {
    int node_i;      // Original Index
    int node_j;      // Original Index
    float branch_i;
    float branch_j;
};

struct UpdateParams {
    int i_real; // Original ID (for history)
    int j_real; // Original ID
    int i_curr; // Current Compact ID (location in D)
    int j_curr; // Current Compact ID
    float lambda;
    float mu;
    float d_corr;
    float v_corr;
};

__global__ void k_calc_sums_initial(const float* D, float* R, int N) {
    int row = blockIdx.x; 
    if (row >= N) return;

    int tid = threadIdx.x;
    float local_sum = 0.0f;

    for (int k = tid; k < N; k += blockDim.x) {
        if (k != row) {
            local_sum += D[row * N + k];
        }
    }

    // Block Reduction
    __shared__ float s_data[BLOCK_SIZE];
    s_data[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_data[tid] += s_data[tid + s];
        __syncthreads();
    }
    if (tid == 0) R[row] = s_data[0];
}

__global__ void k_find_min_compact(const float* D, const float* R, int count, int N_orig, 
                                   float* row_mins, int* row_idxs) {
    int row = blockIdx.x;
    if (row >= count) return;

    int tid = threadIdx.x;
    float r_i = R[row];
    float my_min_val = FLT_MAX;
    int my_min_idx = -1;

    // Scan columns k > row (Symmetry optimization)
    // Access D[row * N + k] is perfectly coalesced
    for (int k = row + 1 + tid; k < count; k += blockDim.x) {
        float d_ik = D[row * N_orig + k];
        float q = (count - 2) * d_ik - r_i - R[k];

        if (q < my_min_val) {
            my_min_val = q;
            my_min_idx = row * N_orig + k; // Encoded Compact Index
        }
    }

    // Block Reduction
    __shared__ float s_vals[BLOCK_SIZE];
    __shared__ int s_idxs[BLOCK_SIZE];
    s_vals[tid] = my_min_val;
    s_idxs[tid] = my_min_idx;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_vals[tid + s] < s_vals[tid]) {
                s_vals[tid] = s_vals[tid + s];
                s_idxs[tid] = s_idxs[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        row_mins[row] = s_vals[0];
        row_idxs[row] = s_idxs[0];
    }
}

__global__ void k_reduce_generic(const float* in_vals, const int* in_idxs, int n, float* out_vals, int* out_idxs) {
    extern __shared__ float s_vals[]; int* s_idxs = (int*)&s_vals[blockDim.x];
    int tid = threadIdx.x; int gid = blockIdx.x * blockDim.x + tid;
    float local_val = FLT_MAX; int local_idx = -1;
    for (int i = gid; i < n; i += gridDim.x * blockDim.x) {
        float v = in_vals[i]; if (v < local_val) { local_val = v; local_idx = in_idxs[i]; }
    }
    s_vals[tid] = local_val; s_idxs[tid] = local_idx; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { if (s_vals[tid + s] < s_vals[tid]) { s_vals[tid] = s_vals[tid + s]; s_idxs[tid] = s_idxs[tid + s]; } } __syncthreads();
    }
    if (tid == 0) { out_vals[blockIdx.x] = s_vals[0]; out_idxs[blockIdx.x] = s_idxs[0]; }
}

__global__ void k_reduce_final_pointer(const float* in_vals, const int* in_idxs, int n, float* d_global_min, int* d_global_idx) {
    extern __shared__ float s_vals[]; int* s_idxs = (int*)&s_vals[blockDim.x];
    int tid = threadIdx.x; float local_val = FLT_MAX; int local_idx = -1;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = in_vals[i]; if (v < local_val) { local_val = v; local_idx = in_idxs[i]; }
    }
    s_vals[tid] = local_val; s_idxs[tid] = local_idx; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { if (s_vals[tid + s] < s_vals[tid]) { s_vals[tid] = s_vals[tid + s]; s_idxs[tid] = s_idxs[tid + s]; } } __syncthreads();
    }
    if (tid == 0) { *d_global_min = s_vals[0]; *d_global_idx = s_idxs[0]; }
}

__global__ void k_calc_update_params_compact(const float* D, const float* V, float* R, 
                                             const int* map, int count, 
                                             const int* best_idx_ptr, UpdateParams* params, 
                                             GPUMergeEvent* history, int step, int N_orig) {
    __shared__ float s_sum_diff_v;
    int tid = threadIdx.x;

    // 1. Decode Best Pair (Current Indices)
    __shared__ int i_curr, j_curr;
    if (tid == 0) {
        int best = *best_idx_ptr;
        i_curr = best / N_orig;
        j_curr = best % N_orig;
        if (i_curr > j_curr) { int t = i_curr; i_curr = j_curr; j_curr = t; }
    }
    __syncthreads();

    // 2. Parallel Reduction for sum_diff_v
    float local_diff = 0.0f;
    for (int k = tid; k < count; k += blockDim.x) {
        if (k != i_curr && k != j_curr) {
            local_diff += (V[j_curr * N_orig + k] - V[i_curr * N_orig + k]);
        }
    }

    __shared__ float s_data[BLOCK_SIZE];
    s_data[tid] = local_diff;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_data[tid] += s_data[tid + s];
        __syncthreads();
    }

    // 3. Thread 0 Computes Final Scalar Constants
    if (tid == 0) {
        s_sum_diff_v = s_data[0];

        float d_ij = D[i_curr * N_orig + j_curr];
        float v_ij = V[i_curr * N_orig + j_curr];
        float r_i = R[i_curr];
        float r_j = R[j_curr];

        float lambda = 0.5f;
        if (count > 2 && v_ij > 1e-12f) {
             lambda = 0.5f + s_sum_diff_v / (2.0f * (count - 2) * v_ij);
        }
        if (lambda < 0.0f) lambda = 0.0f; else if (lambda > 1.0f) lambda = 1.0f;
        
        float mu = 1.0f - lambda;
        float median = 0.5f * d_ij;
        float fudge = (count > 2) ? (r_i - r_j) / (2.0f * (count - 2)) : 0.0f;
        float bi = median + fudge;
        float bj = median - fudge;

        // Record History
        int i_real = map[i_curr];
        int j_real = map[j_curr];
        history[step].node_i = i_real;
        history[step].node_j = j_real;
        history[step].branch_i = bi;
        history[step].branch_j = bj;

        // Write Params
        params->i_curr = i_curr;
        params->j_curr = j_curr;
        params->i_real = i_real;
        params->j_real = j_real;
        params->lambda = lambda;
        params->mu = mu;
        params->d_corr = -lambda * bi - mu * bj;
        params->v_corr = -lambda * mu * v_ij;

        // INCREMENTAL OPTIMIZATION:
        // Reset R for the new node u (which overwrites i_curr)
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

    // Grid Stride
    for (int k = blockIdx.x * blockDim.x + tid; k < count; k += gridDim.x * blockDim.x) {
        if (k != p.i_curr && k != p.j_curr) {
            float d_ik = D[p.i_curr * N_orig + k];
            float d_jk = D[p.j_curr * N_orig + k];

            // Calc New D and V
            float d_new = p.lambda * d_ik + p.mu * d_jk + p.d_corr;
            float v_new = p.lambda * V[p.i_curr * N_orig + k] + p.mu * V[p.j_curr * N_orig + k] + p.v_corr;

            // Write D
            D[p.i_curr * N_orig + k] = d_new;
            D[k * N_orig + p.i_curr] = d_new;

            // Write V
            V[p.i_curr * N_orig + k] = v_new;
            V[k * N_orig + p.i_curr] = v_new;

            // INCREMENTAL R UPDATE for neighbor k
            // R[k] = R[k] - d_ik - d_jk + d_new
            // This is safe without atomics because 'k' is unique per thread
            float r_k = R[k];
            r_k = r_k - d_ik - d_jk + d_new;
            R[k] = r_k;

            // Accumulate sum for new node u
            local_u_sum += d_new;
        }
    }

    // Reduce local_u_sum and atomic add to R[u]
    // R[u] (at p.i_curr) was reset to 0 in previous kernel
    if (local_u_sum != 0.0f) {
        atomicAdd(&R[p.i_curr], local_u_sum);
    }
}

__global__ void k_compact_matrix(float* D, float* V, float* R, int* map, 
                                 const UpdateParams* params, int old_count, int N_orig) {
    int dst = params->j_curr; // The removed node
    int src = old_count - 1;  // The last node

    if (dst == src) return;

    int tid = threadIdx.x;
    
    if (tid == 0 && blockIdx.x == 0) {
        R[dst] = R[src];
        map[dst] = map[src];
    }

    for (int k = blockIdx.x * blockDim.x + tid; k < old_count; k += gridDim.x * blockDim.x) {
        // Read Source
        float d_val = D[src * N_orig + k];
        float v_val = V[src * N_orig + k];
        
        // Write Row 'dst'
        D[dst * N_orig + k] = d_val;
        V[dst * N_orig + k] = v_val;

        // Write Col 'dst'
        D[k * N_orig + dst] = D[k * N_orig + src];
        V[k * N_orig + dst] = V[k * N_orig + src];
    }
}

__global__ void k_get_final_dist_compact(const float* D, int* map, int N_orig, 
                                         float* out_dist, int* out_pair) {
    if (threadIdx.x == 0) {
        // Last 2 nodes are always at 0 and 1
        *out_dist = D[0 * N_orig + 1];
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
    int *d_row_idxs, *d_blk_idxs, *d_gl_idx;

    size_t mat_sz = N * N * sizeof(float);

    // Allocations
    CHECK_CUDA(cudaMalloc(&d_D, mat_sz));
    CHECK_CUDA(cudaMalloc(&d_V, mat_sz));
    CHECK_CUDA(cudaMalloc(&d_R, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_map, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_history, (N-2) * sizeof(GPUMergeEvent)));
    CHECK_CUDA(cudaMalloc(&d_params, sizeof(UpdateParams)));
    CHECK_CUDA(cudaMalloc(&d_fin_dist, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fin_pair, 2 * sizeof(int)));

    // Reduction Buffers
    CHECK_CUDA(cudaMalloc(&d_row_mins, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_row_idxs, N * sizeof(int)));
    int reduce_blocks = 256;
    CHECK_CUDA(cudaMalloc(&d_blk_mins, reduce_blocks * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_blk_idxs, reduce_blocks * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_gl_min, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gl_idx, sizeof(int)));

    // Data Transfer
    CHECK_CUDA(cudaMemcpy(d_D, h_dist, mat_sz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_dist, mat_sz, cudaMemcpyHostToDevice));
    
    // Init Map
    std::vector<int> h_map(N);
    for(int i=0; i<N; ++i) h_map[i] = i;
    CHECK_CUDA(cudaMemcpy(d_map, h_map.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    int threads = 256;
    size_t reduce_smem = threads * (sizeof(float) + sizeof(int));
    int fixed_large_grid = 256;

    // -------------------------------------------------------------------------
    // Initialization
    // -------------------------------------------------------------------------
    // Calculate initial row sums ONCE
    k_calc_sums_initial<<<N, threads>>>(d_D, d_R, N);
    
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("GPU BioNJ Init Complete (%.3fs)\n", getRealTime() - startTime);

    // -------------------------------------------------------------------------
    // Main Loop
    // -------------------------------------------------------------------------
    for (int step = 0; step < N - 2; ++step) {
        int count = N - step;
        
        // Find Min (Row-based + Parallel Reduction)
        k_find_min_compact<<<count, threads>>>(d_D, d_R, count, N, d_row_mins, d_row_idxs);
        k_reduce_generic<<<reduce_blocks, threads, reduce_smem>>>(d_row_mins, d_row_idxs, count, d_blk_mins, d_blk_idxs);
        k_reduce_final_pointer<<<1, threads, reduce_smem>>>(d_blk_mins, d_blk_idxs, reduce_blocks, d_gl_min, d_gl_idx);

        // Calculate Constants (Thread 0) & Reset R[u]
        k_calc_update_params_compact<<<1, threads>>>(d_D, d_V, d_R, d_map, count, d_gl_idx, d_params, d_history, step, N);

        // Apply Update + Incremental Sums
        k_apply_update_incremental<<<fixed_large_grid, threads>>>(d_D, d_V, d_R, count, d_params, N);

        // Compact Matrix (Move last node to hole)
        k_compact_matrix<<<fixed_large_grid, threads>>>(d_D, d_V, d_R, d_map, d_params, count, N);
    }

    // -------------------------------------------------------------------------
    // Finalize
    // -------------------------------------------------------------------------
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Get History
    CHECK_CUDA(cudaMemcpy(h_history, d_history, (N - 2) * sizeof(GPUMergeEvent), cudaMemcpyDeviceToHost));
    
    // Get Final Nodes
    k_get_final_dist_compact<<<1, 1>>>(d_D, d_map, N, d_fin_dist, d_fin_pair);
    CHECK_CUDA(cudaMemcpy(final_dist, d_fin_dist, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(final_pair, d_fin_pair, 2 * sizeof(int), cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(d_D); cudaFree(d_V); cudaFree(d_R); cudaFree(d_map);
    cudaFree(d_history); cudaFree(d_params); cudaFree(d_fin_dist); cudaFree(d_fin_pair);
    cudaFree(d_row_mins); cudaFree(d_row_idxs); cudaFree(d_blk_mins); cudaFree(d_blk_idxs); 
    cudaFree(d_gl_min); cudaFree(d_gl_idx);
}