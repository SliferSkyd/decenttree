#include "hip/hip_runtime.h"
#include <cfloat>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <utils/timeutil.h>
#include "../phasetimer.h"

#define BLOCK_SIZE 256
#define REDUCE_THREADS 256

/* Set to 1 to build a sequential host implementation of launch_bionj_gpu()
 * with identical semantics instead of the HIP kernels.  It is far too slow for
 * production (O(N^3) minimum search), but it lets the whole BIONJ-GPU code
 * path - clusterDuplicates(), the merge-history replay, the final
 * trifurcation handoff to finishClustering() - be validated on a machine with
 * no ROCm device, and gives a reference to diff the kernels against. */
#ifndef DT_BIONJ_GPU_CPU_REFERENCE
#define DT_BIONJ_GPU_CPU_REFERENCE 0
#endif

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

/**
 * Tie-break rule for the Q minimum.
 *
 * The CPU path (NJMatrix::getRowMinima + UPGMA_Matrix::getMinimumEntry) scans
 * the lower triangle with a strict "<", columns ascending inside a row and
 * rows ascending across rows, so among equal Q values it takes the pair with
 * the smallest higher index, then the smallest lower index.  Encoding the
 * candidate as key = (higher * N_orig + lower) makes "smallest key" reproduce
 * exactly that order.
 *
 * Without a tie-break the winner depends on which lane survives the shuffle
 * cascade and which of the fixed 256 blocks owned the element, which is not
 * reproducible.  Ties are not a corner case here: any set of identical taxa
 * gives an identical Q for every pair among them.
 */
__device__ __forceinline__ bool betterMin(float val, size_t key,
                                          float best_val, size_t best_key) {
    return (val < best_val) || (val == best_val && key < best_key);
}

__device__ __forceinline__ void warpReduceMinIdx(float &val, size_t &idx) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down(val, offset);
        size_t other_idx = (size_t)__shfl_down((unsigned long long)idx, offset);
        if (betterMin(other_val, other_idx, val, idx)) {
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

        // key is (higher index, lower index) - see betterMin()
        size_t key = (size_t)k * N_orig + row;
        if (betterMin(q, key, my_min_val, my_min_idx)) {
            my_min_val = q;
            my_min_idx = key;
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
        size_t k = in_idxs[i];
        if (betterMin(v, k, local_val, local_idx)) { 
            local_val = v; 
            local_idx = k; 
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
        size_t k = in_idxs[i];
        if (betterMin(v, k, local_val, local_idx)) { 
            local_val = v; 
            local_idx = k; 
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
        // key was packed as (higher * N_orig + lower)
        j_curr = (int)(best / N_orig);
        i_curr = (int)(best % N_orig);
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

        // R[i_curr] is rebuilt from scratch by k_finalise_row_total; it is not
        // read again before then, so there is nothing to reset here.
    }
}

__global__ void k_apply_update_incremental(float* D, float* V, float* R, int count, 
                                           const UpdateParams* params, int N_orig,
                                           float* row_partials) {
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

    // Deterministic, well-conditioned accumulation of the merged row's total.
    // The previous version had every one of the (grid x block) threads issue
    // its own atomicAdd to R[i_curr]: float addition is not associative and
    // atomic order is not reproducible, so the row total - and through it the
    // next Q minimum, and hence the topology - varied between identical runs.
    // A per-block tree reduction into row_partials[], summed by
    // k_finalise_row_total over a fixed grid, is reproducible and much more
    // accurate (and issues no atomics at all).
    local_u_sum = blockReduceSum(local_u_sum);
    if (tid == 0) {
        row_partials[blockIdx.x] = local_u_sum;
    }
}

__global__ void k_finalise_row_total(const float* row_partials, int n_partials,
                                     float* R, const UpdateParams* params) {
    int tid = threadIdx.x;
    float s = 0.0f;
    for (int i = tid; i < n_partials; i += blockDim.x) {
        s += row_partials[i];
    }
    s = blockReduceSum(s);
    if (tid == 0) {
        R[params->i_curr] = s;
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
        // The moved cluster's self-distance must stay zero.  The previous
        // version let k == dst fall through the copy loop below, which set
        // D[dst][dst] = d(src,dst); and the k == src iteration wrote
        // D[src][dst] while the k == dst iteration read it - a real race.
        // Both are avoided by skipping those two k values here.
        D[(size_t)dst * N_orig + dst] = 0.0f;
        V[(size_t)dst * N_orig + dst] = 0.0f;
    }

    for (int k = blockIdx.x * blockDim.x + tid; k < old_count; k += gridDim.x * blockDim.x) {
        if (k == dst || k == src) continue;

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

/**
 * Hand the final three active clusters back to the host, so that the shared
 * UPGMA_Matrix::finishClustering() can build the degree-3 root the rest of
 * decentTree expects.  The previous version instead ran one extra join down to
 * two nodes and split the remaining distance evenly, which produced N-1
 * internal nodes (one too many for an unrooted tree on N leaves), a degree-2
 * root, and root-adjacent branch lengths that did not match the CPU path.
 */
__global__ void k_get_final3_compact(const float* D, const int* map, int N_orig,
                                     float* out_d3, int* out_map3) {
    if (threadIdx.x == 0) {
        out_d3[0] = D[(size_t)0 * N_orig + 1];   // d(0,1)
        out_d3[1] = D[(size_t)0 * N_orig + 2];   // d(0,2)
        out_d3[2] = D[(size_t)1 * N_orig + 2];   // d(1,2)
        out_map3[0] = map[0];
        out_map3[1] = map[1];
        out_map3[2] = map[2];
    }
}

#if !DT_BIONJ_GPU_CPU_REFERENCE

extern "C" 
void launch_bionj_gpu(const float* h_dist, const float* h_var, int N, 
                      GPUMergeEvent* h_history, 
                      float* final_d3, int* final_map3) {
    auto startTime = getRealTime();
    
    float *d_D, *d_V, *d_R;
    int *d_map;
    GPUMergeEvent *d_history;
    UpdateParams *d_params;
    float *d_fin_d3; int *d_fin_map3;
    
    float *d_row_mins, *d_blk_mins, *d_gl_min, *d_row_partials;
    size_t *d_row_idxs, *d_blk_idxs, *d_gl_idx;

    size_t mat_sz = (size_t)N * (size_t)N * sizeof(float);

    PhaseTimer t_alloc("3c-1. device allocation");
    int threads = 256;
    int fixed_large_grid = 256;
    int reduce_blocks = 256;

    CHECK_HIP(hipMalloc(&d_D, mat_sz));
    CHECK_HIP(hipMalloc(&d_V, mat_sz));
    CHECK_HIP(hipMalloc(&d_R, N * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_map, N * sizeof(int)));
    CHECK_HIP(hipMalloc(&d_history, (N-3) * sizeof(GPUMergeEvent)));
    CHECK_HIP(hipMalloc(&d_params, sizeof(UpdateParams)));
    CHECK_HIP(hipMalloc(&d_fin_d3, 3 * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_fin_map3, 3 * sizeof(int)));

    CHECK_HIP(hipMalloc(&d_row_mins, N * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_row_idxs, N * sizeof(size_t)));
    CHECK_HIP(hipMalloc(&d_blk_mins, reduce_blocks * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_blk_idxs, reduce_blocks * sizeof(size_t)));
    CHECK_HIP(hipMalloc(&d_gl_min, sizeof(float)));
    CHECK_HIP(hipMalloc(&d_gl_idx, sizeof(size_t)));
    CHECK_HIP(hipMalloc(&d_row_partials, fixed_large_grid * sizeof(float)));

    t_alloc.report();

    PhaseTimer t_h2d("3c-2. host-to-device copy");
    CHECK_HIP(hipMemcpy(d_D, h_dist, mat_sz, hipMemcpyHostToDevice));
    // The variance matrix is passed in separately.  It is only equal to the
    // distance matrix on untouched input; once clusterDuplicates() has joined
    // identical taxa, V and D have diverged, so copying h_dist into both (as
    // the previous version did) would have been wrong.
    CHECK_HIP(hipMemcpy(d_V, h_var, mat_sz, hipMemcpyHostToDevice));
    
    std::vector<int> h_map(N);
    for(int i = 0; i < N; ++i) h_map[i] = i;
    CHECK_HIP(hipMemcpy(d_map, h_map.data(), N * sizeof(int), hipMemcpyHostToDevice));

    CHECK_HIP(hipDeviceSynchronize());
    t_h2d.report();

    PhaseTimer t_sums("3c-3. initial row sums");
    k_calc_sums_initial<<<N, threads>>>(d_D, d_R, N);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    t_sums.report();

    PhaseTimer t_loop("3c-4. neighbour-joining loop (N-3 merges)");

    // Stop with three active clusters; the host finishes the trifurcation.
    for (int step = 0; step < N - 3; ++step) {
        int count = N - step;
        
        // Find Min (Row-based + Parallel Reduction)
        k_find_min_compact<<<count, threads>>>(d_D, d_R, count, N, d_row_mins, d_row_idxs);

        k_reduce_generic<<<reduce_blocks, threads>>>(d_row_mins, d_row_idxs, count, d_blk_mins, d_blk_idxs);
        k_reduce_final_pointer<<<1, threads>>>(d_blk_mins, d_blk_idxs, reduce_blocks, d_gl_min, d_gl_idx);

        // Calculate Constants
        k_calc_update_params_compact<<<1, threads>>>(d_D, d_V, d_R, d_map, count, d_gl_idx, d_params, d_history, step, N);

        // Apply Update
        k_apply_update_incremental<<<fixed_large_grid, threads>>>(d_D, d_V, d_R, count, d_params, N, d_row_partials);
        k_finalise_row_total<<<1, threads>>>(d_row_partials, fixed_large_grid, d_R, d_params);

        // Compact Matrix
        k_compact_matrix<<<fixed_large_grid, threads>>>(d_D, d_V, d_R, d_map, d_params, count, N);
    }

    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    t_loop.report();

    PhaseTimer t_d2h("3c-5. device-to-host copy of merge history");
    if (N > 3) {
        CHECK_HIP(hipMemcpy(h_history, d_history, (N - 3) * sizeof(GPUMergeEvent), hipMemcpyDeviceToHost));
    }
    
    k_get_final3_compact<<<1, 1>>>(d_D, d_map, N, d_fin_d3, d_fin_map3);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipMemcpy(final_d3, d_fin_d3, 3 * sizeof(float), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(final_map3, d_fin_map3, 3 * sizeof(int), hipMemcpyDeviceToHost));
    t_d2h.report();

    hipFree(d_D); hipFree(d_V); hipFree(d_R); hipFree(d_map);
    hipFree(d_history); hipFree(d_params); hipFree(d_fin_d3); hipFree(d_fin_map3);
    hipFree(d_row_mins); hipFree(d_row_idxs); hipFree(d_blk_mins); hipFree(d_blk_idxs); 
    hipFree(d_gl_min); hipFree(d_gl_idx); hipFree(d_row_partials);
}

#else /* DT_BIONJ_GPU_CPU_REFERENCE */

/**
 * Sequential host reference for launch_bionj_gpu().
 *
 * Mirrors the kernels exactly: same float32 arithmetic, same Q criterion,
 * same (higher,lower) tie-break, same incremental row-total update, same
 * swap-with-last compaction, and the same "stop with three active clusters"
 * contract.  Summation order differs from the GPU's tree reductions, so it is
 * not bit-identical, but any structural disagreement with this reference is a
 * kernel bug.
 */
extern "C"
void launch_bionj_gpu(const float* h_dist, const float* h_var, int N,
                      GPUMergeEvent* h_history,
                      float* final_d3, int* final_map3) {
    std::vector<float> D(h_dist, h_dist + (size_t)N * N);
    std::vector<float> V(h_var,  h_var  + (size_t)N * N);
    std::vector<float> R((size_t)N, 0.0f);
    std::vector<int>   map((size_t)N);
    for (int i = 0; i < N; ++i) map[i] = i;

    for (int i = 0; i < N; ++i) {                       // k_calc_sums_initial
        float s = 0.0f;
        for (int k = 0; k < N; ++k) {
            if (k != i) s += D[(size_t)i * N + k];
        }
        R[i] = s;
    }

    for (int step = 0; step < N - 3; ++step) {
        int count = N - step;

        // k_find_min_compact + the two reduction kernels
        float  best_q   = FLT_MAX;
        size_t best_key = SIZE_MAX;
        for (int i = 0; i < count; ++i) {
            float r_i = R[i];
            for (int k = i + 1; k < count; ++k) {
                float  q   = (count - 2) * D[(size_t)i * N + k] - r_i - R[k];
                size_t key = (size_t)k * N + i;
                if (q < best_q || (q == best_q && key < best_key)) {
                    best_q   = q;
                    best_key = key;
                }
            }
        }
        int j_curr = (int)(best_key / N);
        int i_curr = (int)(best_key % N);
        if (i_curr > j_curr) { int t = i_curr; i_curr = j_curr; j_curr = t; }

        // k_calc_update_params_compact
        float local_diff = 0.0f;
        for (int k = 0; k < count; ++k) {
            if (k != i_curr && k != j_curr) {
                local_diff += (V[(size_t)j_curr * N + k] - V[(size_t)i_curr * N + k]);
            }
        }
        float d_ij = D[(size_t)i_curr * N + j_curr];
        float v_ij = V[(size_t)i_curr * N + j_curr];
        float r_i  = R[i_curr];
        float r_j  = R[j_curr];

        float lambda = 0.5f;
        if (count > 2 && v_ij > 1e-12f) {
            lambda = 0.5f + local_diff / (2.0f * (count - 2) * v_ij);
        }
        if (lambda < 0.0f) lambda = 0.0f; else if (lambda > 1.0f) lambda = 1.0f;
        float mu     = 1.0f - lambda;
        float median = 0.5f * d_ij;
        float fudge  = (count > 2) ? (r_i - r_j) / (2.0f * (count - 2)) : 0.0f;
        float bi     = median + fudge;
        float bj     = median - fudge;
        float d_corr = -lambda * bi - mu * bj;
        float v_corr = -lambda * mu * v_ij;

        h_history[step].node_i   = map[i_curr];
        h_history[step].node_j   = map[j_curr];
        h_history[step].branch_i = bi;
        h_history[step].branch_j = bj;

        // k_apply_update_incremental + k_finalise_row_total
        float u_sum = 0.0f;
        for (int k = 0; k < count; ++k) {
            if (k == i_curr || k == j_curr) continue;
            size_t idx_ik = (size_t)i_curr * N + k;
            size_t idx_jk = (size_t)j_curr * N + k;
            float  d_ik   = D[idx_ik];
            float  d_jk   = D[idx_jk];
            float  d_new  = lambda * d_ik + mu * d_jk + d_corr;
            float  v_new  = lambda * V[idx_ik] + mu * V[idx_jk] + v_corr;

            D[idx_ik] = d_new;
            D[(size_t)k * N + i_curr] = d_new;
            V[idx_ik] = v_new;
            V[(size_t)k * N + i_curr] = v_new;

            R[k] = R[k] - d_ik - d_jk + d_new;
            u_sum += d_new;
        }
        R[i_curr] = u_sum;

        // k_compact_matrix
        int dst = j_curr;
        int src = count - 1;
        if (dst != src) {
            R[dst]   = R[src];
            map[dst] = map[src];
            D[(size_t)dst * N + dst] = 0.0f;
            V[(size_t)dst * N + dst] = 0.0f;
            for (int k = 0; k < count; ++k) {
                if (k == dst || k == src) continue;
                D[(size_t)dst * N + k] = D[(size_t)src * N + k];
                V[(size_t)dst * N + k] = V[(size_t)src * N + k];
                D[(size_t)k * N + dst] = D[(size_t)k * N + src];
                V[(size_t)k * N + dst] = V[(size_t)k * N + src];
            }
        }
    }

    final_d3[0]  = D[(size_t)0 * N + 1];
    final_d3[1]  = D[(size_t)0 * N + 2];
    final_d3[2]  = D[(size_t)1 * N + 2];
    final_map3[0] = map[0];
    final_map3[1] = map[1];
    final_map3[2] = map[2];
}

#endif /* DT_BIONJ_GPU_CPU_REFERENCE */
