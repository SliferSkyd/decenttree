#ifndef BIONJ_GPU_H
#define BIONJ_GPU_H

#include "nj.h"
#include "hip_kernels/bionj.h"
#include <vector>
#include <iostream>

// Struct to receive merge data from GPU
struct GPUMergeEvent {
    int node_i;      // Original Index of kept node
    int node_j;      // Original Index of removed node
    float branch_i;  // Branch length for node_i
    float branch_j;  // Branch length for node_j
};

// C-linkage for CUDA launcher
extern "C" 
void launch_bionj_gpu(const float* h_dist, int N, 
                      GPUMergeEvent* h_history, 
                      float* final_dist, int* final_pair);

namespace StartTree {

class BioNJGPU : public BIONJMatrix<float> {
public:
    BioNJGPU() : BIONJMatrix<float>() {}

    virtual std::string getAlgorithmName() const override {
        return "BIONJ-GPU";
    }

    /**
     * @brief Main Driver: Offloads to GPU, then updates 'clusters' 
     * using the GPU's merge history.
     */
    virtual bool constructTree() {
        int N = (int)this->row_count;
        if (N < 3) {
            finishClustering(); // Fallback for tiny inputs
            return true;
        }

        // 1. Flatten Matrix (DecentTree T** -> float*)
        std::vector<float> flat_dist(N * N);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                flat_dist[i * N + j] = (float)this->rows[i][j];
            }
        }

        // 2. Prepare Buffers
        // We need N-2 merge events
        std::vector<GPUMergeEvent> history(N - 2); 
        float final_dist = 0.0f;
        int final_pair[2] = {0, 0};

        // 3. Launch GPU Kernel
        std::cout << "Running BioNJ on GPU (" << N << " taxa)..." << std::endl;
        launch_bionj_gpu(flat_dist.data(), N, history.data(), &final_dist, final_pair);

        // 4. Replay History to build the DecentTree
        // DecentTree normally uses a row swap/remove logic (rowToCluster).
        // Since GPU tracked merges by *Original Index*, we use a simpler map:
        // Map[Original_Matrix_Index] -> Cluster_ID
        std::vector<size_t> origToCluster(N);
        
        // Initialize: Original indices map to the initial leaf clusters (0..N-1)
        for(size_t i=0; i<(size_t)N; ++i) {
            // Usually, rowToCluster[i] is initialized to i by base class
            origToCluster[i] = this->rowToCluster[i];
        }

        // Replay the N-2 merges
        for (int step = 0; step < N - 2; ++step) {
            int u_idx = history[step].node_i; // GPU reused this original index
            int v_idx = history[step].node_j; // GPU removed this original index
            
            size_t cluster_u = origToCluster[u_idx];
            size_t cluster_v = origToCluster[v_idx];
            
            float len_u = history[step].branch_i;
            float len_v = history[step].branch_j;

            // DecentTree Core: Add the new cluster
            this->clusters.addCluster(cluster_u, len_u, cluster_v, len_v);

            // The new cluster is always at the end of the list
            size_t new_cluster_id = this->clusters.size() - 1;

            // Update Map: u_idx now represents the new cluster
            origToCluster[u_idx] = new_cluster_id;
            origToCluster[v_idx] = (size_t)-1; // Invalidate
        }

        // 5. Finalize (Last 2 nodes)
        // The GPU loop stopped when 2 active nodes remained.
        // Their indices are returned in final_pair[0] and final_pair[1].
        int last_u = final_pair[0];
        int last_v = final_pair[1];
        
        size_t c_last_u = origToCluster[last_u];
        size_t c_last_v = origToCluster[last_v];

        // Standard unrooted tree finish: split distance evenly
        float half_dist = final_dist * 0.5f;
        
        this->clusters.addCluster(c_last_u, half_dist, c_last_v, half_dist);

        // 6. Cleanup internal state
        this->row_count = 0; // Mark as finished
        return true;
    }
};

} // namespace StartTree

#endif // BIONJ_GPU_H