#ifndef BIONJ_GPU_H
#define BIONJ_GPU_H

#include "nj.h"
#include "phasetimer.h"
#include <vector>
#include <iostream>

// Struct to receive merge data from GPU
struct GPUMergeEvent {
    int node_i;      // Original Index of kept node
    int node_j;      // Original Index of removed node
    float branch_i;  // Branch length for node_i
    float branch_j;  // Branch length for node_j
};

// C-linkage for CUDA launcher.  Declared once, in the kernel's own header,
// so the host and device sides cannot drift apart.
#include "hip_kernels/bionj.h"

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
     * @note  This mirrors UPGMA_Matrix::constructTree(): the same
     *        prepareToConstructTree() / clusterDuplicates() preamble and the
     *        same finishClustering() epilogue, with only the join loop
     *        offloaded.  Both were missing before, which is why BIONJ-GPU
     *        used to resolve every duplicate taxon by an arbitrary tie-break
     *        (the CPU algorithms collapse up to a third of the taxa on these
     *        datasets) and used to emit N-1 internal nodes with a degree-2
     *        root instead of the unrooted N-2 with a degree-3 root.
     */
    virtual bool constructTree() override {
        {
            PhaseTimer t("3a. cluster duplicate taxa");
            prepareToConstructTree(); // BIONJMatrix: sets variance = *this
            clusterDuplicates();      // collapse identical/near-identical taxa
        }

        int N = (int)this->row_count;
        if (N <= 3) {
            finishClustering(); // nothing left worth offloading
            return true;
        }

        // 1. Flatten both matrices (DecentTree T** -> float*).
        //    V must be sent separately: clusterDuplicates() has already
        //    joined taxa, so the variance matrix is no longer equal to D.
        std::vector<float> flat_dist((size_t)N * (size_t)N);
        std::vector<float> flat_var ((size_t)N * (size_t)N);
        {
            PhaseTimer t("3b. flatten D and V for the device");
            for (int i = 0; i < N; ++i) {
                const float* d_row = this->rows[i];
                const float* v_row = this->variance.rows[i];
                for (int j = 0; j < N; ++j) {
                    flat_dist[(size_t)i * N + j] = d_row[j];
                    flat_var [(size_t)i * N + j] = v_row[j];
                }
            }
        }

        // 2. Prepare Buffers.  The GPU stops with 3 active clusters, so it
        //    reports N-3 merges and the surviving 3x3 block.
        std::vector<GPUMergeEvent> history(N - 3);
        float final_d3[3]  = { 0.0f, 0.0f, 0.0f };
        int   final_map3[3] = { 0, 1, 2 };

        // 3. Launch GPU Kernel
        std::cout << "Running BioNJ on GPU (" << N << " taxa)..." << std::endl;
        {
            PhaseTimer t("3c. GPU total (alloc + copies + joins)");
            launch_bionj_gpu(flat_dist.data(), flat_var.data(), N,
                             history.data(), final_d3, final_map3);
        }

        // 4. Replay History to build the DecentTree
        // DecentTree normally uses a row swap/remove logic (rowToCluster).
        // Since GPU tracked merges by *Original Index*, we use a simpler map:
        // Map[Original_Matrix_Index] -> Cluster_ID
        PhaseTimer replay("3d. replay merge history + finish tree");
        std::vector<size_t> origToCluster(N);
        for (size_t i = 0; i < (size_t)N; ++i) {
            origToCluster[i] = this->rowToCluster[i];
        }

        for (int step = 0; step < N - 3; ++step) {
            int u_idx = history[step].node_i; // GPU reused this original index
            int v_idx = history[step].node_j; // GPU removed this original index

            size_t cluster_u = origToCluster[u_idx];
            size_t cluster_v = origToCluster[v_idx];

            this->clusters.addCluster(cluster_u, history[step].branch_i,
                                      cluster_v, history[step].branch_j);

            origToCluster[u_idx] = this->clusters.size() - 1;
            origToCluster[v_idx] = (size_t)-1; // Invalidate
        }

        // 5. Finalize: hand the last three clusters to the shared
        //    finishClustering(), which builds the degree-3 root with
        //    taxon-count-weighted branch lengths (and sets row_count = 0).
        this->row_count = 3;
        for (int k = 0; k < 3; ++k) {
            this->rowToCluster[k] = origToCluster[final_map3[k]];
        }
        this->rows[0][1] = this->rows[1][0] = final_d3[0];
        this->rows[0][2] = this->rows[2][0] = final_d3[1];
        this->rows[1][2] = this->rows[2][1] = final_d3[2];
        finishClustering();
        replay.report();

        return true;
    }
};

} // namespace StartTree

#endif // BIONJ_GPU_H
