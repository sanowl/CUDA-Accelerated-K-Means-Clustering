#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <cooperative_groups.h>
#include "../include/initialization.h"
#include "../include/distance.h"

#define BLOCK_SIZE 256
#define WARP_SIZE 32

namespace cg = cooperative_groups;

__global__ void setupRNG(curandState *state, unsigned long seed, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

__global__ void randomInitialization(const float* __restrict__ points, float* __restrict__ centroids, 
                                     int n_points, int n_clusters, int dims, curandState* state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_clusters) {
        curandState localState = state[idx];
        int random_point = curand(&localState) % n_points;

        #pragma unroll 8
        for (int d = 0; d < dims; d += 8) {
            if (d + 8 <= dims) {
                float4 point_data1 = reinterpret_cast<const float4*>(&points[random_point * dims + d])[0];
                float4 point_data2 = reinterpret_cast<const float4*>(&points[random_point * dims + d + 4])[0];
                reinterpret_cast<float4*>(&centroids[idx * dims + d])[0] = point_data1;
                reinterpret_cast<float4*>(&centroids[idx * dims + d + 4])[0] = point_data2;
            } else {
                for (int r = d; r < dims; ++r) {
                    centroids[idx * dims + r] = points[random_point * dims + r];
                }
            }
        }

        state[idx] = localState;
    }
}

__device__ __forceinline__ float warpReduceMin(float val, int* idx) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, *idx, offset);
        if (other_val < val) {
            val = other_val;
            *idx = other_idx;
        }
    }
    return val;
}

__global__ void advancedComputeDistancesToNearest(const float* __restrict__ points, const float* __restrict__ centroids,
                                                  float* __restrict__ distances, int* __restrict__ nearest,
                                                  int n_points, int n_clusters, int dims) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> warp = cg::tiled_partition<WARP_SIZE>(block);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int wid = threadIdx.x / WARP_SIZE;

    extern __shared__ float shared_mem[];
    float* s_point = &shared_mem[wid * dims];
    float* s_centroid = &shared_mem[(blockDim.x / WARP_SIZE) * dims + wid * dims];

    if (tid < n_points) {
        float min_dist = FLT_MAX;
        int closest = 0;

        // Load point data into shared memory
        for (int d = warp.thread_rank(); d < dims; d += WARP_SIZE) {
            s_point[d] = points[tid * dims + d];
        }
        warp.sync();

        // Compute distances
        for (int c = 0; c < n_clusters; c++) {
            // Load centroid data into shared memory
            for (int d = warp.thread_rank(); d < dims; d += WARP_SIZE) {
                s_centroid[d] = centroids[c * dims + d];
            }
            warp.sync();

            float dist = advancedEuclideanDistance(s_point, s_centroid, dims);

            // Warp-level reduction to find minimum distance
            int local_c = c;
            dist = warpReduceMin(dist, &local_c);

            if (warp.thread_rank() == 0) {
                if (dist < min_dist) {
                    min_dist = dist;
                    closest = local_c;
                }
            }
        }

        if (warp.thread_rank() == 0) {
            distances[tid] = min_dist * min_dist;  // Store squared distance
            nearest[tid] = closest;
        }
    }
}

__global__ void kmeansppSelection(const float* __restrict__ points, float* __restrict__ centroids,
                                  const float* __restrict__ distances, int n_points, int n_clusters,
                                  int dims, int current_cluster, curandState* state) {
    cg::thread_block block = cg::this_thread_block();
    
    __shared__ float s_sum_distances;
    __shared__ int s_selected_point;

    if (threadIdx.x == 0) {
        s_sum_distances = 0.0f;
        s_selected_point = -1;
    }
    block.sync();

    // Parallel sum of distances using warp-level reduction
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < n_points; i += blockDim.x) {
        local_sum += distances[i];
    }

    local_sum = cg::reduce(block, local_sum, cg::plus<float>());

    if (threadIdx.x == 0) {
        s_sum_distances = local_sum;
        float random_value = curand_uniform(&state[0]) * s_sum_distances;
        float cumulative_sum = 0.0f;

        for (int i = 0; i < n_points; i++) {
            cumulative_sum += distances[i];
            if (cumulative_sum >= random_value) {
                s_selected_point = i;
                break;
            }
        }
    }
    block.sync();

    // Copy selected point to centroids using vectorized memory operations
    for (int d = threadIdx.x; d < dims; d += blockDim.x) {
        if (d + 4 <= dims) {
            reinterpret_cast<float4*>(&centroids[current_cluster * dims + d])[0] = 
                reinterpret_cast<const float4*>(&points[s_selected_point * dims + d])[0];
        } else {
            centroids[current_cluster * dims + d] = points[s_selected_point * dims + d];
        }
    }
}

bool supportsDynamicParallelism() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return prop.major >= 3 && prop.minor >= 5;
}

void hybridKmeansppSelection(const float* d_points, float* d_centroids, const float* d_distances,
                             int n_points, int n_clusters, int dims, int current_cluster, curandState* d_rng_state) {
    if (supportsDynamicParallelism()) {
        kmeansppSelection<<<1, BLOCK_SIZE>>>(d_points, d_centroids, d_distances, n_points, n_clusters, dims, current_cluster, d_rng_state);
    } else {
        thrust::device_vector<float> h_distances(d_distances, d_distances + n_points);
        float sum_distances = thrust::reduce(thrust::device, h_distances.begin(), h_distances.end());

        // CPU selection logic
        float random_value = static_cast<float>(rand()) / RAND_MAX * sum_distances;
        thrust::device_vector<float> cumulative_sum(n_points);
        thrust::inclusive_scan(thrust::device, h_distances.begin(), h_distances.end(), cumulative_sum.begin());

        int selected_point = thrust::lower_bound(thrust::device, cumulative_sum.begin(), cumulative_sum.end(), random_value) - cumulative_sum.begin();

        // Copy selected point to centroids
        cudaMemcpy(&d_centroids[current_cluster * dims], &d_points[selected_point * dims],
                   dims * sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

void adaptiveInitializeCentroids(float* d_points, float* d_centroids, int n_points, int n_clusters, int dims, InitMethod method) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    curandState* d_rng_state;
    cudaMalloc(&d_rng_state, n_clusters * sizeof(curandState));

    int block_size = BLOCK_SIZE;
    int grid_size = (n_points + block_size - 1) / block_size;

    // Setup RNG
    setupRNG<<<grid_size, block_size, 0, stream>>>(d_rng_state, time(NULL), n_clusters);

    if (method == RANDOM) {
        randomInitialization<<<(n_clusters + block_size - 1) / block_size, block_size, 0, stream>>>
            (d_points, d_centroids, n_points, n_clusters, dims, d_rng_state);
    } else if (method == KMEANS_PLUS_PLUS) {
        // Initialize first centroid randomly
        randomInitialization<<<1, 1, 0, stream>>>(d_points, d_centroids, n_points, 1, dims, d_rng_state);

        thrust::device_vector<float> d_distances(n_points);
        thrust::device_vector<int> d_nearest(n_points);

        for (int k = 1; k < n_clusters; k++) {
            // Compute distances
            size_t sharedMemSize = (block_size / WARP_SIZE) * dims * sizeof(float) * 2;  // For both point and centroid
            advancedComputeDistancesToNearest<<<grid_size, block_size, sharedMemSize, stream>>>
                (d_points, d_centroids, thrust::raw_pointer_cast(d_distances.data()),
                 thrust::raw_pointer_cast(d_nearest.data()), n_points, k, dims);

            // Adaptive selection based on GPU capabilities
            hybridKmeansppSelection(d_points, d_centroids, thrust::raw_pointer_cast(d_distances.data()),
                                    n_points, n_clusters, dims, k, d_rng_state);
        }
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(d_rng_state);
}