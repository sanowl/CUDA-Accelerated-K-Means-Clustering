#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "../include/kmeans.h"
#include "../include/distance.h"
#include "../include/initialization.h"

namespace cg = cooperative_groups;

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define MAX_SHARED_MEMORY 48000 // Adjust based on your GPU's capabilities

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Advanced distance calculation using vector operations and template metaprogramming
template<int VectorSize = 4>
__device__ __forceinline__ float advancedEuclideanDistance(const float* __restrict__ a, const float* __restrict__ b, int dims) {
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < dims; i += VectorSize) {
        float4 diff;
        diff.x = a[i] - b[i];
        if constexpr (VectorSize > 1) diff.y = (i + 1 < dims) ? a[i + 1] - b[i + 1] : 0.0f;
        if constexpr (VectorSize > 2) diff.z = (i + 2 < dims) ? a[i + 2] - b[i + 2] : 0.0f;
        if constexpr (VectorSize > 3) diff.w = (i + 3 < dims) ? a[i + 3] - b[i + 3] : 0.0f;
        sum += diff.x * diff.x + diff.y * diff.y + diff.z * diff.z + diff.w * diff.w;
    }
    return sqrtf(sum);
}

// Warp-level reduction
template<typename T>
__device__ __forceinline__ T warpReduceMin(T val, int* idx) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        T other_val = __shfl_down_sync(0xffffffff, val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, *idx, offset);
        if (other_val < val) {
            val = other_val;
            *idx = other_idx;
        }
    }
    return val;
}

// Kernel for assigning points to clusters using shared memory, warp-level primitives, and dynamic parallelism
template<int ThreadsPerPoint = 32>
__global__ void assignClustersAdvanced(const float* __restrict__ d_points, const float* __restrict__ d_centroids, 
                                       int* __restrict__ d_assignments, int n_points, int n_clusters, int n_dims) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<ThreadsPerPoint> tile = cg::tiled_partition<ThreadsPerPoint>(block);

    int point_idx = blockIdx.x;
    
    extern __shared__ float shared_mem[];
    float* s_point = shared_mem;
    float* s_centroid = &shared_mem[n_dims];

    if (point_idx < n_points) {
        // Load point data into shared memory
        for (int d = tile.thread_rank(); d < n_dims; d += ThreadsPerPoint) {
            s_point[d] = d_points[point_idx * n_dims + d];
        }
        tile.sync();

        float min_dist = FLT_MAX;
        int closest_centroid = 0;

        // Compute distances and find minimum
        for (int c = 0; c < n_clusters; c++) {
            // Load centroid data
            for (int d = tile.thread_rank(); d < n_dims; d += ThreadsPerPoint) {
                s_centroid[d] = d_centroids[c * n_dims + d];
            }
            tile.sync();

            float dist = advancedEuclideanDistance<4>(s_point, s_centroid, n_dims);

            // Warp-level reduction to find minimum distance
            int local_c = c;
            dist = warpReduceMin(dist, &local_c);

            if (tile.thread_rank() == 0) {
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_centroid = local_c;
                }
            }
        }

        // Write result
        if (tile.thread_rank() == 0) {
            d_assignments[point_idx] = closest_centroid;
        }
    }
}

// Kernel for updating centroids using atomic operations, shared memory, and cooperative groups
template<int ThreadsPerCluster = 256>
__global__ void updateCentroidsAdvanced(const float* __restrict__ d_points, float* __restrict__ d_centroids, 
                                        const int* __restrict__ d_assignments, int* __restrict__ d_cluster_sizes, 
                                        int n_points, int n_clusters, int n_dims) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<ThreadsPerCluster> tile = cg::tiled_partition<ThreadsPerCluster>(block);

    int cluster = blockIdx.x;
    
    extern __shared__ float s_sum[];
    float* s_count = (float*)&s_sum[n_dims];

    // Initialize shared memory
    for (int d = tile.thread_rank(); d < n_dims; d += ThreadsPerCluster) {
        s_sum[d] = 0.0f;
    }
    if (tile.thread_rank() == 0) {
        s_count[0] = 0.0f;
    }
    tile.sync();

    // Accumulate points
    for (int p = tile.thread_rank(); p < n_points; p += ThreadsPerCluster) {
        if (d_assignments[p] == cluster) {
            for (int d = 0; d < n_dims; d++) {
                atomicAdd(&s_sum[d], d_points[p * n_dims + d]);
            }
            atomicAdd(&s_count[0], 1.0f);
        }
    }
    tile.sync();

    // Update centroid
    float count = s_count[0];
    if (count > 0) {
        for (int d = tile.thread_rank(); d < n_dims; d += ThreadsPerCluster) {
            d_centroids[cluster * n_dims + d] = s_sum[d] / count;
        }
        if (tile.thread_rank() == 0) {
            d_cluster_sizes[cluster] = count;
        }
    }
}

// Kernel for checking convergence using CUB for efficient reduction
__global__ void checkConvergence(const float* __restrict__ d_old_centroids, const float* __restrict__ d_new_centroids, 
                                 float* __restrict__ d_max_change, int n_clusters, int n_dims) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float max_diff = 0.0f;

    if (tid < n_clusters * n_dims) {
        float diff = fabsf(d_new_centroids[tid] - d_old_centroids[tid]);
        max_diff = diff;
    }

    float block_max = BlockReduce(temp_storage).Reduce(max_diff, cub::Max());

    if (threadIdx.x == 0) {
        atomicMax(d_max_change, block_max);
    }
}

// Advanced K-means clustering function
void kMeansClusteringAdvanced(float* h_points, float* h_centroids, int* h_assignments, int n_points, int n_clusters, int n_dims, int max_iterations, float tolerance) {
    // Allocate device memory
    float *d_points, *d_centroids, *d_old_centroids, *d_max_change;
    int *d_assignments, *d_cluster_sizes;

    CUDA_CHECK(cudaMalloc((void**)&d_points, n_points * n_dims * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_centroids, n_clusters * n_dims * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_old_centroids, n_clusters * n_dims * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_assignments, n_points * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_cluster_sizes, n_clusters * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_max_change, sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_points, h_points, n_points * n_dims * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids, n_clusters * n_dims * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize centroids using K-means++
    initializeCentroidsKmeanspp(d_points, d_centroids, n_points, n_clusters, n_dims);

    // Create CUDA streams for overlapping computation
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    // Main k-means loop
    int iter;
    for (iter = 0; iter < max_iterations; iter++) {
        // Assign points to clusters
        size_t shared_mem_size = (n_dims * 2) * sizeof(float);
        assignClustersAdvanced<32><<<n_points, 32, shared_mem_size, stream1>>>(d_points, d_centroids, d_assignments, n_points, n_clusters, n_dims);

        // Save old centroids
        CUDA_CHECK(cudaMemcpyAsync(d_old_centroids, d_centroids, n_clusters * n_dims * sizeof(float), cudaMemcpyDeviceToDevice, stream2));

        // Update centroids
        size_t update_shared_mem_size = (n_dims + 1) * sizeof(float);
        updateCentroidsAdvanced<256><<<n_clusters, 256, update_shared_mem_size, stream1>>>(d_points, d_centroids, d_assignments, d_cluster_sizes, n_points, n_clusters, n_dims);

        // Check for convergence
        CUDA_CHECK(cudaMemsetAsync(d_max_change, 0, sizeof(float), stream2));
        int conv_grid_size = (n_clusters * n_dims + BLOCK_SIZE - 1) / BLOCK_SIZE;
        checkConvergence<<<conv_grid_size, BLOCK_SIZE, 0, stream2>>>(d_old_centroids, d_centroids, d_max_change, n_clusters, n_dims);

        float h_max_change;
        CUDA_CHECK(cudaMemcpyAsync(&h_max_change, d_max_change, sizeof(float), cudaMemcpyDeviceToHost, stream2));

        // Synchronize streams
        CUDA_CHECK(cudaStreamSynchronize(stream1));
        CUDA_CHECK(cudaStreamSynchronize(stream2));

        if (h_max_change < tolerance) {
            break;
        }
    }

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_centroids, d_centroids, n_clusters * n_dims * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_assignments, d_assignments, n_points * sizeof(int), cudaMemcpyDeviceToHost));

    // Free device memory and destroy streams
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_old_centroids));
    CUDA_CHECK(cudaFree(d_assignments));
    CUDA_CHECK(cudaFree(d_cluster_sizes));
    CUDA_CHECK(cudaFree(d_max_change));
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));

    printf("K-means clustering completed in %d iterations.\n", iter + 1);
}