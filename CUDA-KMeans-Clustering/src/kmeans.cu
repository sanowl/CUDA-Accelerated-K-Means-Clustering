#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "../include/kmeans.h"
#include "../include/distance.h"
#include "../include/initialization.h"

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Shared memory for partial sums
extern __shared__ float shared_mem[];

// Advanced distance calculation using vector operations
__device__ float advancedEuclideanDistance(const float* __restrict__ a, const float* __restrict__ b, int dims) {
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < dims; i += 4) {
        float4 diff;
        diff.x = a[i] - b[i];
        diff.y = (i + 1 < dims) ? a[i + 1] - b[i + 1] : 0.0f;
        diff.z = (i + 2 < dims) ? a[i + 2] - b[i + 2] : 0.0f;
        diff.w = (i + 3 < dims) ? a[i + 3] - b[i + 3] : 0.0f;
        sum += diff.x * diff.x + diff.y * diff.y + diff.z * diff.z + diff.w * diff.w;
    }
    return sqrtf(sum);
}

// Kernel for assigning points to clusters using shared memory and warp-level primitives
__global__ void assignClustersAdvanced(const float* __restrict__ d_points, const float* __restrict__ d_centroids, 
                                       int* __restrict__ d_assignments, int n_points, int n_clusters, int n_dims) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    if (tid < n_points) {
        float min_dist = FLT_MAX;
        int closest_centroid = 0;

        // Load point data into shared memory
        for (int d = threadIdx.x; d < n_dims; d += blockDim.x) {
            shared_mem[d] = d_points[tid * n_dims + d];
        }
        __syncthreads();

        // Compute distances and find minimum
        for (int c = warp_id; c < n_clusters; c += blockDim.x / WARP_SIZE) {
            float dist = advancedEuclideanDistance(shared_mem, &d_centroids[c * n_dims], n_dims);
            
            // Warp-level reduction to find minimum distance
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                float other_dist = __shfl_down_sync(0xffffffff, dist, offset);
                int other_centroid = __shfl_down_sync(0xffffffff, c, offset);
                if (other_dist < dist) {
                    dist = other_dist;
                    c = other_centroid;
                }
            }

            if (lane_id == 0) {
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_centroid = c;
                }
            }
        }

        // Write result
        if (lane_id == 0) {
            d_assignments[tid] = closest_centroid;
        }
    }
}

// Kernel for updating centroids using atomic operations and shared memory
__global__ void updateCentroidsAdvanced(const float* __restrict__ d_points, float* __restrict__ d_centroids, 
                                        const int* __restrict__ d_assignments, int* __restrict__ d_cluster_sizes, 
                                        int n_points, int n_clusters, int n_dims) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cluster = blockIdx.y;

    extern __shared__ float s_sum[];
    float* s_count = (float*)&s_sum[blockDim.x];

    if (tid < n_points && cluster < n_clusters) {
        if (d_assignments[tid] == cluster) {
            for (int d = 0; d < n_dims; d++) {
                atomicAdd(&s_sum[d], d_points[tid * n_dims + d]);
            }
            atomicAdd(&s_count[0], 1.0f);
        }
    }
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            for (int d = 0; d < n_dims; d++) {
                s_sum[threadIdx.x * n_dims + d] += s_sum[(threadIdx.x + s) * n_dims + d];
            }
            s_count[threadIdx.x] += s_count[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Write results to global memory
    if (threadIdx.x == 0) {
        float count = s_count[0];
        if (count > 0) {
            for (int d = 0; d < n_dims; d++) {
                d_centroids[cluster * n_dims + d] = s_sum[d] / count;
            }
            d_cluster_sizes[cluster] = count;
        }
    }
}

// Function to check for convergence
__global__ void checkConvergence(const float* __restrict__ d_old_centroids, const float* __restrict__ d_new_centroids, 
                                 float* __restrict__ d_max_change, int n_clusters, int n_dims, float tolerance) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n_clusters * n_dims) {
        int cluster = tid / n_dims;
        int dim = tid % n_dims;
        
        float diff = fabsf(d_new_centroids[tid] - d_old_centroids[tid]);
        atomicMax(d_max_change, diff);
    }
}

// Advanced K-means clustering function
void kMeansClusteringAdvanced(float* h_points, float* h_centroids, int* h_assignments, int n_points, int n_clusters, int n_dims, int max_iterations, float tolerance) {
    // Allocate device memory
    float *d_points, *d_centroids, *d_old_centroids, *d_max_change;
    int *d_assignments, *d_cluster_sizes;
    
    cudaMalloc((void**)&d_points, n_points * n_dims * sizeof(float));
    cudaMalloc((void**)&d_centroids, n_clusters * n_dims * sizeof(float));
    cudaMalloc((void**)&d_old_centroids, n_clusters * n_dims * sizeof(float));
    cudaMalloc((void**)&d_assignments, n_points * sizeof(int));
    cudaMalloc((void**)&d_cluster_sizes, n_clusters * sizeof(int));
    cudaMalloc((void**)&d_max_change, sizeof(float));

    // Copy data to device
    cudaMemcpy(d_points, h_points, n_points * n_dims * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, n_clusters * n_dims * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize centroids using K-means++
    initializeCentroidsKmeanspp(d_points, d_centroids, n_points, n_clusters, n_dims);

    // Main k-means loop
    int iter;
    for (iter = 0; iter < max_iterations; iter++) {
        // Assign points to clusters
        int grid_size = (n_points + BLOCK_SIZE - 1) / BLOCK_SIZE;
        assignClustersAdvanced<<<grid_size, BLOCK_SIZE, n_dims * sizeof(float)>>>(d_points, d_centroids, d_assignments, n_points, n_clusters, n_dims);

        // Save old centroids
        cudaMemcpy(d_old_centroids, d_centroids, n_clusters * n_dims * sizeof(float), cudaMemcpyDeviceToDevice);

        // Update centroids
        dim3 update_grid((n_points + BLOCK_SIZE - 1) / BLOCK_SIZE, n_clusters);
        dim3 update_block(BLOCK_SIZE);
        size_t shared_mem_size = BLOCK_SIZE * (n_dims + 1) * sizeof(float);
        updateCentroidsAdvanced<<<update_grid, update_block, shared_mem_size>>>(d_points, d_centroids, d_assignments, d_cluster_sizes, n_points, n_clusters, n_dims);

        // Check for convergence
        cudaMemset(d_max_change, 0, sizeof(float));
        int conv_grid_size = (n_clusters * n_dims + BLOCK_SIZE - 1) / BLOCK_SIZE;
        checkConvergence<<<conv_grid_size, BLOCK_SIZE>>>(d_old_centroids, d_centroids, d_max_change, n_clusters, n_dims, tolerance);

        float h_max_change;
        cudaMemcpy(&h_max_change, d_max_change, sizeof(float), cudaMemcpyDeviceToHost);

        if (h_max_change < tolerance) {
            break;
        }
    }

    // Copy results back to host
    cudaMemcpy(h_centroids, d_centroids, n_clusters * n_dims * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_assignments, d_assignments, n_points * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_old_centroids);
    cudaFree(d_assignments);
    cudaFree(d_cluster_sizes);
    cudaFree(d_max_change);

    printf("K-means clustering completed in %d iterations.\n", iter + 1);
}