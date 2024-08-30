#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>
#include "../include/initialization.h"
#include "../include/distance.h"

__global__ void setupRNG(curandState *state, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void randomInitialization(float* points, float* centroids, int n_points, int n_clusters, int dims, curandState* state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_clusters) {
        int random_point = curand(&state[idx]) % n_points;
        for (int d = 0; d < dims; d++) {
            centroids[idx * dims + d] = points[random_point * dims + d];
        }
    }
}

__global__ void computeDistancesToNearest(float* points, float* centroids, float* distances, int* nearest, int n_points, int n_clusters, int dims) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_points) {
        float min_dist = FLT_MAX;
        int closest = 0;
        for (int c = 0; c < n_clusters; c++) {
            float dist = euclideanDistance(&points[idx * dims], &centroids[c * dims], dims);
            if (dist < min_dist) {
                min_dist = dist;
                closest = c;
            }
        }
        distances[idx] = min_dist * min_dist;  // Store squared distance
        nearest[idx] = closest;
    }
}

__global__ void kmeansppSelection(float* points, float* centroids, float* distances, int* nearest, int n_points, int n_clusters, int dims, int current_cluster, curandState* state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        float sum_distances = 0;
        for (int i = 0; i < n_points; i++) {
            sum_distances += distances[i];
        }
        
        float random_value = curand_uniform(&state[0]) * sum_distances;
        float cumulative_sum = 0;
        int selected_point = -1;
        
        for (int i = 0; i < n_points; i++) {
            cumulative_sum += distances[i];
            if (cumulative_sum >= random_value) {
                selected_point = i;
                break;
            }
        }
        
        for (int d = 0; d < dims; d++) {
            centroids[current_cluster * dims + d] = points[selected_point * dims + d];
        }
    }
}

void initializeCentroids(float* d_points, float* d_centroids, int n_points, int n_clusters, int dims, InitMethod method) {
    curandState* d_rng_state;
    cudaMalloc(&d_rng_state, n_points * sizeof(curandState));
    
    int block_size = 256;
    int grid_size = (n_points + block_size - 1) / block_size;
    
    setupRNG<<<grid_size, block_size>>>(d_rng_state, time(NULL));
    
    if (method == RANDOM) {
        randomInitialization<<<(n_clusters + block_size - 1) / block_size, block_size>>>(d_points, d_centroids, n_points, n_clusters, dims, d_rng_state);
    } else if (method == KMEANS_PLUS_PLUS) {
        // Initialize first centroid randomly
        randomInitialization<<<1, 1>>>(d_points, d_centroids, n_points, 1, dims, d_rng_state);
        
        float* d_distances;
        int* d_nearest;
        cudaMalloc(&d_distances, n_points * sizeof(float));
        cudaMalloc(&d_nearest, n_points * sizeof(int));
        
        for (int k = 1; k < n_clusters; k++) {
            computeDistancesToNearest<<<grid_size, block_size>>>(d_points, d_centroids, d_distances, d_nearest, n_points, k, dims);
            kmeansppSelection<<<1, 1>>>(d_points, d_centroids, d_distances, d_nearest, n_points, n_clusters, dims, k, d_rng_state);
        }
        
        cudaFree(d_distances);
        cudaFree(d_nearest);
    }
    
    cudaFree(d_rng_state);
}