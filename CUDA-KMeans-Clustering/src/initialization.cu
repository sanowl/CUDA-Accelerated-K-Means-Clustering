#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <cooperative_groups.h>
#include <cuda/std/atomic>
#include "../include/initialization.h"
#include "../include/distance.h"

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define MAX_SHARED_MEMORY 48000  // Adjust based on your GPU's capabilities

namespace cg = cooperative_groups;

__global__ void setupRNG(curandState *state, unsigned long long seed, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

template<int VectorSize>
__global__ void vectorizedRandomInitialization(const float* __restrict__ points, float* __restrict__ centroids, 
                                               int n_points, int n_clusters, int dims, curandState* state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_clusters) {
        curandState localState = state[idx];
        int random_point = curand(&localState) % n_points;

        #pragma unroll
        for (int d = 0; d < dims; d += VectorSize * sizeof(float4) / sizeof(float)) {
            if (d + VectorSize * sizeof(float4) / sizeof(float) <= dims) {
                using Vector = float4[VectorSize];
                reinterpret_cast<Vector*>(&centroids[idx * dims + d])[0] = 
                    reinterpret_cast<const Vector*>(&points[random_point * dims + d])[0];
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
        float other_val = __shfl_xor_sync(0xffffffff, val, offset);
        int other_idx = __shfl_xor_sync(0xffffffff, *idx, offset);
        if (other_val < val) {
            val = other_val;
            *idx = other_idx;
        }
    }
    return val;
}

template<int ThreadsPerPoint>
__global__ void ultraFastComputeDistancesToNearest(const float* __restrict__ points, const float* __restrict__ centroids,
                                                   float* __restrict__ distances, int* __restrict__ nearest,
                                                   int n_points, int n_clusters, int dims) {
    constexpr int kPointsPerBlock = BLOCK_SIZE / ThreadsPerPoint;
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<ThreadsPerPoint> tile = cg::tiled_partition<ThreadsPerPoint>(block);

    int block_start = blockIdx.x * kPointsPerBlock;
    int point_idx = block_start + (threadIdx.x / ThreadsPerPoint);

    extern __shared__ float shared_mem[];
    float* s_centroids = shared_mem;
    
    float min_dist = FLT_MAX;
    int closest = 0;

    if (point_idx < n_points) {
        // Load point data directly from global memory
        float point_data[ThreadsPerPoint];
        #pragma unroll
        for (int d = tile.thread_rank(); d < dims; d += ThreadsPerPoint) {
            point_data[d / ThreadsPerPoint] = points[point_idx * dims + d];
        }

        // Compute distances
        for (int c = 0; c < n_clusters; c++) {
            // Collaborative loading of centroid data
            for (int d = threadIdx.x; d < dims; d += BLOCK_SIZE) {
                s_centroids[d] = centroids[c * dims + d];
            }
            block.sync();

            float dist = 0.0f;
            #pragma unroll
            for (int d = tile.thread_rank(); d < dims; d += ThreadsPerPoint) {
                float diff = point_data[d / ThreadsPerPoint] - s_centroids[d];
                dist += diff * diff;
            }

            // Warp-level reduction
            dist = cg::reduce(tile, dist, cg::plus<float>());

            if (tile.thread_rank() == 0) {
                if (dist < min_dist) {
                    min_dist = dist;
                    closest = c;
                }
            }
        }

        // Write results
        if (tile.thread_rank() == 0) {
            distances[point_idx] = min_dist;
            nearest[point_idx] = closest;
        }
    }
}

__global__ void kmeansppSelection(const float* __restrict__ points, float* __restrict__ centroids,
                                  const float* __restrict__ distances, int n_points, int n_clusters,
                                  int dims, int current_cluster, curandState* state,
                                  cub::KeyValuePair<int, float>* d_argmax) {
    cg::thread_block block = cg::this_thread_block();
    
    using BlockReduce = cub::BlockReduce<cub::KeyValuePair<int, float>, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    cub::KeyValuePair<int, float> thread_data(0, 0.0f);
    
    if (tid < n_points) {
        thread_data = cub::KeyValuePair<int, float>(tid, distances[tid]);
    }
    
    cub::KeyValuePair<int, float> aggregate = BlockReduce(temp_storage).Reduce(thread_data, cub::ArgMax());
    
    if (threadIdx.x == 0) {
        atomicMax(reinterpret_cast<unsigned long long*>(d_argmax),
                  reinterpret_cast<unsigned long long&>(aggregate));
    }
    
    block.sync();
    
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int selected_point = d_argmax->key;
        for (int d = 0; d < dims; ++d) {
            centroids[current_cluster * dims + d] = points[selected_point * dims + d];
        }
    }
}

__host__ __device__ float advancedDistance(const float* a, const float* b, int dims) {
    float dist = 0.0f;
    #pragma unroll 8
    for (int d = 0; d < dims; ++d) {
        float diff = a[d] - b[d];
        dist += diff * diff;
    }
    return dist;
}

struct DistanceFunctor {
    const float* points;
    const float* centroids;
    int dims;
    int n_clusters;
    
    __host__ __device__
    DistanceFunctor(const float* p, const float* c, int d, int nc)
        : points(p), centroids(c), dims(d), n_clusters(nc) {}
    
    __host__ __device__
    thrust::tuple<float, int> operator()(int idx) {
        float min_dist = FLT_MAX;
        int nearest = 0;
        for (int c = 0; c < n_clusters; ++c) {
            float dist = advancedDistance(&points[idx * dims], &centroids[c * dims], dims);
            if (dist < min_dist) {
                min_dist = dist;
                nearest = c;
            }
        }
        return thrust::make_tuple(min_dist, nearest);
    }
};

void hybridKmeansppSelection(const float* d_points, float* d_centroids, const float* d_distances,
                             int n_points, int n_clusters, int dims, int current_cluster, curandState* d_rng_state) {
    thrust::device_vector<cub::KeyValuePair<int, float>> d_argmax(1);
    
    kmeansppSelection<<<(n_points + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>
        (d_points, d_centroids, d_distances, n_points, n_clusters, dims, current_cluster, d_rng_state, thrust::raw_pointer_cast(d_argmax.data()));
    
    cudaDeviceSynchronize();
}

void adaptiveInitializeCentroids(float* d_points, float* d_centroids, int n_points, int n_clusters, int dims, InitMethod method) {
    cudaStream_t compute_stream, memory_stream;
    cudaStreamCreate(&compute_stream);
    cudaStreamCreate(&memory_stream);

    curandState* d_rng_state;
    cudaMalloc(&d_rng_state, n_clusters * sizeof(curandState));

    int block_size = BLOCK_SIZE;
    int grid_size = (n_points + block_size - 1) / block_size;

    // Setup RNG
    setupRNG<<<grid_size, block_size, 0, compute_stream>>>(d_rng_state, time(NULL), n_clusters);

    if (method == RANDOM) {
        constexpr int kVectorSize = 2;  // Adjust based on your GPU's vector capabilities
        vectorizedRandomInitialization<kVectorSize><<<(n_clusters + block_size - 1) / block_size, block_size, 0, compute_stream>>>
            (d_points, d_centroids, n_points, n_clusters, dims, d_rng_state);
    } else if (method == KMEANS_PLUS_PLUS) {
        // Initialize first centroid randomly
        vectorizedRandomInitialization<1><<<1, 1, 0, compute_stream>>>(d_points, d_centroids, n_points, 1, dims, d_rng_state);

        thrust::device_vector<float> d_distances(n_points);
        thrust::device_vector<int> d_nearest(n_points);

        // Determine optimal threads per point based on dimensionality
        constexpr int kThreadsPerPoint = 32;  // Adjust based on your specific use case
        
        for (int k = 1; k < n_clusters; k++) {
            // Asynchronously compute distances
            size_t sharedMemSize = dims * sizeof(float);
            ultraFastComputeDistancesToNearest<kThreadsPerPoint><<<grid_size, block_size, sharedMemSize, compute_stream>>>
                (d_points, d_centroids, thrust::raw_pointer_cast(d_distances.data()),
                 thrust::raw_pointer_cast(d_nearest.data()), n_points, k, dims);

            // Overlap computation with memory operations
            cudaMemcpyAsync(&d_centroids[k * dims], &d_points[0], dims * sizeof(float), cudaMemcpyDeviceToDevice, memory_stream);

            // Adaptive selection based on GPU capabilities
            hybridKmeansppSelection(d_points, d_centroids, thrust::raw_pointer_cast(d_distances.data()),
                                    n_points, n_clusters, dims, k, d_rng_state);

            // Synchronize streams
            cudaStreamSynchronize(compute_stream);
            cudaStreamSynchronize(memory_stream);
        }
    }

    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(memory_stream);
    cudaFree(d_rng_state);
}

// Persistent kernel for continuous processing
__global__ void persistentKMeansKernel(float* points, float* centroids, int* assignments,
                                       int n_points, int n_clusters, int dims,
                                       cuda::std::atomic<bool>* convergence_flag) {
    extern __shared__ float shared_mem[];
    float* local_centroids = shared_mem;
    float* centroid_sums = &shared_mem[n_clusters * dims];
    int* centroid_counts = reinterpret_cast<int*>(&centroid_sums[n_clusters * dims]);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int points_per_thread = (n_points + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);
    
    while (!convergence_flag->load(cuda::std::memory_order_relaxed)) {
        // Load centroids into shared memory
        for (int i = threadIdx.x; i < n_clusters * dims; i += blockDim.x) {
            local_centroids[i] = centroids[i];
        }
        __syncthreads();

        // Reset centroid sums and counts
        for (int i = threadIdx.x; i < n_clusters * dims; i += blockDim.x) {
            centroid_sums[i] = 0.0f;
        }
        for (int i = threadIdx.x; i < n_clusters; i += blockDim.x) {
            centroid_counts[i] = 0;
        }
        __syncthreads();

        // Assign points to centroids
        for (int i = 0; i < points_per_thread; ++i) {
            int point_idx = tid + i * blockDim.x * gridDim.x;
            if (point_idx < n_points) {
                float min_dist = FLT_MAX;
                int nearest = 0;
                for (int c = 0; c < n_clusters; ++c) {
                    float dist = advancedDistance(&points[point_idx * dims], &local_centroids[c * dims], dims);
                    if (dist < min_dist) {
                        min_dist = dist;
                        nearest = c;
                    }
                }
                assignments[point_idx] = nearest;

                // Update centroid sums and counts
                for (int d = 0; d < dims; ++d) {
                    atomicAdd(&centroid_sums[nearest * dims + d], points[point_idx * dims + d]);
                }
                atomicAdd(&centroid_counts[nearest], 1);
            }
        }
        __syncthreads();

        // Update centroids
        for (int i = threadIdx.x; i < n_clusters * dims; i += blockDim.x) {
            int c = i / dims;
            if (centroid_counts[c] > 0) {
                float new_value = centroid_sums[i] / centroid_counts[c];
                float old_value = local_centroids[i];
                if (fabsf(new_value - old_value) > 1e-6) {
                    convergence_flag->store(false, cuda::std::memory_order_relaxed);
                }
                local_centroids[i] = new_value;
            }
        }
       __syncthreads();

        // Write updated centroids back to global memory
        for (int i = threadIdx.x; i < n_clusters * dims; i += blockDim.x) {
            centroids[i] = local_centroids[i];
        }
    }
}

// Kernel for final assignment and inertia calculation
__global__ void finalAssignmentAndInertia(const float* __restrict__ points, const float* __restrict__ centroids,
                                          int* __restrict__ assignments, float* __restrict__ inertia,
                                          int n_points, int n_clusters, int dims) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float s_centroids[];

    // Collaborative loading of centroids into shared memory
    for (int i = threadIdx.x; i < n_clusters * dims; i += blockDim.x) {
        s_centroids[i] = centroids[i];
    }
    __syncthreads();

    float local_inertia = 0.0f;

    for (int i = tid; i < n_points; i += blockDim.x * gridDim.x) {
        float min_dist = FLT_MAX;
        int nearest = 0;

        for (int c = 0; c < n_clusters; ++c) {
            float dist = advancedDistance(&points[i * dims], &s_centroids[c * dims], dims);
            if (dist < min_dist) {
                min_dist = dist;
                nearest = c;
            }
        }

        assignments[i] = nearest;
        local_inertia += min_dist;
    }

    // Warp-level reduction of local inertia
    local_inertia = warpReduceSum(local_inertia);

    // Write out partial inertia for this block
    if (threadIdx.x % WARP_SIZE == 0) {
        atomicAdd(inertia, local_inertia);
    }
}

// Helper function for warp-level sum reduction
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Advanced K-means clustering function
void advancedKMeansClustering(float* d_points, float* d_centroids, int* d_assignments,
                              int n_points, int n_clusters, int dims, int max_iterations) {
    // Initialize centroids
    adaptiveInitializeCentroids(d_points, d_centroids, n_points, n_clusters, dims, KMEANS_PLUS_PLUS);

    // Allocate memory for convergence flag
    cuda::std::atomic<bool>* d_convergence_flag;
    cudaMalloc(&d_convergence_flag, sizeof(cuda::std::atomic<bool>));

    // Calculate grid and block sizes
    int block_size = BLOCK_SIZE;
    int grid_size = (n_points + block_size - 1) / block_size;

    // Calculate shared memory size
    size_t shared_mem_size = n_clusters * dims * sizeof(float) + // local_centroids
                             n_clusters * dims * sizeof(float) + // centroid_sums
                             n_clusters * sizeof(int);           // centroid_counts

    // Launch persistent kernel
    persistentKMeansKernel<<<grid_size, block_size, shared_mem_size>>>(
        d_points, d_centroids, d_assignments, n_points, n_clusters, dims, d_convergence_flag);

    // Wait for kernel to finish or reach max iterations
    for (int iter = 0; iter < max_iterations; ++iter) {
        bool host_convergence_flag = false;
        cudaMemcpyAsync(&host_convergence_flag, d_convergence_flag, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        if (host_convergence_flag) {
            break;
        }

        // Reset convergence flag for next iteration
        cudaMemsetAsync(d_convergence_flag, false, sizeof(bool));
    }

    // Allocate memory for inertia
    float* d_inertia;
    cudaMalloc(&d_inertia, sizeof(float));
    cudaMemset(d_inertia, 0, sizeof(float));

    // Launch final assignment and inertia calculation
    size_t final_shared_mem_size = n_clusters * dims * sizeof(float);
    finalAssignmentAndInertia<<<grid_size, block_size, final_shared_mem_size>>>(
        d_points, d_centroids, d_assignments, d_inertia, n_points, n_clusters, dims);

    // Clean up
    cudaFree(d_convergence_flag);
    cudaFree(d_inertia);
}

// Main function to demonstrate usage
int main(int argc, char** argv) {
    // Parse command line arguments or set default values
    int n_points = 1000000;
    int n_clusters = 100;
    int dims = 128;
    int max_iterations = 100;

    // Allocate memory on host
    float* h_points = new float[n_points * dims];
    float* h_centroids = new float[n_clusters * dims];
    int* h_assignments = new int[n_points];

    // Initialize points with random data
    for (int i = 0; i < n_points * dims; ++i) {
        h_points[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate memory on device
    float *d_points, *d_centroids;
    int *d_assignments;
    cudaMalloc(&d_points, n_points * dims * sizeof(float));
    cudaMalloc(&d_centroids, n_clusters * dims * sizeof(float));
    cudaMalloc(&d_assignments, n_points * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_points, h_points, n_points * dims * sizeof(float), cudaMemcpyHostToDevice);

    // Perform K-means clustering
    advancedKMeansClustering(d_points, d_centroids, d_assignments, n_points, n_clusters, dims, max_iterations);

    // Copy results back to host
    cudaMemcpy(h_centroids, d_centroids, n_clusters * dims * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_assignments, d_assignments, n_points * sizeof(int), cudaMemcpyDeviceToHost);

    // Clean up
    delete[] h_points;
    delete[] h_centroids;
    delete[] h_assignments;
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_assignments);

    return 0;
}