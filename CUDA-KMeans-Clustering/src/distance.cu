#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <type_traits>
#include <cmath>
#include "../include/distance.h"

namespace cg = cooperative_groups;

template <typename T>
constexpr T BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

// CUDA error checking wrapper
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Warp-level reduction
template<typename T, typename BinaryOp>
__device__ __forceinline__ T warpReduce(T val, BinaryOp op) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val = op(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Distance functors
struct EuclideanDistance {
    __device__ __forceinline__ float operator()(float a, float b) const {
        float diff = a - b;
        return diff * diff;
    }
    
    __device__ __forceinline__ float finalize(float sum) const {
        return std::sqrt(sum);
    }
};

struct ManhattanDistance {
    __device__ __forceinline__ float operator()(float a, float b) const {
        return std::abs(a - b);
    }
    
    __device__ __forceinline__ float finalize(float sum) const {
        return sum;
    }
};

// Generic distance computation
template<typename DistanceFunc, int UnrollFactor = 4>
__device__ float computeDistance(const float* __restrict__ a, const float* __restrict__ b, int dims, DistanceFunc func) {
    float sum = 0.0f;
    
    #pragma unroll UnrollFactor
    for (int i = 0; i < dims; i++) {
        sum += func(a[i], b[i]);
    }
    
    return sum;
}

// Advanced distance matrix computation kernel
template<typename DistanceFunc>
__global__ void advancedComputeDistanceMatrix(
    const float* __restrict__ points,
    const float* __restrict__ centroids,
    float* __restrict__ distance_matrix,
    int n_points, int n_centroids, int dims
) {
    extern __shared__ float s_centroids[];

    const int tid = threadIdx.x;
    const int point_idx = blockIdx.x * blockDim.x + tid;

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WARP_SIZE>(block);

    // Collaborative loading of centroids into shared memory
    for (int i = tid; i < n_centroids * dims; i += blockDim.x) {
        s_centroids[i] = centroids[i];
    }
    block.sync();

    if (point_idx < n_points) {
        const float* point = &points[point_idx * dims];
        DistanceFunc distance_func;
        
        for (int c = 0; c < n_centroids; c++) {
            const float* centroid = &s_centroids[c * dims];
            
            float distance = computeDistance(point, centroid, dims, distance_func);
            
            // Warp-level reduction
            distance = warpReduce(distance, [](float a, float b) { return a + b; });
            
            if (warp.thread_rank() == 0) {
                distance_matrix[point_idx * n_centroids + c] = distance_func.finalize(distance);
            }
        }
    }
}

// Host-side distance calculation function
template<typename DistanceMetric>
void calculateDistances(
    float* d_points, float* d_centroids, float* d_distance_matrix,
    int n_points, int n_centroids, int dims,
    cudaStream_t stream = nullptr
) {
    static_assert(std::is_same_v<DistanceMetric, EuclideanDistance> || 
                  std::is_same_v<DistanceMetric, ManhattanDistance>,
                  "Unsupported distance metric");

    const int block_size = BLOCK_SIZE<int>;
    const int grid_size = (n_points + block_size - 1) / block_size;
    
    const size_t shared_mem_size = n_centroids * dims * sizeof(float);

    advancedComputeDistanceMatrix<DistanceMetric><<<grid_size, block_size, shared_mem_size, stream>>>(
        d_points, d_centroids, d_distance_matrix, n_points, n_centroids, dims
    );

    CUDA_CHECK(cudaGetLastError());
    if (stream == nullptr) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// Dispatcher function to choose the appropriate distance metric
void dispatchDistanceCalculation(
    float* d_points, float* d_centroids, float* d_distance_matrix,
    int n_points, int n_centroids, int dims, DistanceMetric metric,
    cudaStream_t stream = nullptr
) {
    switch (metric) {
        case EUCLIDEAN:
            calculateDistances<EuclideanDistance>(d_points, d_centroids, d_distance_matrix,
                                                  n_points, n_centroids, dims, stream);
            break;
        case MANHATTAN:
            calculateDistances<ManhattanDistance>(d_points, d_centroids, d_distance_matrix,
                                                  n_points, n_centroids, dims, stream);
            break;
        default:
            throw std::runtime_error("Unsupported distance metric");
    }
}

// Example usage
int main() {
    // Device memory pointers and problem dimensions
    float *d_points, *d_centroids, *d_distance_matrix;
    int n_points = 1000000, n_centroids = 100, dims = 128;

    // Allocate device memory and copy data (not shown for brevity)

    // Create CUDA stream for asynchronous execution
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Calculate distances using Euclidean metric
    dispatchDistanceCalculation(d_points, d_centroids, d_distance_matrix,
                                n_points, n_centroids, dims, EUCLIDEAN, stream);

    // Calculate distances using Manhattan metric
    dispatchDistanceCalculation(d_points, d_centroids, d_distance_matrix,
                                n_points, n_centroids, dims, MANHATTAN, stream);

    // Clean up
    CUDA_CHECK(cudaStreamDestroy(stream));
    // Free device memory (not shown for brevity)

    return 0;
}