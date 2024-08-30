#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <memory>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "../include/kmeans.h"
#include "../include/initialization.h"
#include "../include/utils.h"

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(error) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " <input_file> <num_clusters> <max_iterations> <initialization_method>\n"
              << "Initialization methods: 0 for Random, 1 for K-means++\n";
}

class KMeansRunner {
private:
    std::vector<float> points;
    std::vector<float> centroids;
    std::vector<int> assignments;
    int numPoints, numDimensions, numClusters, maxIterations;
    InitMethod initMethod;

    // CUDA device pointers
    float *d_points, *d_centroids;
    int *d_assignments;

    // CUDA stream for asynchronous operations
    cudaStream_t stream;

public:
    KMeansRunner(const char* inputFile, int clusters, int iterations, InitMethod method)
        : numClusters(clusters), maxIterations(iterations), initMethod(method) {
        if (!loadDataFromFile(inputFile, points, numPoints, numDimensions)) {
            throw std::runtime_error("Failed to load data from file.");
        }

        centroids.resize(numClusters * numDimensions);
        assignments.resize(numPoints);

        std::cout << "Loaded " << numPoints << " points with " << numDimensions << " dimensions." << std::endl;

        // Create CUDA stream
        CUDA_CHECK(cudaStreamCreate(&stream));

        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_points, points.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_centroids, centroids.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_assignments, assignments.size() * sizeof(int)));

        // Copy data to device asynchronously
        CUDA_CHECK(cudaMemcpyAsync(d_points, points.data(), points.size() * sizeof(float), 
                                   cudaMemcpyHostToDevice, stream));
    }

    ~KMeansRunner() {
        CUDA_CHECK(cudaFree(d_points));
        CUDA_CHECK(cudaFree(d_centroids));
        CUDA_CHECK(cudaFree(d_assignments));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    void run() {
        // Start CUDA profiling
        CUDA_CHECK(cudaProfilerStart());

        auto start = std::chrono::high_resolution_clock::now();

        // Initialize centroids
        initializeCentroids(d_points, d_centroids, numPoints, numClusters, numDimensions, initMethod, stream);

        // Run K-means clustering
        kMeansClusteringAdvanced(d_points, d_centroids, d_assignments, numPoints, numClusters, 
                                 numDimensions, maxIterations, stream);

        // Synchronize stream to ensure all operations are complete
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        // Stop CUDA profiling
        CUDA_CHECK(cudaProfilerStop());

        // Copy results back to host asynchronously
        CUDA_CHECK(cudaMemcpyAsync(centroids.data(), d_centroids, centroids.size() * sizeof(float), 
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(assignments.data(), d_assignments, assignments.size() * sizeof(int), 
                                   cudaMemcpyDeviceToHost, stream));

        // Synchronize to ensure copy is complete
        CUDA_CHECK(cudaStreamSynchronize(stream));

        printResults(elapsed.count());
        writeResultsToFile("kmeans_results.txt", centroids, assignments, numClusters, numPoints, numDimensions);
    }

private:
    void printResults(double elapsedTime) const {
        std::cout << "K-means clustering completed in " << elapsedTime << " seconds." << std::endl;
        std::cout << "Final centroids:" << std::endl;
        for (int i = 0; i < numClusters; ++i) {
            std::cout << "Centroid " << i << ": ";
            std::for_each(centroids.begin() + i * numDimensions, 
                          centroids.begin() + (i + 1) * numDimensions,
                          [](float val) { std::cout << val << " "; });
            std::cout << std::endl;
        }
    }
};

int main(int argc, char** argv) {
    try {
        if (argc != 5) {
            printUsage(argv[0]);
            return 1;
        }

        const char* inputFile = argv[1];
        int numClusters = std::stoi(argv[2]);
        int maxIterations = std::stoi(argv[3]);
        InitMethod initMethod = static_cast<InitMethod>(std::stoi(argv[4]));

        KMeansRunner runner(inputFile, numClusters, maxIterations, initMethod);
        runner.run();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}