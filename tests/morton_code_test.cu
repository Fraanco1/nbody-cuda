#include <iostream>
#include <thrust/sort.h>
#include "morton.cuh"
#include "tree.cuh"
using namespace std;

int main() {
    int n = 1024;
    size_t vecSize    = n * sizeof(Vec3);
    size_t mortonSize = n * sizeof(unsigned int);

    // Allocate host memory
    Vec3*         h_points  = (Vec3*)malloc(vecSize);
    unsigned int* h_mortons = (unsigned int*)malloc(mortonSize); // fixed

    // Fill points
    for (int i = 0; i < n; i++) {
        h_points[i].x = (float)i / n;
        h_points[i].y = (float)i / n;
        h_points[i].z = (float)i / n;
    }

    // Allocate device memory
    Vec3*         d_points;
    unsigned int* d_mortons;
    cudaMalloc(&d_points,  vecSize);
    cudaMalloc(&d_mortons, mortonSize);

    // Copy to device
    cudaMemcpy(d_points, h_points, vecSize, cudaMemcpyHostToDevice);

    // Compute Morton codes
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    morton3D<<<blocks, threadsPerBlock>>>(d_points, d_mortons, n);

    // Sort on GPU
    thrust::sort(d_mortons.begin(), d_mortons.end());

    // Copy back
    cudaMemcpy(h_mortons, d_mortons, mortonSize, cudaMemcpyDeviceToHost);

    // Print
    for (int i = 0; i < n; i++) {
        cout << i << " -> " << h_mortons[i] << endl;
    }

    // Free
    cudaFree(d_points);
    cudaFree(d_mortons);
    free(h_points);
    free(h_mortons);
    return 0;
}