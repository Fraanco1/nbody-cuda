#include <iostream>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
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
    thrust::device_ptr<unsigned int> ptr(d_mortons);
    thrust::sort(ptr, ptr + n);

    int* d_gamma;
    cudaMalloc(&d_gamma, sizeof(int));
    findSplit<<<1, 1>>>(d_mortons, 0, n - 1, d_gamma);
    int gamma;
    cudaMemcpy(&gamma, d_gamma, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_gamma);
    cout << "Split index: " << gamma << endl;

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