#include <iostream>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include "morton.cuh"
#include "tree.cuh"
using namespace std;

int main() {
    int n = 1024;
    size_t vecSize        = n * sizeof(Vec3);
    size_t mortonSize     = n * sizeof(unsigned int);
    size_t internalSize   = (n - 1) * sizeof(InternalNode);

    // Allocate host memory
    Vec3*          h_points        = (Vec3*)malloc(vecSize);
    unsigned int*  h_mortons       = (unsigned int*)malloc(mortonSize);
    InternalNode*  h_internalNodes = (InternalNode*)malloc(internalSize);

    // Fill points
    for (int i = 0; i < n; i++) {
        h_points[i].x = (float)i / n;
        h_points[i].y = (float)i / n;
        h_points[i].z = (float)i / n;
    }

    // Allocate device memory
    Vec3*         d_points;
    unsigned int* d_mortons;
    InternalNode* d_internalNodes;
    cudaMalloc(&d_points,        vecSize);
    cudaMalloc(&d_mortons,       mortonSize);
    cudaMalloc(&d_internalNodes, internalSize);

    // Copy points to device
    cudaMemcpy(d_points, h_points, vecSize, cudaMemcpyHostToDevice);

    // Compute Morton codes
    int threadsPerBlock = 256;
    int blocks          = (n + threadsPerBlock - 1) / threadsPerBlock;
    morton3D<<<blocks, threadsPerBlock>>>(d_points, d_mortons, n);

    // Sort Morton codes on GPU
    thrust::device_ptr<unsigned int> ptr(d_mortons);
    thrust::sort(ptr, ptr + n);

    // Build tree
    int treeBlocks = (n - 1 + threadsPerBlock - 1) / threadsPerBlock;
    buildTree<<<treeBlocks, threadsPerBlock>>>(d_mortons, d_internalNodes, n);
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(h_mortons,       d_mortons,       mortonSize,   cudaMemcpyDeviceToHost);
    cudaMemcpy(h_internalNodes, d_internalNodes, internalSize, cudaMemcpyDeviceToHost);

    // Print tree
    cout << "Internal nodes (n-1 = " << n - 1 << "):" << endl;
    for (int i = 0; i < n - 1; i++) {
        cout << "Node " << i
             << " | left: "  << h_internalNodes[i].left
             << (h_internalNodes[i].leftIsLeaf  ? " (leaf)" : " (internal)")
             << " | right: " << h_internalNodes[i].right
             << (h_internalNodes[i].rightIsLeaf ? " (leaf)" : " (internal)")
             << endl;
    }

    // Free
    cudaFree(d_points);
    cudaFree(d_mortons);
    cudaFree(d_internalNodes);
    free(h_points);
    free(h_mortons);
    free(h_internalNodes);
    return 0;
}