#include <iostream>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include "morton.cuh"
#include "tree.cuh"
using namespace std;

int main() {
    int n = 1024;
    size_t vecSize      = n * sizeof(Vec3);
    size_t mortonSize   = n * sizeof(unsigned int);
    size_t internalSize = (n - 1) * sizeof(InternalNode);

    // Host memory
    Vec3*         h_points        = (Vec3*)malloc(vecSize);
    unsigned int* h_mortons       = (unsigned int*)malloc(mortonSize);
    InternalNode* h_internalNodes = (InternalNode*)malloc(internalSize);

    for (int i = 0; i < n; i++) {
        h_points[i].x = (float)i / n;
        h_points[i].y = (float)i / n;
        h_points[i].z = (float)i / n;
    }

    // Device memory
    Vec3*         d_points;
    unsigned int* d_mortons;
    InternalNode* d_internalNodes;
    cudaMalloc(&d_points,        vecSize);
    cudaMalloc(&d_mortons,       mortonSize);
    cudaMalloc(&d_internalNodes, internalSize);

    cudaMemcpy(d_points, h_points, vecSize, cudaMemcpyHostToDevice);

    // Morton codes + sort
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    morton3D<<<blocks, threads>>>(d_points, d_mortons, n);

    thrust::device_ptr<unsigned int> ptr(d_mortons);
    thrust::sort(ptr, ptr + n);

    // ── Karras build: single kernel launch, all n-1 nodes in parallel ────
    int treeBlocks = (n - 1 + threads - 1) / threads;
    buildTree<<<treeBlocks, threads>>>(d_mortons, d_internalNodes, n);
    cudaDeviceSynchronize();

    // Copy back and print
    cudaMemcpy(h_mortons,       d_mortons,       mortonSize,   cudaMemcpyDeviceToHost);
    cudaMemcpy(h_internalNodes, d_internalNodes, internalSize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n - 1; i++) {
        cout << "Node " << i
             << " | range=[" << h_internalNodes[i].first
             << ", "         << h_internalNodes[i].last << "]"
             << " | left="   << h_internalNodes[i].left
             << (h_internalNodes[i].leftIsLeaf  ? " (leaf)" : " (internal)")
             << " | right="  << h_internalNodes[i].right
             << (h_internalNodes[i].rightIsLeaf ? " (leaf)" : " (internal)")
             << endl;
    }

    // Cleanup
    cudaFree(d_points);
    cudaFree(d_mortons);
    cudaFree(d_internalNodes);
    free(h_points);
    free(h_mortons);
    free(h_internalNodes);
    return 0;
}