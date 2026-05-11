#include <iostream>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include "morton.cuh"
#include "tree.cuh"
#include "bvh.cuh"
using namespace std;

int main() {
    int n = 1024;
    size_t vecSize    = n * sizeof(Vec3);
    size_t mortonSize = n * sizeof(unsigned int);

    Vec3*         h_points  = (Vec3*)malloc(vecSize);
    unsigned int* h_mortons = (unsigned int*)malloc(mortonSize);

    for (int i = 0; i < n; i++) {
        h_points[i].x = (float)i / n;
        h_points[i].y = (float)i / n;
        h_points[i].z = (float)i / n;
    }

    Vec3*         d_points;
    unsigned int* d_mortons;
    cudaMalloc(&d_points,  vecSize);
    cudaMalloc(&d_mortons, mortonSize);
    cudaMemcpy(d_points, h_points, vecSize, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    morton3D<<<blocks, threads>>>(d_points, d_mortons, n);

    thrust::device_ptr<unsigned int> ptr(d_mortons);
    thrust::sort(ptr, ptr + n);

    // Allocate and build BVH
    BVH bvh = bvhAlloc(n);
    int treeBlocks = (n - 1 + threads - 1) / threads;
    buildTree<<<treeBlocks, threads>>>(d_mortons, bvh);
    cudaDeviceSynchronize();

    // Verify — copy back and print a few nodes
    int* h_left      = (int*)malloc((n-1) * sizeof(int));
    int* h_right     = (int*)malloc((n-1) * sizeof(int));
    int* h_parent    = (int*)malloc((n-1) * sizeof(int));
    int* h_leafParent= (int*)malloc( n    * sizeof(int));
    bool* h_leftIsLeaf  = (bool*)malloc((n-1) * sizeof(bool));
    bool* h_rightIsLeaf = (bool*)malloc((n-1) * sizeof(bool));
    int* h_first     = (int*)malloc((n-1) * sizeof(int));
    int* h_last      = (int*)malloc((n-1) * sizeof(int));

    cudaMemcpy(h_left,        bvh.left,        (n-1)*sizeof(int),  cudaMemcpyDeviceToHost);
    cudaMemcpy(h_right,       bvh.right,       (n-1)*sizeof(int),  cudaMemcpyDeviceToHost);
    cudaMemcpy(h_parent,      bvh.parent,      (n-1)*sizeof(int),  cudaMemcpyDeviceToHost);
    cudaMemcpy(h_leafParent,  bvh.leafParent,  n    *sizeof(int),  cudaMemcpyDeviceToHost);
    cudaMemcpy(h_leftIsLeaf,  bvh.leftIsLeaf,  (n-1)*sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rightIsLeaf, bvh.rightIsLeaf, (n-1)*sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_first,       bvh.first,       (n-1)*sizeof(int),  cudaMemcpyDeviceToHost);
    cudaMemcpy(h_last,        bvh.last,        (n-1)*sizeof(int),  cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        cout << "Node " << i
             << " | range=[" << h_first[i] << ", " << h_last[i] << "]"
             << " | parent=" << h_parent[i]
             << " | left="   << h_left[i]
             << (h_leftIsLeaf[i]  ? " (leaf)" : " (internal)")
             << " | right="  << h_right[i]
             << (h_rightIsLeaf[i] ? " (leaf)" : " (internal)")
             << endl;
    }

    // Cleanup
    bvhFree(bvh);
    cudaFree(d_points);
    cudaFree(d_mortons);
    free(h_points);
    free(h_mortons);
    free(h_left);     free(h_right);
    free(h_parent);   free(h_leafParent);
    free(h_leftIsLeaf); free(h_rightIsLeaf);
    free(h_first);    free(h_last);
    return 0;
}