#include <iostream>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include "morton.cuh"
#include "tree.cuh"
#include "bvh.cuh"
using namespace std;

int main() {
    int n = 1024;
    size_t vecSize    = n * sizeof(Vec3);
    size_t mortonSize = n * sizeof(unsigned int);

    // ── Host memory ───────────────────────────────────────────────────────
    Vec3*         h_points  = (Vec3*)malloc(vecSize);
    unsigned int* h_mortons = (unsigned int*)malloc(mortonSize);

    for (int i = 0; i < n; i++) {
        h_points[i].x = (float)i / n;
        h_points[i].y = (float)i / n;
        h_points[i].z = (float)i / n;
    }

    // ── Device memory ─────────────────────────────────────────────────────
    Vec3*         d_points;
    unsigned int* d_mortons;
    cudaMalloc(&d_points,  vecSize);
    cudaMalloc(&d_mortons, mortonSize);
    cudaMemcpy(d_points, h_points, vecSize, cudaMemcpyHostToDevice);

    // ── Morton codes ──────────────────────────────────────────────────────
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    morton3D<<<blocks, threads>>>(d_points, d_mortons, n);

    // ── Sort ──────────────────────────────────────────────────────────────
    thrust::device_ptr<unsigned int> ptr(d_mortons);
    thrust::sort(ptr, ptr + n);

    // ── Build BVH ─────────────────────────────────────────────────────────
    BVH bvh = bvhAlloc(n);
    int treeBlocks = (n - 1 + threads - 1) / threads;
    buildTree<<<treeBlocks, threads>>>(d_mortons, bvh);
    cudaDeviceSynchronize();

    // ── Masses (uniform) ──────────────────────────────────────────────────
    float* d_masses;
    cudaMalloc(&d_masses, n * sizeof(float));
    thrust::device_ptr<float> massPtr(d_masses);
    thrust::fill(massPtr, massPtr + n, 1.0f);

    // ── NodeData + flags ──────────────────────────────────────────────────
    NodeData* d_nodeData;
    int*      d_flags;
    cudaMalloc(&d_nodeData, (2 * n - 1) * sizeof(NodeData));
    cudaMalloc(&d_flags,    (n - 1)     * sizeof(int));
    cudaMemset(d_flags, 0,  (n - 1)     * sizeof(int));

    // ── Init leaves ───────────────────────────────────────────────────────
    int leafBlocks = (n + threads - 1) / threads;
    initLeaves<<<leafBlocks, threads>>>(d_nodeData, d_points, d_masses, n);
    cudaDeviceSynchronize();

    // ── Bottom-up AABB + CoM ──────────────────────────────────────────────
    computeAABBandCoM<<<leafBlocks, threads>>>(d_nodeData, bvh, d_flags, n);
    cudaDeviceSynchronize();

    // ── Verify root ───────────────────────────────────────────────────────
    NodeData h_root;
    cudaMemcpy(&h_root, &d_nodeData[0], sizeof(NodeData), cudaMemcpyDeviceToHost);

    cout << "=== Root node ===" << endl;
    cout << "AABB x: [" << h_root.minX << ", " << h_root.maxX << "]" << endl;
    cout << "AABB y: [" << h_root.minY << ", " << h_root.maxY << "]" << endl;
    cout << "AABB z: [" << h_root.minZ << ", " << h_root.maxZ << "]" << endl;
    cout << "CoM:    (" << h_root.comX << ", " << h_root.comY << ", " << h_root.comZ << ")" << endl;
    cout << "Mass:   " << h_root.mass << endl;

    // ── Verify a few leaves ───────────────────────────────────────────────
    cout << "\n=== First 5 leaves ===" << endl;
    for (int i = 0; i < 5; i++) {
        NodeData h_leaf;
        cudaMemcpy(&h_leaf, &d_nodeData[n - 1 + i], sizeof(NodeData), cudaMemcpyDeviceToHost);
        cout << "Leaf " << i
             << " | pos=(" << h_leaf.comX << ", " << h_leaf.comY << ", " << h_leaf.comZ << ")"
             << " | mass=" << h_leaf.mass
             << endl;
    }

    // ── Verify a few internal nodes ───────────────────────────────────────
    cout << "\n=== First 5 internal nodes ===" << endl;
    for (int i = 0; i < 5; i++) {
        NodeData h_node;
        cudaMemcpy(&h_node, &d_nodeData[i], sizeof(NodeData), cudaMemcpyDeviceToHost);
        cout << "Node " << i
             << " | AABB x:[" << h_node.minX << ", " << h_node.maxX << "]"
             << " | CoM=("    << h_node.comX << ", " << h_node.comY << ", " << h_node.comZ << ")"
             << " | mass="    << h_node.mass
             << endl;
    }

    // ── Verify BVH structure ──────────────────────────────────────────────
    cout << "\n=== First 10 internal nodes (structure) ===" << endl;
    int* h_left       = (int*)malloc((n-1)  * sizeof(int));
    int* h_right      = (int*)malloc((n-1)  * sizeof(int));
    int* h_parent     = (int*)malloc((n-1)  * sizeof(int));
    int* h_leafParent = (int*)malloc( n     * sizeof(int));
    bool* h_leftIsLeaf  = (bool*)malloc((n-1) * sizeof(bool));
    bool* h_rightIsLeaf = (bool*)malloc((n-1) * sizeof(bool));
    int* h_first      = (int*)malloc((n-1)  * sizeof(int));
    int* h_last       = (int*)malloc((n-1)  * sizeof(int));

    cudaMemcpy(h_left,        bvh.left,        (n-1)*sizeof(int),  cudaMemcpyDeviceToHost);
    cudaMemcpy(h_right,       bvh.right,       (n-1)*sizeof(int),  cudaMemcpyDeviceToHost);
    cudaMemcpy(h_parent,      bvh.parent,      (n-1)*sizeof(int),  cudaMemcpyDeviceToHost);
    cudaMemcpy(h_leafParent,  bvh.leafParent,   n   *sizeof(int),  cudaMemcpyDeviceToHost);
    cudaMemcpy(h_leftIsLeaf,  bvh.leftIsLeaf,  (n-1)*sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rightIsLeaf, bvh.rightIsLeaf, (n-1)*sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_first,       bvh.first,       (n-1)*sizeof(int),  cudaMemcpyDeviceToHost);
    cudaMemcpy(h_last,        bvh.last,        (n-1)*sizeof(int),  cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        cout << "Node " << i
             << " | range=[" << h_first[i] << ", " << h_last[i] << "]"
             << " | parent=" << h_parent[i]
             << " | left="   << h_left[i]
             << (h_leftIsLeaf[i]  ? " (leaf)" : " (internal)")
             << " | right="  << h_right[i]
             << (h_rightIsLeaf[i] ? " (leaf)" : " (internal)")
             << endl;
    }

    // ── Cleanup ───────────────────────────────────────────────────────────
    bvhFree(bvh);
    cudaFree(d_points);
    cudaFree(d_mortons);
    cudaFree(d_masses);
    cudaFree(d_nodeData);
    cudaFree(d_flags);
    free(h_points);
    free(h_mortons);
    free(h_left);
    free(h_right);
    free(h_parent);
    free(h_leafParent);
    free(h_leftIsLeaf);
    free(h_rightIsLeaf);
    free(h_first);
    free(h_last);

    return 0;
}