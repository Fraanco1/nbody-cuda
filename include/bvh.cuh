#pragma once
#include "morton.cuh"

struct NodeData {
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
    float comX, comY, comZ;
    float mass;
};

struct BVH {
    int*  parent;
    int*  left;
    int*  right;
    bool* leftIsLeaf;
    bool* rightIsLeaf;
    int*  first;
    int*  last;
    int*  leafParent;
    int   n;
};

inline BVH bvhAlloc(int n) {
    BVH bvh;
    bvh.n = n;
    int internal = n - 1;
    cudaMalloc(&bvh.parent,      internal * sizeof(int));
    cudaMalloc(&bvh.left,        internal * sizeof(int));
    cudaMalloc(&bvh.right,       internal * sizeof(int));
    cudaMalloc(&bvh.leftIsLeaf,  internal * sizeof(bool));
    cudaMalloc(&bvh.rightIsLeaf, internal * sizeof(bool));
    cudaMalloc(&bvh.first,       internal * sizeof(int));
    cudaMalloc(&bvh.last,        internal * sizeof(int));
    cudaMalloc(&bvh.leafParent,  n        * sizeof(int));
    cudaMemset(bvh.parent,     -1, internal * sizeof(int));
    cudaMemset(bvh.leafParent, -1, n        * sizeof(int));
    return bvh;
}

inline void bvhFree(BVH& bvh) {
    cudaFree(bvh.parent);
    cudaFree(bvh.left);
    cudaFree(bvh.right);
    cudaFree(bvh.leftIsLeaf);
    cudaFree(bvh.rightIsLeaf);
    cudaFree(bvh.first);
    cudaFree(bvh.last);
    cudaFree(bvh.leafParent);
}

// ── Kernel declarations ───────────────────────────────────────────────────
__global__ void initLeaves(
    NodeData* nodeData,
    Vec3*     positions,
    float*    masses,
    int       n);

__global__ void computeAABBandCoM(
    NodeData* nodeData,
    BVH       bvh,
    int*      flags,
    int       n);