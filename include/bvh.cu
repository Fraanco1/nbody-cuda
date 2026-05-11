#pragma once

struct BVH {
    // internal nodes — sized (n-1)
    int*  parent;
    int*  left;
    int*  right;
    bool* leftIsLeaf;
    bool* rightIsLeaf;
    int*  first;
    int*  last;

    // leaves — sized n
    int*  leafParent;

    int n;
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

    // root has no parent
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