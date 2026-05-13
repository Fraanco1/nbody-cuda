#pragma once
#include "morton.cuh"

// Per-node payload: AABB + center of mass + total mass.
// Stored for every node in the flat array (both internal and leaf).
struct NodeData {
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
    float comX, comY, comZ;
    float mass;
};

// A lightweight view of the BVH's topology buffers, cheap to copy into a kernel.
// All pointers are device pointers owned by Tree below.
//
// Index convention (matters for traversal later):
//   internal nodes live at indices [0, n-1)
//   leaves        live at indices [n-1, 2n-1)
// The leaf for body i (in sorted order) is at index (n - 1 + i).
struct BVHArrays {
    int*  parent;       // [n-1]   parent of each internal node (root has -1)
    int*  left;         // [n-1]   left  child index (into the full flat array)
    int*  right;        // [n-1]   right child index (into the full flat array)
    bool* leftIsLeaf;   // [n-1]
    bool* rightIsLeaf;  // [n-1]
    int*  first;        // [n-1]   first index of this node's range
    int*  last;         // [n-1]   last  index of this node's range
    int*  leafParent;   // [n]     parent internal node of each leaf
    int   n;            // number of bodies (and number of leaves)
};

// Owns all device memory for a BVH over n bodies, and knows how to (re)build it.
//
// Usage:
//   Tree tree(n);
//   tree.rebuild(d_positions, d_masses);   // call once per timestep
//   // pass tree.arrays() and tree.nodeData() to your force kernel
//
class Tree {
public:
    explicit Tree(int n);
    ~Tree();

    // No copying — owns GPU memory.
    Tree(const Tree&)            = delete;
    Tree& operator=(const Tree&) = delete;

    // Build (or rebuild) the tree from current body positions and masses.
    // Positions must already be normalized into [0,1]^3 by the caller.
    // Body order in `positions` / `masses` is permuted by Morton sorting,
    // so after this call, the caller's arrays reflect the sorted order.
    void rebuild(Vec3* positions, float* masses);

    // Accessors for use by force kernels and inspection code.
    BVHArrays arrays()   const { return arrays_; }
    NodeData* nodeData() const { return nodeData_; }
    int       n()        const { return n_; }

private:
    int           n_;
    BVHArrays     arrays_;       // device pointers, owned here
    NodeData*     nodeData_;     // [2n-1] device array
    unsigned int* mortonCodes_;  // [n]    device scratch
    int*          flags_;        // [n-1]  device scratch (atomic counters)
};

// ── Kernel declarations (implemented in tree.cu / bvh.cu) ──────────────────
__global__ void buildTree(unsigned int* mortons, BVHArrays bvh);

__global__ void initLeaves(
    NodeData* nodeData,
    Vec3*     positions,
    float*    masses,
    int       n);

__global__ void computeAABBandCoM(
    NodeData* nodeData,
    BVHArrays bvh,
    int*      flags,
    int       n);
