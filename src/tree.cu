#include "tree.cuh"

#include <cfloat>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>

// ── AABB reduction helpers ────────────────────────────────────────────────────

struct Box {
    float x0, y0, z0, x1, y1, z1;
};

struct Vec3ToBox {
    __host__ __device__
    Box operator()(const Vec3& v) const {
        return { v.x, v.y, v.z, v.x, v.y, v.z };
    }
};

struct BoxUnion {
    __host__ __device__
    Box operator()(const Box& a, const Box& b) const {
        return {
            fminf(a.x0, b.x0), fminf(a.y0, b.y0), fminf(a.z0, b.z0),
            fmaxf(a.x1, b.x1), fmaxf(a.y1, b.y1), fmaxf(a.z1, b.z1)
        };
    }
};

// ── Karras buildTree helpers (private to this translation unit) ──────────

__device__ __forceinline__ int safeClz(unsigned int x) {
    return x == 0 ? 32 : __clz(x);
}

__device__ __forceinline__ int sign(int x) {
    return x >= 0 ? 1 : -1;
}

// Extended delta: when Morton codes are equal, break ties by body index.
// This guarantees unique keys so buildTree never produces degenerate ranges.
__device__ __forceinline__ int delta(unsigned int* mortons, int i, int j, int n) {
    if (j < 0 || j >= n) return -1;
    if (mortons[i] != mortons[j])
        return safeClz(mortons[i] ^ mortons[j]);
    return 32 + safeClz((unsigned int)(i ^ j));
}

__device__ int findSplit(unsigned int* sortedMortonCodes, int first, int last) {
    unsigned int firstCode = sortedMortonCodes[first];
    unsigned int lastCode  = sortedMortonCodes[last];

    if (firstCode == lastCode)
        return (first + last) >> 1;

    int commonPrefix = safeClz(firstCode ^ lastCode);
    int split = first;
    int step  = last - first;

    do {
        step = (step + 1) >> 1;
        int newSplit = split + step;
        if (newSplit < last) {
            unsigned int splitCode   = sortedMortonCodes[newSplit];
            int          splitPrefix = safeClz(firstCode ^ splitCode);
            if (splitPrefix > commonPrefix)
                split = newSplit;
        }
    } while (step > 1);

    return split;
}

// One thread per internal node; builds the binary radix tree from sorted codes.
__global__ void buildTree(unsigned int* mortons, BVHArrays bvh) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = bvh.n;
    if (i >= n - 1) return;

    // 1. Determine direction
    int d = sign(delta(mortons, i, i + 1, n) - delta(mortons, i, i - 1, n));

    // 2. Find upper bound
    int deltaMin = delta(mortons, i, i - d, n);
    int lmax = 2;
    while (delta(mortons, i, i + lmax * d, n) > deltaMin)
        lmax <<= 1;

    // 3. Find exact end via binary search
    int l = 0;
    for (int t = lmax >> 1; t >= 1; t >>= 1)
        if (delta(mortons, i, i + (l + t) * d, n) > deltaMin)
            l += t;

    int j     = i + l * d;
    int first = min(i, j);
    int last  = max(i, j);

    bvh.first[i] = first;
    bvh.last[i]  = last;

    // 4. Find split
    int gamma = findSplit(mortons, first, last);

    // 5. Left child
    bool leftIsLeaf   = (first == gamma);
    int  leftIdx      = leftIsLeaf ? (n - 1 + gamma) : gamma;
    bvh.left[i]       = leftIdx;
    bvh.leftIsLeaf[i] = leftIsLeaf;

    if (leftIsLeaf) bvh.leafParent[gamma] = i;
    else            bvh.parent[gamma]     = i;

    // 6. Right child
    bool rightIsLeaf   = (gamma + 1 == last);
    int  rightIdx      = rightIsLeaf ? (n - 1 + gamma + 1) : (gamma + 1);
    bvh.right[i]       = rightIdx;
    bvh.rightIsLeaf[i] = rightIsLeaf;

    if (rightIsLeaf) bvh.leafParent[gamma + 1] = i;
    else             bvh.parent[gamma + 1]     = i;
}

// ── Tree class ────────────────────────────────────────────────────────────

Tree::Tree(int n) : n_(n) {
    int internal = n - 1;

    arrays_.n = n;
    cudaMalloc(&arrays_.parent,      internal * sizeof(int));
    cudaMalloc(&arrays_.left,        internal * sizeof(int));
    cudaMalloc(&arrays_.right,       internal * sizeof(int));
    cudaMalloc(&arrays_.leftIsLeaf,  internal * sizeof(bool));
    cudaMalloc(&arrays_.rightIsLeaf, internal * sizeof(bool));
    cudaMalloc(&arrays_.first,       internal * sizeof(int));
    cudaMalloc(&arrays_.last,        internal * sizeof(int));
    cudaMalloc(&arrays_.leafParent,  n        * sizeof(int));

    cudaMalloc(&nodeData_,    (2 * n - 1) * sizeof(NodeData));
    cudaMalloc(&mortonCodes_,  n          * sizeof(unsigned int));
    cudaMalloc(&flags_,       (n - 1)     * sizeof(int));
}

Tree::~Tree() {
    cudaFree(arrays_.parent);
    cudaFree(arrays_.left);
    cudaFree(arrays_.right);
    cudaFree(arrays_.leftIsLeaf);
    cudaFree(arrays_.rightIsLeaf);
    cudaFree(arrays_.first);
    cudaFree(arrays_.last);
    cudaFree(arrays_.leafParent);
    cudaFree(nodeData_);
    cudaFree(mortonCodes_);
    cudaFree(flags_);
}

void Tree::rebuild(Vec3* positions, Vec3* velocities, float* masses) {
    const int threads = 256;
    const int nBlocks    = (n_     + threads - 1) / threads;
    const int treeBlocks = (n_ - 1 + threads - 1) / threads;

    // Reset per-rebuild scratch state.
    cudaMemset(arrays_.parent,     -1, (n_ - 1) * sizeof(int));
    cudaMemset(arrays_.leafParent, -1,  n_      * sizeof(int));
    cudaMemset(flags_,              0, (n_ - 1) * sizeof(int));

    // 1. Morton codes from current positions.
    // Compute AABB so positions can live anywhere outside [0,1]^3.
    thrust::device_ptr<Vec3> pos_ptr(positions);
    Box identity = { FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX };
    Box aabb = thrust::transform_reduce(pos_ptr, pos_ptr + n_,
                                        Vec3ToBox(), identity, BoxUnion());
    float extent = fmaxf(aabb.x1 - aabb.x0,
                   fmaxf(aabb.y1 - aabb.y0, aabb.z1 - aabb.z0));
    float invExtent = (extent > 0.0f) ? 1.0f / extent : 1.0f;
    morton3D<<<nBlocks, threads>>>(positions, mortonCodes_, n_,
                                   aabb.x0, aabb.y0, aabb.z0, invExtent);

    // 2. Sort positions, velocities, and masses by Morton code.
    //    All per-body arrays must stay in lockstep, or velocities will
    //    become mismatched with the accelerations the force kernel computes.
    thrust::device_ptr<unsigned int> keys(mortonCodes_);
    thrust::device_ptr<Vec3>         pos(positions);
    thrust::device_ptr<Vec3>         vel(velocities);
    thrust::device_ptr<float>        mas(masses);
    thrust::sort_by_key(keys, keys + n_,
                        thrust::make_zip_iterator(thrust::make_tuple(pos, vel, mas)));

    // 3. Build the binary radix tree from sorted codes.
    buildTree<<<treeBlocks, threads>>>(mortonCodes_, arrays_);

    // 4. Initialize leaf NodeData from the (now-sorted) bodies.
    initLeaves<<<nBlocks, threads>>>(nodeData_, positions, masses, n_);

    // 5. Bottom-up merge: AABB and CoM for every internal node.
    computeAABBandCoM<<<nBlocks, threads>>>(nodeData_, arrays_, flags_, n_);
}