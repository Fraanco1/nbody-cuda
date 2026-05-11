#pragma once
#include <cstdint>

// ─── helpers ────────────────────────────────────────────────────────────────

__device__ __forceinline__ int safeClz(unsigned int x) {
    return x == 0 ? 32 : __clz(x);
}

__device__ __forceinline__ int sign(int x) {
    return x > 0 ? 1 : -1;
}

__device__ __forceinline__ int delta(unsigned int* mortons, int i, int j, int n) {
    if (j < 0 || j >= n) return -1;
    return safeClz(mortons[i] ^ mortons[j]);
}

// ─── findSplit (must come before buildTree) ──────────────────────────────────

__device__ int findSplit(
    unsigned int* sortedMortonCodes,
    int           first,
    int           last)
{
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

// ─── buildTree (uses findSplit, must come after) ─────────────────────────────

struct InternalNode {
    int  left,  right;
    bool leftIsLeaf, rightIsLeaf;
    int  first, last;
};

__global__ void buildTree(
    unsigned int* sortedMortonCodes,
    InternalNode* internalNodes,
    int           n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1) return;

    int gamma = findSplit(sortedMortonCodes, i, n - 1);

    // Left child covers [i, gamma]
    internalNodes[i].left       = gamma;
    internalNodes[i].leftIsLeaf = (i == gamma);

    // Right child covers [gamma+1, n-1]
    internalNodes[i].right       = gamma + 1;
    internalNodes[i].rightIsLeaf = (gamma + 1 == n - 1);
}