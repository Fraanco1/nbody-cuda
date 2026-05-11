#pragma once
#include <cstdint>
#include "bvh.cuh"

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

__global__ void buildTree(
    unsigned int* mortons,
    BVH           bvh)
{
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
    bool leftIsLeaf      = (first == gamma);
    int  leftIdx         = leftIsLeaf ? (n - 1 + gamma) : gamma;
    bvh.left[i]          = leftIdx;
    bvh.leftIsLeaf[i]    = leftIsLeaf;

    if (leftIsLeaf)
        bvh.leafParent[gamma] = i;
    else
        bvh.parent[gamma]     = i;

    // 6. Right child
    bool rightIsLeaf     = (gamma + 1 == last);
    int  rightIdx        = rightIsLeaf ? (n - 1 + gamma + 1) : (gamma + 1);
    bvh.right[i]         = rightIdx;
    bvh.rightIsLeaf[i]   = rightIsLeaf;

    if (rightIsLeaf)
        bvh.leafParent[gamma + 1] = i;
    else
        bvh.parent[gamma + 1]     = i;
}