#pragma once
#include <cstdint>

__device__ __forceinline__ int safeClz(unsigned int x) {
    return x == 0 ? 32 : __clz(x);
}

struct InternalNode {
    int  left,  right;
    bool leftIsLeaf, rightIsLeaf;
    int  first, last;
};

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

__global__ void buildTreeLevel(
    unsigned int* mortons,
    InternalNode* nodes,
    int*          currFrontier,   // (nodeIdx, first, last) triplets
    int*          nextFrontier,
    int*          nextFrontierSize,
    int*          nodeCounter,
    int           currSize)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= currSize) return;

    // Unpack this node's triplet
    int nodeIdx = currFrontier[tid * 3 + 0];
    int first   = currFrontier[tid * 3 + 1];
    int last    = currFrontier[tid * 3 + 2];

    nodes[nodeIdx].first = first;
    nodes[nodeIdx].last  = last;

    int gamma = findSplit(mortons, first, last);

    // ── Left child [first, gamma] ─────────────────────────────────────────
    if (first == gamma) {
        // leaf
        nodes[nodeIdx].left       = gamma;
        nodes[nodeIdx].leftIsLeaf = true;
    } else {
        // internal — claim a node slot and push to next frontier
        int leftIdx = atomicAdd(nodeCounter, 1);
        nodes[nodeIdx].left       = leftIdx;
        nodes[nodeIdx].leftIsLeaf = false;

        int pos = atomicAdd(nextFrontierSize, 1) * 3;
        nextFrontier[pos + 0] = leftIdx;
        nextFrontier[pos + 1] = first;
        nextFrontier[pos + 2] = gamma;
    }

    // ── Right child [gamma+1, last] ───────────────────────────────────────
    if (gamma + 1 == last) {
        // leaf
        nodes[nodeIdx].right       = gamma + 1;
        nodes[nodeIdx].rightIsLeaf = true;
    } else {
        // internal — claim a node slot and push to next frontier
        int rightIdx = atomicAdd(nodeCounter, 1);
        nodes[nodeIdx].right       = rightIdx;
        nodes[nodeIdx].rightIsLeaf = false;

        int pos = atomicAdd(nextFrontierSize, 1) * 3;
        nextFrontier[pos + 0] = rightIdx;
        nextFrontier[pos + 1] = gamma + 1;
        nextFrontier[pos + 2] = last;
    }
}