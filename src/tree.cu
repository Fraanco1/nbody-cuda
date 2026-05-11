// Given an array of Morton codes, constructs an ordered binary radix tree
// https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf
#pragma once
#include <cstdint>

struct InternalNode {
    int  left,  right;
    bool leftIsLeaf;
    bool rightIsLeaf;
};

__global__ void buildTree(
    unsigned int* mortons,
    InternalNode* internalNodes,
    int           n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1) return;

    // 1. Determine direction
    int d = sign(delta(mortons, i, i + 1, n) - delta(mortons, i, i - 1, n));

    // 2. Compute upper bound
    int deltaMin = delta(mortons, i, i - d, n);
    int lmax = 2;
    while (delta(mortons, i, i + lmax * d, n) > deltaMin)
        lmax <<= 1;

    // 3. Find exact end with binary search
    int l = 0;
    for (int t = lmax >> 1; t >= 1; t >>= 1)
        if (delta(mortons, i, i + (l + t) * d, n) > deltaMin)
            l += t;

    int j     = i + l * d;
    int first = min(i, j);
    int last  = max(i, j);

    internalNodes[i].first = first;
    internalNodes[i].last  = last;

    // 4. Find split within THIS node's range [first, last]
    int gamma = findSplit(mortons, first, last);  // ← not (i, n-1)

    internalNodes[i].left       = gamma;
    internalNodes[i].leftIsLeaf = (first == gamma);

    internalNodes[i].right       = gamma + 1;
    internalNodes[i].rightIsLeaf = (gamma + 1 == last);
}

__device__ int findSplit(
    unsigned int* sortedMortonCodes,
    int           first,
    int           last)
{
    // Identical Morton codes => split the range in the middle.
    unsigned int firstCode = sortedMortonCodes[first];
    unsigned int lastCode  = sortedMortonCodes[last];

    if (firstCode == lastCode)
        return (first + last) >> 1;

    // Calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic.
    int commonPrefix = safeClz(firstCode ^ lastCode);

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one.
    int split = first;
    int step  = last - first;

    do
    {
        step = (step + 1) >> 1;
        int newSplit = split + step;

        if (newSplit < last)
        {
            unsigned int splitCode   = sortedMortonCodes[newSplit];
            int          splitPrefix = safeClz(firstCode ^ splitCode);

            if (splitPrefix > commonPrefix)
                split = newSplit;
        }
    }
    while (step > 1);

    return split;
}