// Given an array of Morton codes, constructs an ordered binary radix tree
// https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf
#pragma once
#include <cstdint>

__device__ __forceinline__ int safeClz(unsigned int x) {
    return x == 0 ? 32 : __clz(x);
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