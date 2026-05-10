// Given an array of Morton codes, constructs an ordered binary radix tree
// https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf

#pragma once
#include <cstdint>

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
    int commonPrefix = __clz(firstCode ^ lastCode);

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one.
    int split = first; // initial guess
    int step  = last - first;

    do
    {
        step = (step + 1) >> 1;      // exponential decrease
        int newSplit = split + step; // proposed new position

        if (newSplit < last)
        {
            unsigned int splitCode   = sortedMortonCodes[newSplit];
            int          splitPrefix = __clz(firstCode ^ splitCode);
            if (splitPrefix > commonPrefix)
                split = newSplit;    // accept proposal
        }
    }
    while (step > 1);

    return split;
}