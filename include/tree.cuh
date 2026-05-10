#pragma once

__device__ int findSplit(
    unsigned int* sortedMortonCodes,
    int           first,
    int           last);