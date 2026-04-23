#pragma once

// Vec3 declaration
struct Vec3 {
    float x, y, z;
};

// Kernel declaration
__device__ unsigned int expandBits(unsigned int v);

__global__ void morton3D(Vec3 *point, unsigned int *mortonCode);