// Morton-code computation, per the NVIDIA blog series:
// https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/

#include "morton.cuh"

__device__ unsigned int expandBits(unsigned int v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__global__ void morton3D(Vec3* points, unsigned int* mortonCodes, int n,
                         float minX, float minY, float minZ, float invExtent) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x = fminf(fmaxf((points[i].x - minX) * invExtent * 1024.0f, 0.0f), 1023.0f);
    float y = fminf(fmaxf((points[i].y - minY) * invExtent * 1024.0f, 0.0f), 1023.0f);
    float z = fminf(fmaxf((points[i].z - minZ) * invExtent * 1024.0f, 0.0f), 1023.0f);

    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);

    mortonCodes[i] = xx * 4 + yy * 2 + zz;
}
