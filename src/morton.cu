// Generates a Morton code for a given 3D point.
// https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.

#include <iostream>
using namespace std;

struct Vec3 {
    float x, y, z;
};

__device__ unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
};

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__global__ void morton3D(Vec3 *point, unsigned int *mortonCode)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = point[i].x;
    float y = point[i].y;
    float z = point[i].z;
    x = min(max(x * 1024.0f, 0.0f), 1023.0f);
    y = min(max(y * 1024.0f, 0.0f), 1023.0f);
    z = min(max(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);
    mortonCode[i] = xx * 4 + yy * 2 + zz;
}

int main() {
    int n = 1024;
    size_t vecSize = n * sizeof(Vec3);
    size_t nortonSize = n * sizeof(unsigned int);

    // Allocate host memory
    Vec3 *h_points = (Vec3*)malloc(vecSize);
    unsigned int *h_mortons = (unsigned int*)malloc(vecSize);

    // Puts some values into points
    for(int i = 0; i < n; i++) {
        h_points[i].x = (float)i/n;
        h_points[i].y = (float)i/n;
        h_points[i].z = (float)i/n;
    } 

    // Allocate device memory
    Vec3 *d_points;
    unsigned int *d_mortons;

    cudaMalloc(&d_points, vecSize);
    cudaMalloc(&d_mortons, nortonSize)

    // Copy host to device
    cudaMemcpy(d_points, h_points, vecSize, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1)/threadsPerBlock;
    norton3D<<<blocks, threadsPerBlock>>>(d_points, d_nortons);

    // Copy device to host
    cudaMemcpy(h_nortons, d_nortons, nortonSize, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_points);
    cudaFree(d_nortons);
    free(h_points);
    free(h_nortons;)

    return 0;
}