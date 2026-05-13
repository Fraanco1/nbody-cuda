#pragma once

// Simple 3D vector used everywhere for positions, velocities, accelerations.
struct Vec3 {
    float x, y, z;
};

// Expand a 10-bit integer into 30 bits by inserting two zero bits after each.
__device__ unsigned int expandBits(unsigned int v);

// Compute a 30-bit Morton code for each point in [0,1]^3.
__global__ void morton3D(Vec3* points, unsigned int* mortonCodes, int n);
