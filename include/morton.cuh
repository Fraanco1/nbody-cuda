#pragma once

// Simple 3D vector used everywhere for positions, velocities, accelerations.
struct Vec3 {
    float x, y, z;
};

struct Vec2 {
    float x, y;
};

// Expand a 10-bit integer into 30 bits by inserting two zero bits after each.
__device__ unsigned int expandBits(unsigned int v);

// Compute a 30-bit Morton code for each point.
// Positions are normalized on-the-fly using the provided bounding box:
//   minX/minY/minZ  — world-space minimum per axis
//   invExtent       — 1 / max(rangeX, rangeY, rangeZ)
// This allows positions to live anywhere; only the sorting order matters.
__global__ void morton3D(Vec3* points, unsigned int* mortonCodes, int n,
                         float minX, float minY, float minZ, float invExtent);
