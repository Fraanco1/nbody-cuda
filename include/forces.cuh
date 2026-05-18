#pragma once
#include "tree.cuh"

__global__ void computeForces(NodeData* nodeData,
                              BVHArrays bvh,
                              Vec3*     positions,
                              Vec3*     accelerations,
                              int       n,
                              float     theta,
                              float     eps,
                              float     G);

__global__ void accumulateAcc(Vec3* acc, float* sum, int n);
__global__ void subtractMeanAcc(Vec3* acc, float* sum, int n);

__global__ void halfKick(Vec3 *velocities, Vec3 *accelerations, float dt, int n);

__global__ void fullDrift(Vec3 *positions, Vec3 *velocities, float dt, int n);