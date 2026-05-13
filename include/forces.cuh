#pragma once
#include "tree.cuh"

__global__ void computeForces(NodeData* nodeData,
                              BVHArrays bvh,
                              Vec3*     positions,
                              Vec3*     accelerations,
                              int       n,
                              float     theta,
                              float     eps);