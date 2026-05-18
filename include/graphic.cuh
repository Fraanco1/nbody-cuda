#pragma once
#include "morton.cuh"

// Projects n world-space positions onto a 2D image plane using a pinhole
// camera model.
//
// Camera basis is built from forward and up: right = cross(forward, up),
// then up is reorthogonalized so the basis is orthonormal.
//
// Perspective division: proj = f * cam_xy / cam_z
// Pixel coords:         out  = proj + principal_point
//
// Points at or behind the camera (cam_z <= 0) are written as {-1, -1}.
__global__ void calculateProjection(
    const Vec3 *positions,
    Vec2       *projections,
    int         n,
    Vec3        camera_pos,
    Vec3        camera_forward,
    Vec3        camera_up,
    float       focal,
    float       cx,
    float       cy);

// Reduces sum(mass_i * velocity_i) into comVel[0..2] via atomicAdd.
// comVel must be zeroed before the call.
// Call with shared memory = 3 * blockDim.x * sizeof(float).
__global__ void accumulateMomenta(
    const Vec3  *velocities,
    const float *masses,
    float       *comVel,
    int          n);

// Writes |v_i - v_com| for each body into speeds[].
// comVel holds the (already divided by total mass) CoM velocity.
__global__ void computeRelativeSpeeds(
    const Vec3  *velocities,
    const float *comVel,
    float       *speeds,
    int          n);
