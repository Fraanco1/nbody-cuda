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
    Vec3        camera_forward,   // unit vector pointing into the scene
    Vec3        camera_up,        // approximate up direction
    float       focal,            // focal length in pixels
    float       cx,               // principal point x (usually width/2)
    float       cy                // principal point y (usually height/2)
);
