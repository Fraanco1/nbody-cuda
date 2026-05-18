#include "../include/graphic.cuh"
#include <math.h>

__device__ static inline Vec3 cross(Vec3 a, Vec3 b) {
    return { a.y*b.z - a.z*b.y,
             a.z*b.x - a.x*b.z,
             a.x*b.y - a.y*b.x };
}

__device__ static inline float dot(Vec3 a, Vec3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ static inline Vec3 normalize(Vec3 v) {
    float inv = rsqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    return { v.x*inv, v.y*inv, v.z*inv };
}

__global__ void accumulateMomenta(
    const Vec3  *velocities,
    const float *masses,
    float       *comVel,
    int          n)
{
    extern __shared__ float sdata[];
    float* sdX = sdata;
    float* sdY = sdata +     blockDim.x;
    float* sdZ = sdata + 2 * blockDim.x;
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    float m = (i < n) ? masses[i] : 0.f;
    sdX[tid] = (i < n) ? m * velocities[i].x : 0.f;
    sdY[tid] = (i < n) ? m * velocities[i].y : 0.f;
    sdZ[tid] = (i < n) ? m * velocities[i].z : 0.f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { sdX[tid] += sdX[tid+s]; sdY[tid] += sdY[tid+s]; sdZ[tid] += sdZ[tid+s]; }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(&comVel[0], sdX[0]);
        atomicAdd(&comVel[1], sdY[0]);
        atomicAdd(&comVel[2], sdZ[0]);
    }
}

__global__ void computeRelativeSpeeds(
    const Vec3  *velocities,
    const float *comVel,
    float       *speeds,
    int          n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float dvx = velocities[i].x - comVel[0];
    float dvy = velocities[i].y - comVel[1];
    float dvz = velocities[i].z - comVel[2];
    speeds[i] = sqrtf(dvx*dvx + dvy*dvy + dvz*dvz);
}

__global__ void calculateProjection(
    const Vec3 *positions,
    Vec2       *projections,
    int         n,
    Vec3        camera_pos,
    Vec3        camera_forward,
    Vec3        camera_up,
    float       focal,
    float       cx,
    float       cy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Build orthonormal camera basis
    Vec3 fwd   = normalize(camera_forward);
    Vec3 right = normalize(cross(fwd, camera_up));
    Vec3 up    = cross(right, fwd);   // reorthogonalized, already unit length

    // Vector from camera to the body
    Vec3 pos = positions[i];
    Vec3 d   = { pos.x - camera_pos.x,
                 pos.y - camera_pos.y,
                 pos.z - camera_pos.z };

    // Project onto camera axes
    float cam_x = dot(d, right);
    float cam_y = dot(d, up);
    float cam_z = dot(d, fwd);       // depth along optical axis

    if (cam_z <= 0.0f) {
        projections[i] = { -1.0f, -1.0f };
        return;
    }

    // Perspective division + shift to pixel coordinates
    projections[i] = { focal * cam_x / cam_z + cx,
                       focal * cam_y / cam_z + cy };
}
