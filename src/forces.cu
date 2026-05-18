#include <math.h>
#include "tree.cuh"

__global__ void computeForces(NodeData *nodeData,
                              BVHArrays bvh,
                              Vec3 *positions,
                              Vec3 *accelerations,
                              int n,
                              float theta,
                              float eps,
                              float G,
                              float haloMass,
                              float haloScale,
                              Vec3  haloCen)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    Vec3 myPos = positions[i];
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    int stack[64];
    int sp = 0;
    stack[sp++] = 0;

    while(sp > 0) {
        int nodeIdx = stack[--sp];
        NodeData node = nodeData[nodeIdx];

        float dx = node.comX - myPos.x;
        float dy = node.comY - myPos.y;
        float dz = node.comZ - myPos.z;
        float d2 = dx*dx + dy*dy + dz*dz;

        float sx = node.maxX - node.minX;
        float sy = node.maxY - node.minY;
        float sz = node.maxZ - node.minZ;

        float s = fmaxf(sx, fmaxf(sy, sz));
        bool isLeaf = (nodeIdx >= n - 1);

        if(isLeaf || s*s < theta*theta * d2) {
            float invDist = rsqrtf(d2 + eps*eps);
            float invDist3 = invDist * invDist * invDist;
            float f = G * node.mass * invDist3;
            ax += f * dx;
            ay += f * dy;
            az += f * dz;
        }
        else {
            stack[sp++] = bvh.left[nodeIdx];
            stack[sp++] = bvh.right[nodeIdx];
        }
    }

    // Static Hernquist dark-matter halo: F_r = -G*M_h / (r*(r+a)^2)
    if (haloMass > 0.0f) {
        float hx = myPos.x - haloCen.x;
        float hy = myPos.y - haloCen.y;
        float hz = myPos.z - haloCen.z;
        float hr = sqrtf(hx*hx + hy*hy + hz*hz) + 1e-6f;
        float hf = -G * haloMass / (hr * (hr + haloScale) * (hr + haloScale));
        ax += hf * hx;
        ay += hf * hy;
        az += hf * hz;
    }

    accelerations[i].x = ax;
    accelerations[i].y = ay;
    accelerations[i].z = az;
} 

// Shared-memory reduction + atomicAdd to accumulate the sum of all accelerations.
// Call with shared memory size = 3 * blockDim.x * sizeof(float).
__global__ void accumulateAcc(Vec3* acc, float* sum, int n) {
    extern __shared__ float sdata[];
    float* sdX = sdata;
    float* sdY = sdata +     blockDim.x;
    float* sdZ = sdata + 2 * blockDim.x;
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    sdX[tid] = (i < n) ? acc[i].x : 0.f;
    sdY[tid] = (i < n) ? acc[i].y : 0.f;
    sdZ[tid] = (i < n) ? acc[i].z : 0.f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { sdX[tid] += sdX[tid+s]; sdY[tid] += sdY[tid+s]; sdZ[tid] += sdZ[tid+s]; }
        __syncthreads();
    }
    if (tid == 0) { atomicAdd(&sum[0], sdX[0]); atomicAdd(&sum[1], sdY[0]); atomicAdd(&sum[2], sdZ[0]); }
}

__global__ void subtractMeanAcc(Vec3* acc, float* sum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    acc[i].x -= sum[0] / n;
    acc[i].y -= sum[1] / n;
    acc[i].z -= sum[2] / n;
}

__global__ void halfKick(Vec3 *velocities, Vec3 *accelerations, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    velocities[i].x += accelerations[i].x * dt * 0.5f;
    velocities[i].y += accelerations[i].y * dt * 0.5f;
    velocities[i].z += accelerations[i].z * dt * 0.5f;
}

__global__ void fullDrift(Vec3 *positions, Vec3 *velocities, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    positions[i].x += velocities[i].x * dt;
    positions[i].y += velocities[i].y * dt;
    positions[i].z += velocities[i].z * dt;
}