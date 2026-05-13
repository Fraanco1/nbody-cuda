#include <math.h>
#include "bvh.cuh"

__global__ void computeForces(NodeData *nodeData,
                              BVHArrays bvh,
                              Vec3 *positions,
                              Vec3 *accelerations,
                              int n,
                              float theta,
                              float eps)
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
            float f = node.mass * invDist3;
            ax += f * dx;
            ay += f * dy;
            az = f * dz;
        }
        else {
            stack[sp++] = bvh.left[nodeIdx];
            stack[sp++] = bvh.right[nodeIdx];
        }
    }

    accelerations[i].x = ax;
    accelerations[i].y = ay;
    accelerations[i].z = az;
} 