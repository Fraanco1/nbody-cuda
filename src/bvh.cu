#include "bvh.cuh"
#include "morton.cuh"

// Step 1 — initialize leaves from body positions
__global__ void initLeaves(
    NodeData*    nodeData,
    Vec3*        positions,
    float*       masses,
    int          n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // leaf i lives at index (n - 1 + i) in the flat array
    int idx = n - 1 + i;

    nodeData[idx].minX = positions[i].x;
    nodeData[idx].minY = positions[i].y;
    nodeData[idx].minZ = positions[i].z;
    nodeData[idx].maxX = positions[i].x;
    nodeData[idx].maxY = positions[i].y;
    nodeData[idx].maxZ = positions[i].z;

    nodeData[idx].comX = positions[i].x;
    nodeData[idx].comY = positions[i].y;
    nodeData[idx].comZ = positions[i].z;
    nodeData[idx].mass = masses[i];
}

// Step 2 — bottom-up pass, one thread per leaf
__global__ void computeAABBandCoM(
    NodeData*    nodeData,
    BVH          bvh,
    int*         flags,     // atomic flags, one per internal node, init to 0
    int          n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // start at this leaf's parent
    int nodeIdx = bvh.leafParent[i];

    while (nodeIdx != -1) {

        // first thread to arrive exits
        // second thread to arrive proceeds
        int arrived = atomicAdd(&flags[nodeIdx], 1);
        if (arrived == 0)
            break;

        // both children are ready — read their data
        int   leftIdx  = bvh.left[nodeIdx];
        int   rightIdx = bvh.right[nodeIdx];

        NodeData left  = nodeData[leftIdx];
        NodeData right = nodeData[rightIdx];

        // merge AABBs
        NodeData node;
        node.minX = fminf(left.minX, right.minX);
        node.minY = fminf(left.minY, right.minY);
        node.minZ = fminf(left.minZ, right.minZ);
        node.maxX = fmaxf(left.maxX, right.maxX);
        node.maxY = fmaxf(left.maxY, right.maxY);
        node.maxZ = fmaxf(left.maxZ, right.maxZ);

        // merge centers of mass
        float totalMass = left.mass + right.mass;
        node.comX = (left.mass * left.comX + right.mass * right.comX) / totalMass;
        node.comY = (left.mass * left.comY + right.mass * right.comY) / totalMass;
        node.comZ = (left.mass * left.comZ + right.mass * right.comZ) / totalMass;
        node.mass = totalMass;

        // write result
        nodeData[nodeIdx] = node;

        // walk up
        nodeIdx = bvh.parent[nodeIdx];
    }
}