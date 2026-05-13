#include "tree.cuh"

// Step 1 — initialize one NodeData per leaf from the (sorted) body arrays.
// Leaf i in the tree corresponds to body i in sorted order, stored flat
// at array index (n - 1 + i).
__global__ void initLeaves(
    NodeData* nodeData,
    Vec3*     positions,
    float*    masses,
    int       n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

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

// Step 2 — bottom-up pass, one thread per leaf.
// Each thread walks up the tree; at every internal node it uses an atomic
// flag so that the *second* thread to arrive (when both children are ready)
// is the one that merges them. The first thread exits.
__global__ void computeAABBandCoM(
    NodeData* nodeData,
    BVHArrays bvh,
    int*      flags,
    int       n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int nodeIdx = bvh.leafParent[i];

    while (nodeIdx != -1) {
        int arrived = atomicAdd(&flags[nodeIdx], 1);
        if (arrived == 0) break;   // we're first; the second thread will do the work

        // Both children's NodeData is now ready — merge them.
        int leftIdx  = bvh.left[nodeIdx];
        int rightIdx = bvh.right[nodeIdx];

        NodeData left  = nodeData[leftIdx];
        NodeData right = nodeData[rightIdx];

        NodeData node;
        // AABB union
        node.minX = fminf(left.minX, right.minX);
        node.minY = fminf(left.minY, right.minY);
        node.minZ = fminf(left.minZ, right.minZ);
        node.maxX = fmaxf(left.maxX, right.maxX);
        node.maxY = fmaxf(left.maxY, right.maxY);
        node.maxZ = fmaxf(left.maxZ, right.maxZ);

        // Mass-weighted center of mass
        float totalMass = left.mass + right.mass;
        node.comX = (left.mass * left.comX + right.mass * right.comX) / totalMass;
        node.comY = (left.mass * left.comY + right.mass * right.comY) / totalMass;
        node.comZ = (left.mass * left.comZ + right.mass * right.comZ) / totalMass;
        node.mass = totalMass;

        nodeData[nodeIdx] = node;

        nodeIdx = bvh.parent[nodeIdx];
    }
}
