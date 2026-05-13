__global__ void computeForces(NodeData *nodeData,
                              BVHNode bvh,
                              Vec3 *positions,
                              Vec3 *accelerations,
                              int n,
                              float theta,
                              float eps)
{
    int i = blockIdx.x * blockDIm.x + threadIdx.x;
    if(i >= n) return;

    Vec3 myPos = positions[i];
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
}