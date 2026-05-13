#include <iostream>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include "tree.cuh"
#include "forces.cuh"

int main() {
    const int n = 1024;

    // ── Host data: bodies arranged on the diagonal of [0,1]^3 ────────────
    Vec3* h_points = (Vec3*)malloc(n * sizeof(Vec3));
    for (int i = 0; i < n; i++) {
        h_points[i].x = (float)i / n;
        h_points[i].y = (float)i / n;
        h_points[i].z = (float)i / n;
    }

    // ── Device data ──────────────────────────────────────────────────────
    Vec3*  d_positions;
    float* d_masses;
    Vec3 *d_accelerations;
    cudaMalloc(&d_positions, n * sizeof(Vec3));
    cudaMalloc(&d_masses,    n * sizeof(float));
    cudaMalloc(&d_accelerations, n * sizeof(Vec3));
    cudaMemcpy(d_positions, h_points, n * sizeof(Vec3), cudaMemcpyHostToDevice);
    thrust::fill(thrust::device_ptr<float>(d_masses),
                 thrust::device_ptr<float>(d_masses) + n, 1.0f);

    // ── Build the tree ───────────────────────────────────────────────────
    Tree tree(n);
    tree.rebuild(d_positions, d_masses);
    cudaDeviceSynchronize();

    // Acceleration calculation 
    const int threads = 256;
    const int blocks  = (tree.n() + threads - 1) / threads;

    computeForces<<<blocks, threads>>>(tree.nodeData(), tree.arrays(),
                                        d_positions, d_accelerations, n);
    cudaDeviceSynchronize();

    // ── Verify a few accelerations ─────────────────────────────────────────
    Vec3 h_accelerations[5];
    cudaMemcpy(h_accelerations, d_accelerations, 5 * sizeof(Vec3),
               cudaMemcpyDeviceToHost);
    std::cout << "\n=== Accelerations for first 5 bodies ===\n";
    for (int i = 0; i < 5; i++) {
        std::cout << "Body " << i
                  << " | ax=" << h_accelerations[i].x
                  << " | ay=" << h_accelerations[i].y  
                  << " | az=" << h_accelerations[i].z << "\n";
    }

    // ── Verify root ──────────────────────────────────────────────────────
    NodeData h_root;
    cudaMemcpy(&h_root, &tree.nodeData()[0], sizeof(NodeData),
               cudaMemcpyDeviceToHost);

    std::cout << "=== Root node ===\n"
              << "AABB x: [" << h_root.minX << ", " << h_root.maxX << "]\n"
              << "AABB y: [" << h_root.minY << ", " << h_root.maxY << "]\n"
              << "AABB z: [" << h_root.minZ << ", " << h_root.maxZ << "]\n"
              << "CoM:    (" << h_root.comX << ", " << h_root.comY
                      << ", " << h_root.comZ << ")\n"
              << "Mass:   " << h_root.mass << "\n";

    // ── Spot-check a few leaves ──────────────────────────────────────────
    std::cout << "\n=== First 5 leaves ===\n";
    for (int i = 0; i < 5; i++) {
        NodeData leaf;
        cudaMemcpy(&leaf, &tree.nodeData()[n - 1 + i], sizeof(NodeData),
                   cudaMemcpyDeviceToHost);
        std::cout << "Leaf " << i
                  << " | pos=(" << leaf.comX << ", " << leaf.comY
                          << ", " << leaf.comZ << ")"
                  << " | mass=" << leaf.mass << "\n";
    }

    // ── Spot-check a few internal nodes ──────────────────────────────────
    std::cout << "\n=== First 5 internal nodes ===\n";
    for (int i = 0; i < 5; i++) {
        NodeData node;
        cudaMemcpy(&node, &tree.nodeData()[i], sizeof(NodeData),
                   cudaMemcpyDeviceToHost);
        std::cout << "Node " << i
                  << " | AABB x:[" << node.minX << ", " << node.maxX << "]"
                  << " | CoM=("    << node.comX << ", " << node.comY
                          << ", " << node.comZ << ")"
                  << " | mass="    << node.mass << "\n";
    }

    // ── Cleanup ──────────────────────────────────────────────────────────
    cudaFree(d_positions);
    cudaFree(d_masses);
    cudaFree(d_accelerations);
    free(h_points);
    // tree's destructor frees its own buffers
    return 0;
}
