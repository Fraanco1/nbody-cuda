#include <iostream>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include "tree.cuh"
#include "forces.cuh"

int main() {
    const int n = 1024;

    // ── Host data: bodies arranged on the diagonal of [0,1]^3 ────────────
    Vec3* h_points = (Vec3*)malloc(n * sizeof(Vec3));
    Vec3 *h_velocities = (Vec3*)malloc(n * sizeof(Vec3));
    for (int i = 0; i < n; i++) {
        h_points[i].x = (float)i / n;
        h_points[i].y = (float)i / n;
        h_points[i].z = (float)i / n;
        h_velocities[i].x = 0.0f;
        h_velocities[i].y = 0.0f;
        h_velocities[i].z = 0.0f;
    }

    // ── Device data ──────────────────────────────────────────────────────
    Vec3*  d_positions;
    Vec3*  d_velocities;
    float* d_masses;
    Vec3 *d_accelerations;
    cudaMalloc(&d_positions, n * sizeof(Vec3));
    cudaMalloc(&d_velocities, n * sizeof(Vec3));
    cudaMalloc(&d_masses,    n * sizeof(float));
    cudaMalloc(&d_accelerations, n * sizeof(Vec3));
    cudaMemcpy(d_positions, h_points, n * sizeof(Vec3), cudaMemcpyHostToDevice);
    thrust::fill(thrust::device_ptr<float>(d_masses),
                 thrust::device_ptr<float>(d_masses) + n, 1.0f);

    
    // ── Build the tree ───────────────────────────────────────────────────
    Tree tree(n);
    int steps = 100;
    for(int i = 0; i < steps; i++) {
        tree.rebuild(d_positions, d_masses);
        cudaDeviceSynchronize();

        // Acceleration calculation 
        const int threads = 256;
        const int blocks  = (tree.n() + threads - 1) / threads;

        computeForces<<<blocks, threads>>>(tree.nodeData(), tree.arrays(),
                                            d_positions, d_accelerations, n, 0.5f, 0.001f);
        cudaDeviceSynchronize();

        // Integrate 
        halfKick<<<blocks, threads>>>(d_velocities, d_accelerations, 0.01f, n);
        cudaDeviceSynchronize();
        fullDrift<<<blocks, threads>>>(d_positions, d_velocities, 0.01f, n);
        cudaDeviceSynchronize();
        halfKick<<<blocks, threads>>>(d_velocities, d_accelerations, 0.01f, n);
        cudaDeviceSynchronize();

        // After a step, copy to the host the position of the first body and print it
        Vec3 h_firstPos;
        cudaMemcpy(&h_firstPos, d_positions, sizeof(Vec3), cudaMemcpyDeviceToHost);
        std::cout << "Step " << i << " | First body position: ("
                  << h_firstPos.x << ", "
                  << h_firstPos.y << ", "
                  << h_firstPos.z << ")\n";
    }

    // ── Cleanup ──────────────────────────────────────────────────────────
    cudaFree(d_positions);
    cudaFree(d_masses);
    cudaFree(d_accelerations);
    free(h_points);
    // tree's destructor frees its own buffers
    return 0;
}
