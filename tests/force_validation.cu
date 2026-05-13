// Validates computeForces against a direct O(N^2) host-side reference.
// Small N (16) with random positions in [0,1]^3, all masses = 1.
// Comparison is component-wise, with relative + absolute tolerance.

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include "tree.cuh"
#include "forces.cuh"

// Direct O(N^2) gravitational force on body i, same softening as the kernel.
static Vec3 directForce(const Vec3* pos, const float* mass, int n,
                        int i, float eps)
{
    Vec3 a = {0.0f, 0.0f, 0.0f};
    const float eps2 = eps * eps;
    for (int j = 0; j < n; j++) {
        float dx = pos[j].x - pos[i].x;
        float dy = pos[j].y - pos[i].y;
        float dz = pos[j].z - pos[i].z;
        float d2 = dx*dx + dy*dy + dz*dz + eps2;
        float invDist = 1.0f / std::sqrt(d2);
        float invDist3 = invDist * invDist * invDist;
        float f = mass[j] * invDist3;
        a.x += f * dx;
        a.y += f * dy;
        a.z += f * dz;
    }
    return a;
}

int main() {
    const int   n     = 16;
    const float theta = 0.5f;
    const float eps   = 0.01f;

    // ── Host setup: random positions in [0,1]^3, uniform masses ─────────
    srand(42);
    Vec3*  h_pos  = (Vec3*)malloc(n * sizeof(Vec3));
    float* h_mass = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        h_pos[i].x = (float)rand() / RAND_MAX;
        h_pos[i].y = (float)rand() / RAND_MAX;
        h_pos[i].z = (float)rand() / RAND_MAX;
        h_mass[i] = 1.0f;
    }

    // ── Device buffers ──────────────────────────────────────────────────
    Vec3*  d_pos;
    Vec3*  d_acc;
    float* d_mass;
    cudaMalloc(&d_pos,  n * sizeof(Vec3));
    cudaMalloc(&d_acc,  n * sizeof(Vec3));
    cudaMalloc(&d_mass, n * sizeof(float));
    cudaMemcpy(d_pos,  h_pos,  n * sizeof(Vec3),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, h_mass, n * sizeof(float), cudaMemcpyHostToDevice);

    // ── Build tree and compute forces on device ─────────────────────────
    Tree tree(n);
    tree.rebuild(d_pos, d_mass);    // NOTE: permutes d_pos and d_mass

    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;
    computeForces<<<blocks, threads>>>(tree.nodeData(), tree.arrays(),
                                       d_pos, d_acc, n, theta, eps);

    // ── Copy sorted positions/masses and computed accelerations back ────
    // tree.rebuild reordered d_pos and d_mass; the reference must use the
    // same ordering, so we copy them back to host AFTER the rebuild.
    cudaMemcpy(h_pos,  d_pos,  n * sizeof(Vec3),  cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mass, d_mass, n * sizeof(float), cudaMemcpyDeviceToHost);

    Vec3* h_acc_gpu = (Vec3*)malloc(n * sizeof(Vec3));
    cudaMemcpy(h_acc_gpu, d_acc, n * sizeof(Vec3), cudaMemcpyDeviceToHost);

    // ── Reference: direct O(N^2) sum on host ────────────────────────────
    Vec3* h_acc_ref = (Vec3*)malloc(n * sizeof(Vec3));
    for (int i = 0; i < n; i++)
        h_acc_ref[i] = directForce(h_pos, h_mass, n, i, eps);

    // ── Compare ─────────────────────────────────────────────────────────
    std::cout << "=== Force kernel vs direct O(N^2) reference ===\n";
    std::cout << "N = " << n << ", theta = " << theta << ", eps = " << eps << "\n\n";

    float maxRelErr = 0.0f;
    float maxAbsErr = 0.0f;
    for (int i = 0; i < n; i++) {
        float dx = h_acc_gpu[i].x - h_acc_ref[i].x;
        float dy = h_acc_gpu[i].y - h_acc_ref[i].y;
        float dz = h_acc_gpu[i].z - h_acc_ref[i].z;
        float errMag = std::sqrt(dx*dx + dy*dy + dz*dz);
        float refMag = std::sqrt(h_acc_ref[i].x * h_acc_ref[i].x +
                                 h_acc_ref[i].y * h_acc_ref[i].y +
                                 h_acc_ref[i].z * h_acc_ref[i].z);
        float relErr = (refMag > 1e-6f) ? errMag / refMag : 0.0f;

        if (relErr > maxRelErr) maxRelErr = relErr;
        if (errMag > maxAbsErr) maxAbsErr = errMag;

        std::cout << "Body " << i
                  << " | GPU=(" << h_acc_gpu[i].x << ", " << h_acc_gpu[i].y
                          << ", " << h_acc_gpu[i].z << ")"
                  << " | ref=(" << h_acc_ref[i].x << ", " << h_acc_ref[i].y
                          << ", " << h_acc_ref[i].z << ")"
                  << " | relErr=" << relErr << "\n";
    }

    std::cout << "\n=== Summary ===\n"
              << "Max relative error: " << maxRelErr << "\n"
              << "Max absolute error: " << maxAbsErr << "\n";
    if (maxRelErr < 0.05f)
        std::cout << "PASS (within 5% relative)\n";
    else
        std::cout << "FAIL — relative error exceeds 5%\n";

    free(h_pos);
    free(h_mass);
    free(h_acc_gpu);
    free(h_acc_ref);
    cudaFree(d_pos);
    cudaFree(d_acc);
    cudaFree(d_mass);
    return 0;
}
