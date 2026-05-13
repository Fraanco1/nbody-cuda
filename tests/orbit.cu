// Two-body circular orbit, used to validate the leapfrog integrator.
//
// Setup: two unit-mass bodies separated by 2r along the x-axis, with
// velocities chosen so that each orbits the common center of mass at
// radius r with period T. With G = 1 and m = 1:
//     v = 1 / (2 * sqrt(r))
//     T = 2*pi*r / v
//
// Expected behavior: bodies move in a closed circle around (0.5, 0.5, 0.5).
// If the orbit spirals or drifts, the integrator is broken.

#include <iostream>
#include <cmath>
#include "tree.cuh"
#include "forces.cuh"

int main() {
    const int   n     = 2;
    const float r     = 0.1f;
    const float m     = 1.0f;
    const float v     = 1.0f / (2.0f * std::sqrt(r));        // ≈ 1.581
    const float T     = 2.0f * 3.14159265f * r / v;          // ≈ 0.397
    const float dt    = 0.001f;
    const int   steps = 5;                                    // diagnostic run
    const float theta = 0.5f;
    const float eps   = 1e-4f;

    // ── Initial conditions ──────────────────────────────────────────────
    Vec3 h_pos[2] = {
        {0.5f + r, 0.5f, 0.5f},
        {0.5f - r, 0.5f, 0.5f}
    };
    Vec3 h_vel[2] = {
        {0.0f,  v, 0.0f},
        {0.0f, -v, 0.0f}
    };
    float h_mass[2] = {m, m};

    // ── Device buffers ──────────────────────────────────────────────────
    Vec3*  d_pos;
    Vec3*  d_vel;
    Vec3*  d_acc;
    float* d_mass;
    cudaMalloc(&d_pos,  n * sizeof(Vec3));
    cudaMalloc(&d_vel,  n * sizeof(Vec3));
    cudaMalloc(&d_acc,  n * sizeof(Vec3));
    cudaMalloc(&d_mass, n * sizeof(float));
    cudaMemcpy(d_pos,  h_pos,  n * sizeof(Vec3),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel,  h_vel,  n * sizeof(Vec3),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, h_mass, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_acc, 0, n * sizeof(Vec3));

    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;

    // ── Initial force computation (before any kick) ─────────────────────
    Tree tree(n);
    tree.rebuild(d_pos, d_vel, d_mass);
    computeForces<<<blocks, threads>>>(tree.nodeData(), tree.arrays(),
                                       d_pos, d_acc, n, theta, eps);

    std::cerr << "# Two-body orbit: r=" << r << ", v=" << v
              << ", expected period T=" << T << ", dt=" << dt
              << ", steps=" << steps << "\n";
    std::cerr << "# Expected initial accel on body at x=0.6: a = (-25, 0, 0)\n";
    std::cerr << "# Expected initial accel on body at x=0.4: a = (+25, 0, 0)\n\n";

    // ── Dump state BEFORE the first loop iteration ──────────────────────
    {
        Vec3 pos[2], vel[2], acc[2];
        cudaMemcpy(pos, d_pos, 2 * sizeof(Vec3), cudaMemcpyDeviceToHost);
        cudaMemcpy(vel, d_vel, 2 * sizeof(Vec3), cudaMemcpyDeviceToHost);
        cudaMemcpy(acc, d_acc, 2 * sizeof(Vec3), cudaMemcpyDeviceToHost);
        std::cerr << "initial (after rebuild + first forces, before loop):\n"
                  << "  pos0=(" << pos[0].x << "," << pos[0].y << "," << pos[0].z << ")"
                  <<   " pos1=(" << pos[1].x << "," << pos[1].y << "," << pos[1].z << ")\n"
                  << "  vel0=(" << vel[0].x << "," << vel[0].y << "," << vel[0].z << ")"
                  <<   " vel1=(" << vel[1].x << "," << vel[1].y << "," << vel[1].z << ")\n"
                  << "  acc0=(" << acc[0].x << "," << acc[0].y << "," << acc[0].z << ")"
                  <<   " acc1=(" << acc[1].x << "," << acc[1].y << "," << acc[1].z << ")\n\n";
    }

    // ── Main loop: KDK leapfrog ─────────────────────────────────────────
    for (int step = 0; step < steps; step++) {
        halfKick  <<<blocks, threads>>>(d_vel, d_acc, dt, n);
        fullDrift <<<blocks, threads>>>(d_pos, d_vel, dt, n);

        tree.rebuild(d_pos, d_vel, d_mass);
        computeForces<<<blocks, threads>>>(tree.nodeData(), tree.arrays(),
                                           d_pos, d_acc, n, theta, eps);

        halfKick  <<<blocks, threads>>>(d_vel, d_acc, dt, n);

        Vec3 pos[2], vel[2], acc[2];
        cudaMemcpy(pos, d_pos, 2 * sizeof(Vec3), cudaMemcpyDeviceToHost);
        cudaMemcpy(vel, d_vel, 2 * sizeof(Vec3), cudaMemcpyDeviceToHost);
        cudaMemcpy(acc, d_acc, 2 * sizeof(Vec3), cudaMemcpyDeviceToHost);

        std::cerr << "step " << step << "\n"
                  << "  pos0=(" << pos[0].x << "," << pos[0].y << "," << pos[0].z << ")"
                  <<   " pos1=(" << pos[1].x << "," << pos[1].y << "," << pos[1].z << ")\n"
                  << "  vel0=(" << vel[0].x << "," << vel[0].y << "," << vel[0].z << ")"
                  <<   " vel1=(" << vel[1].x << "," << vel[1].y << "," << vel[1].z << ")\n"
                  << "  acc0=(" << acc[0].x << "," << acc[0].y << "," << acc[0].z << ")"
                  <<   " acc1=(" << acc[1].x << "," << acc[1].y << "," << acc[1].z << ")\n\n";
    }

    cudaFree(d_pos);
    cudaFree(d_vel);
    cudaFree(d_acc);
    cudaFree(d_mass);
    return 0;
}