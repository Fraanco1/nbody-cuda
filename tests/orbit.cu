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
//
// Output: CSV to stdout (step, t, x0, y0, z0, x1, y1, z1) for plotting.

#include <iostream>
#include <cmath>
#include "tree.cuh"
#include "forces.cuh"

int main() {
    const int   n     = 2;
    const float r     = 0.1f;
    const float G     = 1.0f;  // implicit, but stated for clarity
    const float m     = 1.0f;
    const float v     = 1.0f / (2.0f * std::sqrt(r));        // ≈ 1.581
    const float T     = 2.0f * 3.14159265f * r / v;          // ≈ 0.397
    const float dt    = 0.001f;                               // ~400 steps / orbit
    const int   steps = 4000;                                 // ~10 orbits
    const float theta = 0.5f;
    const float eps   = 1e-4f;                                // separation never small

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

    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;

    // ── Initial force computation (needed before the first kick) ────────
    Tree tree(n);
    tree.rebuild(d_pos, d_vel, d_mass);
    computeForces<<<blocks, threads>>>(tree.nodeData(), tree.arrays(),
                                       d_pos, d_acc, n, theta, eps);

    // ── CSV header on stderr (so it doesn't mix with data on stdout) ────
    std::cerr << "# Two-body orbit: r=" << r << ", v=" << v
              << ", period T=" << T << ", dt=" << dt
              << ", steps=" << steps << " (~" << steps*dt/T << " orbits)\n";
    // Outputs:
    //   x0..z1     — raw body positions (note: bodies may swap under Morton sort)
    //   cx,cy,cz   — center of mass (should be constant at (0.5, 0.5, 0.5))
    //   sep        — separation distance |r1 - r0| (should oscillate slightly
    //                around 2r = 0.2 for a stable circular orbit)
    std::cout << "step,t,x0,y0,z0,x1,y1,z1,cx,cy,cz,sep\n";

    // ── Main loop: KDK leapfrog ─────────────────────────────────────────
    // 1. half-kick:  v += a · dt/2
    // 2. drift:      x += v · dt
    // 3. recompute a at new x
    // 4. half-kick:  v += a · dt/2
    for (int step = 0; step < steps; step++) {
        halfKick  <<<blocks, threads>>>(d_vel, d_acc, dt, n);
        fullDrift <<<blocks, threads>>>(d_pos, d_vel, dt, n);

        tree.rebuild(d_pos, d_vel, d_mass);
        computeForces<<<blocks, threads>>>(tree.nodeData(), tree.arrays(),
                                           d_pos, d_acc, n, theta, eps);

        halfKick  <<<blocks, threads>>>(d_vel, d_acc, dt, n);

        // Output every 10 steps to keep CSV manageable
        if (step % 10 == 0) {
            Vec3 pos[2];
            cudaMemcpy(pos, d_pos, 2 * sizeof(Vec3), cudaMemcpyDeviceToHost);

            float cx = 0.5f * (pos[0].x + pos[1].x);
            float cy = 0.5f * (pos[0].y + pos[1].y);
            float cz = 0.5f * (pos[0].z + pos[1].z);
            float dx = pos[1].x - pos[0].x;
            float dy = pos[1].y - pos[0].y;
            float dz = pos[1].z - pos[0].z;
            float sep = std::sqrt(dx*dx + dy*dy + dz*dz);

            std::cout << step << "," << (step * dt) << ","
                      << pos[0].x << "," << pos[0].y << "," << pos[0].z << ","
                      << pos[1].x << "," << pos[1].y << "," << pos[1].z << ","
                      << cx << "," << cy << "," << cz << "," << sep << "\n";
        }
    }

    cudaFree(d_pos);
    cudaFree(d_vel);
    cudaFree(d_acc);
    cudaFree(d_mass);
    return 0;
}