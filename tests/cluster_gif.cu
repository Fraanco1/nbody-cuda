// 1024-body gravitational cluster, animated as a GIF.
//
// Initial conditions: uniform random positions inside a sphere of radius R
// centered at (0.5, 0.5, 0.5), all with zero initial velocity. With nothing
// holding them apart, the cluster will collapse inward, compact near the
// center, and re-expand outward — a "cold collapse" scenario.
//
// Output: cluster.gif

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include "tree.cuh"
#include "forces.cuh"
#include "gif.h"

struct Canvas {
    int w, h;
    std::vector<uint8_t> rgba;

    Canvas(int W, int H) : w(W), h(H), rgba(4 * W * H, 0) {
        for (int i = 0; i < w * h; i++) rgba[4*i + 3] = 255;   // opaque black
    }

    void clear() {
        for (int i = 0; i < w * h; i++) {
            rgba[4*i + 0] = 0;
            rgba[4*i + 1] = 0;
            rgba[4*i + 2] = 0;
            rgba[4*i + 3] = 255;
        }
    }

    void setPixel(int px, int py, uint8_t r, uint8_t g, uint8_t b) {
        if (px < 0 || px >= w || py < 0 || py >= h) return;
        int idx = 4 * (py * w + px);
        rgba[idx + 0] = r;
        rgba[idx + 1] = g;
        rgba[idx + 2] = b;
        rgba[idx + 3] = 255;
    }

    // Small filled square at (x, y) ∈ [0,1]^2
    void plotBody(float x, float y, uint8_t r, uint8_t g, uint8_t b, int radius = 1) {
        int px = (int)(x * w);
        int py = (h - 1) - (int)(y * h);
        for (int dy = -radius; dy <= radius; dy++)
            for (int dx = -radius; dx <= radius; dx++)
                setPixel(px + dx, py + dy, r, g, b);
    }
};

// Sample a single point uniformly inside a unit-radius ball via rejection.
static void sampleInUnitBall(float& x, float& y, float& z) {
    do {
        x = 2.0f * (float)rand() / RAND_MAX - 1.0f;
        y = 2.0f * (float)rand() / RAND_MAX - 1.0f;
        z = 2.0f * (float)rand() / RAND_MAX - 1.0f;
    } while (x*x + y*y + z*z > 1.0f);
}

int main() {
    const int   n     = 64;
    const float R     = 0.15f;          // initial cluster radius
    const float cx    = 0.5f;
    const float cy    = 0.5f;
    const float cz    = 0.5f;
    const float m     = 1.0f / n;       // total mass = 1 (normalizes the scale)
    const float dt    = 0.001f;
    const int   steps = 2000;
    const float theta = 0.5f;
    const float eps   = 0.005f;         // ~ avg inter-body spacing

    const int   imgSize       = 600;
    const int   stepsPerFrame = 10;     // 200 frames
    const int   gifDelayCs    = 3;      // ~33 fps

    // ── Initial conditions on host ──────────────────────────────────────
    std::srand(42);
    std::vector<Vec3>  h_pos(n);
    std::vector<Vec3>  h_vel(n, {0.0f, 0.0f, 0.0f});
    std::vector<float> h_mass(n, m);
    for (int i = 0; i < n; i++) {
        float ux, uy, uz;
        sampleInUnitBall(ux, uy, uz);
        h_pos[i] = { cx + R * ux, cy + R * uy, cz + R * uz };
    }

    // ── Device buffers ──────────────────────────────────────────────────
    Vec3  *d_pos, *d_vel, *d_acc;
    float *d_mass;
    cudaMalloc(&d_pos,  n * sizeof(Vec3));
    cudaMalloc(&d_vel,  n * sizeof(Vec3));
    cudaMalloc(&d_acc,  n * sizeof(Vec3));
    cudaMalloc(&d_mass, n * sizeof(float));
    cudaMemcpy(d_pos,  h_pos.data(),  n * sizeof(Vec3),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel,  h_vel.data(),  n * sizeof(Vec3),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, h_mass.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_acc, 0, n * sizeof(Vec3));

    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;

    // ── Initial force computation ───────────────────────────────────────
    Tree tree(n);
    tree.rebuild(d_pos, d_vel, d_mass);
    computeForces<<<blocks, threads>>>(tree.nodeData(), tree.arrays(),
                                       d_pos, d_acc, n, theta, eps);

    Canvas frame(imgSize, imgSize);

    GifWriter gif;
    if (!GifBegin(&gif, "cluster.gif", imgSize, imgSize, gifDelayCs)) {
        std::cerr << "Failed to open cluster.gif for writing\n";
        return 1;
    }

    std::vector<Vec3> hostPos(n);
    int frameCount = 0;

    for (int step = 0; step < steps; step++) {
        halfKick  <<<blocks, threads>>>(d_vel, d_acc, dt, n);
        fullDrift <<<blocks, threads>>>(d_pos, d_vel, dt, n);

        tree.rebuild(d_pos, d_vel, d_mass);
        computeForces<<<blocks, threads>>>(tree.nodeData(), tree.arrays(),
                                           d_pos, d_acc, n, theta, eps);

        halfKick  <<<blocks, threads>>>(d_vel, d_acc, dt, n);

        if (step % stepsPerFrame == 0) {
            cudaMemcpy(hostPos.data(), d_pos, n * sizeof(Vec3),
                       cudaMemcpyDeviceToHost);

            frame.clear();
            for (int i = 0; i < n; i++)
                frame.plotBody(hostPos[i].x, hostPos[i].y, 255, 255, 255, 1);

            GifWriteFrame(&gif, frame.rgba.data(), imgSize, imgSize, gifDelayCs);
            frameCount++;
        }
    }

    GifEnd(&gif);
    std::cout << "Wrote cluster.gif (" << frameCount << " frames, "
              << imgSize << "x" << imgSize << ")\n";

    cudaFree(d_pos);
    cudaFree(d_vel);
    cudaFree(d_acc);
    cudaFree(d_mass);
    return 0;
}
