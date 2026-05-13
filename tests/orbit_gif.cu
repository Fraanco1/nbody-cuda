// Two-body orbit, animated as a GIF.
//
// Each frame plots both bodies as small white squares on a dark canvas.
// A persistent dim-cyan trail accumulates the orbit history so you can
// see the path the bodies have swept out.
//
// Output: orbit.gif (a few seconds at ~33 fps).

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include "tree.cuh"
#include "forces.cuh"
#include "gif.h"

// ── Canvas helpers ────────────────────────────────────────────────────────
struct Canvas {
    int w, h;
    std::vector<uint8_t> rgba;   // size = w*h*4

    Canvas(int W, int H) : w(W), h(H), rgba(4 * W * H, 0) {
        // start opaque black
        for (int i = 0; i < w * h; i++) rgba[4*i + 3] = 255;
    }

    void setPixel(int px, int py, uint8_t r, uint8_t g, uint8_t b) {
        if (px < 0 || px >= w || py < 0 || py >= h) return;
        int idx = 4 * (py * w + px);
        rgba[idx + 0] = r;
        rgba[idx + 1] = g;
        rgba[idx + 2] = b;
        rgba[idx + 3] = 255;
    }

    // Plot a small filled square so the body is visible at any zoom.
    void plotBody(float x, float y, uint8_t r, uint8_t g, uint8_t b, int radius = 2) {
        int px = (int)(x * w);
        int py = (h - 1) - (int)(y * h);
        for (int dy = -radius; dy <= radius; dy++)
            for (int dx = -radius; dx <= radius; dx++)
                setPixel(px + dx, py + dy, r, g, b);
    }
};

int main() {
    const int   n     = 2;
    const float r     = 0.1f;
    const float m     = 1.0f;
    const float v     = 1.0f / (2.0f * std::sqrt(r));
    const float dt    = 0.001f;
    const int   steps = 4000;
    const float theta = 0.5f;
    const float eps   = 1e-4f;

    const int   imgSize       = 600;
    const int   stepsPerFrame = 10;     // 4000/10 = 400 frames
    const int   gifDelayCs    = 3;      // hundredths of a second per frame -> ~33 fps

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
    Vec3  *d_pos, *d_vel, *d_acc;
    float *d_mass;
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

    // ── Initial force computation ───────────────────────────────────────
    Tree tree(n);
    tree.rebuild(d_pos, d_vel, d_mass);
    computeForces<<<blocks, threads>>>(tree.nodeData(), tree.arrays(),
                                       d_pos, d_acc, n, theta, eps);

    // ── Trail canvas (persistent across frames) ─────────────────────────
    Canvas trail(imgSize, imgSize);

    // ── Open GIF ────────────────────────────────────────────────────────
    GifWriter gif;
    if (!GifBegin(&gif, "orbit.gif", imgSize, imgSize, gifDelayCs)) {
        std::cerr << "Failed to open orbit.gif for writing\n";
        return 1;
    }

    // ── Main loop ───────────────────────────────────────────────────────
    Vec3 hostPos[2];
    int frameCount = 0;
    for (int step = 0; step < steps; step++) {
        halfKick  <<<blocks, threads>>>(d_vel, d_acc, dt, n);
        fullDrift <<<blocks, threads>>>(d_pos, d_vel, dt, n);

        tree.rebuild(d_pos, d_vel, d_mass);
        computeForces<<<blocks, threads>>>(tree.nodeData(), tree.arrays(),
                                           d_pos, d_acc, n, theta, eps);

        halfKick  <<<blocks, threads>>>(d_vel, d_acc, dt, n);

        cudaMemcpy(hostPos, d_pos, n * sizeof(Vec3), cudaMemcpyDeviceToHost);

        // Mark the trail (one pixel per body per simulation step, dim cyan)
        for (int i = 0; i < n; i++) {
            int px = (int)(hostPos[i].x * imgSize);
            int py = (imgSize - 1) - (int)(hostPos[i].y * imgSize);
            trail.setPixel(px, py, 0, 120, 160);
        }

        // Every stepsPerFrame steps, draw a frame: trail + bright bodies
        if (step % stepsPerFrame == 0) {
            Canvas frame = trail;   // start from current trail
            for (int i = 0; i < n; i++)
                frame.plotBody(hostPos[i].x, hostPos[i].y, 255, 255, 255, 3);

            GifWriteFrame(&gif, frame.rgba.data(), imgSize, imgSize, gifDelayCs);
            frameCount++;
        }
    }

    GifEnd(&gif);
    std::cout << "Wrote orbit.gif (" << frameCount << " frames, "
              << imgSize << "x" << imgSize << ")\n";

    cudaFree(d_pos);
    cudaFree(d_vel);
    cudaFree(d_acc);
    cudaFree(d_mass);
    return 0;
}
