// 1024-body gravitational cluster rendered with a pinhole camera via OpenCV.
//
// Cold collapse: uniform random positions inside a sphere of radius R
// centered at (0.5, 0.5, 0.5), zero initial velocity. The cluster collapses,
// bounces, and re-expands.
//
// Output: cluster.mp4

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "tree.cuh"
#include "forces.cuh"
#include "graphic.cuh"

static void sampleInUnitBall(float& x, float& y, float& z) {
    do {
        x = 2.0f * (float)rand() / RAND_MAX - 1.0f;
        y = 2.0f * (float)rand() / RAND_MAX - 1.0f;
        z = 2.0f * (float)rand() / RAND_MAX - 1.0f;
    } while (x*x + y*y + z*z > 1.0f);
}

int main(int argc, char* argv[]) {
    // ── Simulation parameters ───────────────────────────────────────────
    int   n             = 10000;
    float R             = 0.3f;
    float dt            = 0.001f;
    int   steps         = 2000;
    float theta         = 0.5f;
    float eps           = 0.05f;
    int   stepsPerFrame = 5;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (i + 1 >= argc) { std::cerr << "Missing value for " << arg << "\n"; return 1; }
        if      (arg == "--n")             n             = std::atoi(argv[++i]);
        else if (arg == "--R")             R             = std::atof(argv[++i]);
        else if (arg == "--dt")            dt            = std::atof(argv[++i]);
        else if (arg == "--steps")         steps         = std::atoi(argv[++i]);
        else if (arg == "--theta")         theta         = std::atof(argv[++i]);
        else if (arg == "--eps")           eps           = std::atof(argv[++i]);
        else if (arg == "--stepsPerFrame") stepsPerFrame = std::atoi(argv[++i]);
        else { std::cerr << "Unknown argument: " << arg << "\n"; return 1; }
    }

    const float m = 1.0f / n;

    // ── Camera / image parameters ───────────────────────────────────────
    const int   W      = 1024;
    const int   H      = 1024;
    const float focal  = 600.0f;       // pixels; trades FOV vs. zoom
    Vec3 cam_pos     = { 0.5f,  0.5f, 2.5f };   // above and behind the cluster
    Vec3 cam_forward = { 0.0f,  0.0f, -1.0f };  // looking toward -Z
    Vec3 cam_up      = { 0.0f,  1.0f,  0.0f };
    const float cx   = W / 2.0f;
    const float cy   = H / 2.0f;

    // ── Initial conditions on host ──────────────────────────────────────
    std::srand(42);
    std::vector<Vec3>  h_pos(n);
    std::vector<Vec3>  h_vel(n, {0.0f, 0.0f, 0.0f});
    std::vector<float> h_mass(n, m);
    for (int i = 0; i < n; i++) {
        float ux, uy, uz;
        sampleInUnitBall(ux, uy, uz);
        h_pos[i] = { 0.5f + R * ux, 0.5f + R * uy, 0.5f + R * uz };
    }

    // ── Device buffers ──────────────────────────────────────────────────
    Vec3  *d_pos, *d_vel, *d_acc;
    float *d_mass;
    Vec2  *d_proj;
    cudaMalloc(&d_pos,  n * sizeof(Vec3));
    cudaMalloc(&d_vel,  n * sizeof(Vec3));
    cudaMalloc(&d_acc,  n * sizeof(Vec3));
    cudaMalloc(&d_mass, n * sizeof(float));
    cudaMalloc(&d_proj, n * sizeof(Vec2));

    cudaMemcpy(d_pos,  h_pos.data(),  n * sizeof(Vec3),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel,  h_vel.data(),  n * sizeof(Vec3),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, h_mass.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_acc, 0, n * sizeof(Vec3));

    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;

    // ── Video writer ────────────────────────────────────────────────────
    cv::VideoWriter writer(
        "cluster.mp4",
        cv::VideoWriter::fourcc('m','p','4','v'),
        30,
        cv::Size(W, H));
    if (!writer.isOpened()) {
        std::cerr << "Failed to open video writer\n";
        return 1;
    }

    std::vector<Vec2> h_proj(n);

    // ── Initial force computation ───────────────────────────────────────
    Tree tree(n);
    tree.rebuild(d_pos, d_vel, d_mass);
    computeForces<<<blocks, threads>>>(tree.nodeData(), tree.arrays(),
                                       d_pos, d_acc, n, theta, eps);

    // ── Main loop ───────────────────────────────────────────────────────
    for (int step = 0; step < steps; step++) {
        halfKick  <<<blocks, threads>>>(d_vel, d_acc, dt, n);
        fullDrift <<<blocks, threads>>>(d_pos, d_vel, dt, n);

        tree.rebuild(d_pos, d_vel, d_mass);
        computeForces<<<blocks, threads>>>(tree.nodeData(), tree.arrays(),
                                           d_pos, d_acc, n, theta, eps);

        halfKick  <<<blocks, threads>>>(d_vel, d_acc, dt, n);

        if (step % stepsPerFrame == 0) {
            // Project 3D positions to 2D image plane on the GPU
            calculateProjection<<<blocks, threads>>>(
                d_pos, d_proj, n,
                cam_pos, cam_forward, cam_up,
                focal, cx, cy);

            cudaMemcpy(h_proj.data(), d_proj, n * sizeof(Vec2),
                       cudaMemcpyDeviceToHost);

            // Render: black background, white dot per body
            cv::Mat frame(H, W, CV_8UC3, cv::Scalar(0, 0, 0));
            for (int i = 0; i < n; i++) {
                int px = static_cast<int>(h_proj[i].x);
                int py = static_cast<int>(h_proj[i].y);
                if (px >= 0 && px < W && py >= 0 && py < H)
                    cv::circle(frame, cv::Point(px, py), 0,
                               cv::Scalar(255, 255, 255), -1);
            }
            writer.write(frame);
        }
    }

    writer.release();
    std::cout << "Wrote cluster.mp4\n";

    cudaFree(d_pos);
    cudaFree(d_vel);
    cudaFree(d_acc);
    cudaFree(d_mass);
    cudaFree(d_proj);
    return 0;
}
