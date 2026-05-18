// N-body gravitational cluster rendered with a pinhole camera via OpenCV.
//
// Initial conditions are read from a binary file produced by generate_ic.py.
// File format: [int32 n] [n * 7 * float32: x y z vx vy vz mass]
//
// Output: cluster.mp4

#include <cstdio>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "tree.cuh"
#include "forces.cuh"
#include "graphic.cuh"

int main(int argc, char* argv[]) {
    // ── Simulation parameters ───────────────────────────────────────────
    float dt            = 0.0005f;
    int   steps         = 2000;
    float theta         = 0.5f;
    float eps           = 0.1f;
    float G             = 1.0f;
    int   stepsPerFrame = 5;
    float focal         = 600.0f;
    std::string ic_file = "";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (i + 1 >= argc) { std::cerr << "Missing value for " << arg << "\n"; return 1; }
        if      (arg == "--dt")            dt            = std::atof(argv[++i]);
        else if (arg == "--steps")         steps         = std::atoi(argv[++i]);
        else if (arg == "--theta")         theta         = std::atof(argv[++i]);
        else if (arg == "--eps")           eps           = std::atof(argv[++i]);
        else if (arg == "--stepsPerFrame") stepsPerFrame = std::atoi(argv[++i]);
        else if (arg == "--G")             G             = std::atof(argv[++i]);
        else if (arg == "--focal")         focal         = std::atof(argv[++i]);
        else if (arg == "--ic")            ic_file       = argv[++i];
        else { std::cerr << "Unknown argument: " << arg << "\n"; return 1; }
    }

    if (ic_file.empty()) {
        std::cerr << "Usage: " << argv[0] << " --ic <file> [options]\n"
                  << "  Generate IC file with: python generate_ic.py\n";
        return 1;
    }

    // ── Load initial conditions ─────────────────────────────────────────
    FILE* f = std::fopen(ic_file.c_str(), "rb");
    if (!f) { std::cerr << "Cannot open IC file: " << ic_file << "\n"; return 1; }

    int n = 0;
    std::fread(&n, sizeof(int), 1, f);

    std::vector<Vec3>  h_pos(n);
    std::vector<Vec3>  h_vel(n);
    std::vector<float> h_mass(n);

    for (int i = 0; i < n; i++) {
        float buf[7];
        std::fread(buf, sizeof(float), 7, f);
        h_pos[i]  = { buf[0], buf[1], buf[2] };
        h_vel[i]  = { buf[3], buf[4], buf[5] };
        h_mass[i] = buf[6];
    }
    std::fclose(f);

    // ── Camera / image parameters ───────────────────────────────────────
    const int   W      = 1024;
    const int   H      = 1024;
    Vec3 cam_pos     = { 0.5f,  0.5f, 2.5f };
    Vec3 cam_forward = { 0.0f,  0.0f, -1.0f };
    Vec3 cam_up      = { 0.0f,  1.0f,  0.0f };
    const float cx   = W / 2.0f;
    const float cy   = H / 2.0f;

    // ── Device buffers ──────────────────────────────────────────────────
    Vec3  *d_pos, *d_vel, *d_acc;
    float *d_mass, *d_accSum, *d_comVel, *d_speeds;
    Vec2  *d_proj;
    cudaMalloc(&d_pos,    n * sizeof(Vec3));
    cudaMalloc(&d_vel,    n * sizeof(Vec3));
    cudaMalloc(&d_acc,    n * sizeof(Vec3));
    cudaMalloc(&d_mass,   n * sizeof(float));
    cudaMalloc(&d_proj,   n * sizeof(Vec2));
    cudaMalloc(&d_accSum, 3 * sizeof(float));
    cudaMalloc(&d_comVel, 3 * sizeof(float));
    cudaMalloc(&d_speeds, n * sizeof(float));

    cudaMemcpy(d_pos,  h_pos.data(),  n * sizeof(Vec3),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel,  h_vel.data(),  n * sizeof(Vec3),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, h_mass.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_acc, 0, n * sizeof(Vec3));

    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;

    // Subtract the mean acceleration from all bodies to cancel the net momentum
    // error introduced by the Barnes-Hut approximation each step.
    auto removeDrift = [&]() {
        cudaMemset(d_accSum, 0, 3 * sizeof(float));
        accumulateAcc<<<blocks, threads, 3 * threads * sizeof(float)>>>(d_acc, d_accSum, n);
        subtractMeanAcc<<<blocks, threads>>>(d_acc, d_accSum, n);
    };

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

    std::vector<Vec2>  h_proj(n);
    std::vector<float> h_speeds(n);

    // ── Initial force computation ───────────────────────────────────────
    Tree tree(n);
    tree.rebuild(d_pos, d_vel, d_mass);
    computeForces<<<blocks, threads>>>(tree.nodeData(), tree.arrays(),
                                       d_pos, d_acc, n, theta, eps, G);
    removeDrift();

    // ── Main loop ───────────────────────────────────────────────────────
    for (int step = 0; step < steps; step++) {
        halfKick  <<<blocks, threads>>>(d_vel, d_acc, dt, n);
        fullDrift <<<blocks, threads>>>(d_pos, d_vel, dt, n);

        tree.rebuild(d_pos, d_vel, d_mass);
        computeForces<<<blocks, threads>>>(tree.nodeData(), tree.arrays(),
                                           d_pos, d_acc, n, theta, eps, G);
        removeDrift();

        halfKick  <<<blocks, threads>>>(d_vel, d_acc, dt, n);

        if (step % stepsPerFrame == 0) {
            calculateProjection<<<blocks, threads>>>(
                d_pos, d_proj, n,
                cam_pos, cam_forward, cam_up,
                focal, cx, cy);

            // Compute per-body speed relative to CoM velocity.
            // Total mass = 1, so sum(m_i * v_i) == v_com directly.
            cudaMemset(d_comVel, 0, 3 * sizeof(float));
            accumulateMomenta<<<blocks, threads, 3 * threads * sizeof(float)>>>(
                d_vel, d_mass, d_comVel, n);
            computeRelativeSpeeds<<<blocks, threads>>>(d_vel, d_comVel, d_speeds, n);

            cudaMemcpy(h_proj.data(),   d_proj,   n * sizeof(Vec2),  cudaMemcpyDeviceToHost);
            cudaMemcpy(h_speeds.data(), d_speeds, n * sizeof(float), cudaMemcpyDeviceToHost);

            // Normalize speeds to [0,1] and map to blue→cyan→green→yellow→red.
            float maxSpeed = *std::max_element(h_speeds.begin(), h_speeds.end());
            float invMax   = (maxSpeed > 0.f) ? 1.f / maxSpeed : 1.f;

            auto heatmap = [](float t) -> cv::Scalar {
                t = std::max(0.f, std::min(1.f, t));
                float r, g, b;
                if      (t < 0.25f) { r = 0;                    g = t / 0.25f;          b = 1; }
                else if (t < 0.5f)  { r = 0;                    g = 1;                  b = 1 - (t - 0.25f) / 0.25f; }
                else if (t < 0.75f) { r = (t - 0.5f)  / 0.25f; g = 1;                  b = 0; }
                else                { r = 1;                    g = 1 - (t - 0.75f) / 0.25f; b = 0; }
                return cv::Scalar(b * 255, g * 255, r * 255);  // OpenCV uses BGR
            };

            cv::Mat frame(H, W, CV_8UC3, cv::Scalar(0, 0, 0));
            for (int i = 0; i < n; i++) {
                int px = static_cast<int>(h_proj[i].x);
                int py = static_cast<int>(h_proj[i].y);
                if (px >= 0 && px < W && py >= 0 && py < H)
                    cv::circle(frame, cv::Point(px, py), 0,
                               heatmap(h_speeds[i] * invMax), -1);
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
    cudaFree(d_accSum);
    cudaFree(d_comVel);
    cudaFree(d_speeds);
    return 0;
}
