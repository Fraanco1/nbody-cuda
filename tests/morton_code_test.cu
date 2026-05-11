// Allocate node data and flags
NodeData* d_nodeData;
int*      d_flags;
cudaMalloc(&d_nodeData, (2 * n - 1) * sizeof(NodeData));
cudaMalloc(&d_flags,    (n - 1)     * sizeof(int));
cudaMemset(d_flags, 0,  (n - 1)     * sizeof(int));

// Uniform mass for now
float* d_masses;
cudaMalloc(&d_masses, n * sizeof(float));
thrust::device_ptr<float> massPtr(d_masses);
thrust::fill(massPtr, massPtr + n, 1.0f);

// Step 1 — init leaves
int leafBlocks = (n + threads - 1) / threads;
initLeaves<<<leafBlocks, threads>>>(d_nodeData, d_points, d_masses, n);
cudaDeviceSynchronize();

// Step 2 — bottom-up pass
computeAABBandCoM<<<leafBlocks, threads>>>(d_nodeData, bvh, d_flags, n);
cudaDeviceSynchronize();

// Verify — root node (index 0) should have CoM and AABB of all bodies
NodeData h_root;
cudaMemcpy(&h_root, &d_nodeData[0], sizeof(NodeData), cudaMemcpyDeviceToHost);

cout << "Root AABB: "
     << "[" << h_root.minX << ", " << h_root.maxX << "] "
     << "[" << h_root.minY << ", " << h_root.maxY << "] "
     << "[" << h_root.minZ << ", " << h_root.maxZ << "]" << endl;
cout << "Root CoM: "
     << h_root.comX << " "
     << h_root.comY << " "
     << h_root.comZ << endl;
cout << "Root total mass: " << h_root.mass << endl;

// Cleanup
cudaFree(d_nodeData);
cudaFree(d_flags);
cudaFree(d_masses);