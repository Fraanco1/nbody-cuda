#pragma once

struct BVHNode {
    int  start, end;    // half-open range into sortedIDs[start..end-1]
    int  left,  right;  // indices into nodes[]; -1 = none
    bool isLeaf;
};

// Everything the host needs to drive the build.
struct BVHBuild {
    BVHNode*  nodes;        // device: flat pool [2*N]
    int*      frontier[2];  // device: double-buffered frontier [2*N each]
    int*      frontierSize; // device: [2] — current and next counts
    int*      nodeCounter;  // device: [1] — next free node index
    int       N;
};