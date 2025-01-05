#include "meshing.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;

void createMesh(const Mat& colorImage, const Mat& depthMap, vector<Vertex>& vertices, vector<Triangle>& triangles) {
    int rows = depthMap.rows;
    int cols = depthMap.cols;

    // Step 1: Create vertices from the color image and depth map
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            // Get depth for the current pixel (in meters or centimeters depending on your depth map scale)
            float depth = depthMap.at<float>(y, x);

            // Create a 3D point using the pixel's (x, y) and depth
            Vertex vertex;
            vertex.x = x;
            vertex.y = y;
            vertex.z = depth;
            vertex.u = static_cast<float>(x) / (cols - 1); // Normalize x to [0, 1]
            vertex.v = static_cast<float>(y) / (rows - 1); // Normalize y to [0, 1]

            // Get color from the panorama (the color image)
            vertex.color = colorImage.at<Vec3b>(y, x);

            // Add the vertex to the list
            vertices.push_back(vertex);
        }
    }

    // Step 2: Create triangles by connecting each pixel to its 4 neighbors
    for (int y = 0; y < rows - 1; ++y) {
        for (int x = 0; x < cols - 1; ++x) {
            // Get indices of the four neighbors in the pixel grid
            int i1 = y * cols + x;           // Top-left pixel
            int i2 = y * cols + (x + 1);     // Top-right pixel
            int i3 = (y + 1) * cols + x;     // Bottom-left pixel
            int i4 = (y + 1) * cols + (x + 1); // Bottom-right pixel

            // Create two triangles for each quadrilateral
            triangles.push_back({i1, i2, i3}); // Triangle 1
            triangles.push_back({i2, i3, i4}); // Triangle 2
        }
    }
}

void saveMeshToOBJWithTears(const string& filename, const vector<Vertex>& vertices, const vector<Triangle>& triangles, const Mat& depthMap, float maxDisparity = 5) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file for writing: " << filename << endl;
        return;
    }

    int rows = depthMap.rows;
    int cols = depthMap.cols;

    // Helper lambda to calculate depth difference
    auto depthDifference = [&](int index1, int index2) {
        float depth1 = depthMap.at<float>(index1 / cols, index1 % cols);
        float depth2 = depthMap.at<float>(index2 / cols, index2 % cols);
        return fabs(depth1 - depth2);
    };

    // Write vertices to the file
    for (const auto& vertex : vertices) {
        file << "v " 
             << vertex.x << " " << vertex.y << " " << vertex.z << " " 
             << vertex.color[2] / 255.0 << " " // B
             << vertex.color[1] / 255.0 << " " // G
             << vertex.color[0] / 255.0 << "\n"; // R
    }

    // Write texture coordinates to the file
    for (const auto& vertex : vertices) {
        file << "vt " << vertex.u << " " << vertex.v << "\n";
    }

    // Write triangles to the file
    for (const auto& triangle : triangles) {
        bool breakV1V2 = depthDifference(triangle.v1, triangle.v2) > maxDisparity;
        bool breakV2V3 = depthDifference(triangle.v2, triangle.v3) > maxDisparity;
        bool breakV3V1 = depthDifference(triangle.v3, triangle.v1) > maxDisparity;

        // Write the face normally if no edges break
        if (!breakV1V2 && !breakV2V3 && !breakV3V1) {
            file << "f "
                 << triangle.v1 + 1 << "/" << triangle.v1 + 1 << " "
                 << triangle.v2 + 1 << "/" << triangle.v2 + 1 << " "
                 << triangle.v3 + 1 << "/" << triangle.v3 + 1 << "\n";
        } else {
            // For torn edges, write separate edges
            if (!breakV1V2) {
                file << "l " << triangle.v1 + 1 << " " << triangle.v2 + 1 << "\n";
            }
            if (!breakV2V3) {
                file << "l " << triangle.v2 + 1 << " " << triangle.v3 + 1 << "\n";
            }
            if (!breakV3V1) {
                file << "l " << triangle.v3 + 1 << " " << triangle.v1 + 1 << "\n";
            }
        }
    }

    file.close();
    cout << "Mesh saved to " << filename << endl;
}

void saveMeshToOBJ(const string& filename, const vector<Vertex>& vertices, const vector<Triangle>& triangles) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file for writing: " << filename << endl;
        return;
    }

    // Write vertices to the file
    for (const auto& vertex : vertices) {
        file << "v " 
             << vertex.x << " " << vertex.y << " " << vertex.z << " " 
             << vertex.color[2] / 255.0 << " " // B
             << vertex.color[1] / 255.0 << " " // G
             << vertex.color[0] / 255.0 << "\n"; // R
    }

    // Write texture coordinates to the file
    for (const auto& vertex : vertices) {
        file << "vt " << vertex.u << " " << vertex.v << "\n";
    }

    // Write triangles (faces) to the file
    for (const auto& triangle : triangles) {
        file << "f "
             << triangle.v1 + 1 << "/" << triangle.v1 + 1 << " "
             << triangle.v2 + 1 << "/" << triangle.v2 + 1 << " "
             << triangle.v3 + 1 << "/" << triangle.v3 + 1 << "\n";
    }

    file.close();
    cout << "Mesh saved to " << filename << endl;
}

void generateMesh(const string& colorImagePath, const string& depthMapPath, const string& outputMeshPath, bool withTears) {
    cout << "Mesh generation begun successfully!" << endl;

    // Load the color image
    Mat colorImage = imread(colorImagePath);
    if (colorImage.empty()) {
        cerr << "Failed to load color image: " << colorImagePath << endl;
        return;
    }

    // Load the depth map
    Mat depthMap = imread(depthMapPath, IMREAD_UNCHANGED);
    if (depthMap.empty()) {
        cerr << "Failed to load depth map: " << depthMapPath << endl;
        return;
    }

    Mat depthMapOut, depthMapFilter;

    medianBlur(depthMap, depthMapFilter, 9);

    depthMapOut = 255 - depthMapFilter;

    // Ensure the depth map is in float format
    depthMapOut.convertTo(depthMapOut, CV_32F);

    // Create the 3D mesh
    vector<Vertex> vertices;
    vector<Triangle> triangles;
    createMesh(colorImage, depthMapOut, vertices, triangles);

    // Save the mesh to an OBJ file
    if (withTears) {
        saveMeshToOBJWithTears(outputMeshPath, vertices, triangles, depthMapOut);
    } else {
    saveMeshToOBJ(outputMeshPath, vertices, triangles);
    }

    cout << "Mesh generation completed successfully!" << endl;
}