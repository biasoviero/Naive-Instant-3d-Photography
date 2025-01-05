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

void generateMesh(const string& colorImagePath, const string& depthMapPath, const string& outputMeshPath) {
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

    // Ensure the depth map is in float format
    depthMap.convertTo(depthMap, CV_32F);

    // Create the 3D mesh
    vector<Vertex> vertices;
    vector<Triangle> triangles;
    createMesh(colorImage, depthMap, vertices, triangles);

    // Save the mesh to an OBJ file
    saveMeshToOBJ(outputMeshPath, vertices, triangles);

    cout << "Mesh generation completed successfully!" << endl;
}