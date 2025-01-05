#ifndef MESHING_HPP
#define MESHING_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

// Struct to represent a 3D vertex
struct Vertex {
    float x, y, z;  // 3D coordinates
    Vec3b color;    // RGB color of the vertex
    float u, v;     // Texture coordinates
};

// Struct to represent a triangle (indexing the 3 vertices)
struct Triangle {
    int v1, v2, v3;  // Indices of the vertices
};

// Function to create a 3D mesh from the depth map and colored panorama
void createMesh(const Mat& colorImage, const Mat& depthMap, vector<Vertex>& vertices, vector<Triangle>& triangles);

// Function to save the mesh in OBJ format
void saveMeshToOBJ(const string& filename, const vector<Vertex>& vertices, const vector<Triangle>& triangles);

void generateMesh(const string& colorImagePath, const string& depthMapPath, const string& outputMeshPath);

#endif // MESHING_HPP
