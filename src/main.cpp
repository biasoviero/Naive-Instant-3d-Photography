#include "featureMatching.hpp"
#include "meshing.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main() {

    // extractAndMatchFeatures("C:/Users/biaso/Desktop/UFRGS/semestre4/FPI/trabFinal/Naive-Instant-3d-Photography/input/images/image1.png",
    //                         "C:/Users/biaso/Desktop/UFRGS/semestre4/FPI/trabFinal/Naive-Instant-3d-Photography/input/images/image2.png");

    // Mat K = (Mat_<float>(3, 3) << 500, 0, 320, 0, 500, 240, 0, 0, 1);  // Example intrinsic matrix
    // string depthMapPath1 = "C:/Users/biaso/Desktop/UFRGS/semestre4/FPI/trabFinal/Naive-Instant-3d-Photography/input/depthMaps/image1.png";  // Path to the first depth map
    // string depthMapPath2 = "C:/Users/biaso/Desktop/UFRGS/semestre4/FPI/trabFinal/Naive-Instant-3d-Photography/input/depthMaps/image1.png";  // Path to the second depth map

    string colorBasePath = "C:/Users/biaso/Desktop/UFRGS/semestre4/FPI/trabFinal/Naive-Instant-3d-Photography/input/images2/images/image";
    String depthBasePath = "C:/Users/biaso/Desktop/UFRGS/semestre4/FPI/trabFinal/Naive-Instant-3d-Photography/input/images2/depthMaps/image";
    int numImages = 9;

    // Load and stitch color images
    Mat colorPanorama, depthPanorama;
    vector<Mat> transformations;
    if (!loadAndStitchImages(colorBasePath, depthBasePath, numImages, colorPanorama, depthPanorama)) {
        return -1;
    }

    const string colorImagePath = "C:/Users/biaso/Desktop/UFRGS/semestre4/FPI/trabFinal/Naive-Instant-3d-Photography/input/images/image1.png";
    const string depthMapPath = "C:/Users/biaso/Desktop/UFRGS/semestre4/FPI/trabFinal/Naive-Instant-3d-Photography/input/depthMaps/image1.png";
    const string outputMeshPath = "C:/Users/biaso/Desktop/UFRGS/semestre4/FPI/trabFinal/Naive-Instant-3d-Photography/output/mesh.obj";

    // Call the mesh generation method
    generateMesh(colorImagePath, depthMapPath, outputMeshPath);

    return 0;
}
