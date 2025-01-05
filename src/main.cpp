#include "featureMatching.hpp"
#include "meshing.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
#include <iostream>
#include <windows.h>

#include "featureMatching.hpp"
#include "meshing.hpp"

using namespace std;
using namespace cv;

string getRootPath() {
    char buffer[MAX_PATH];
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    string path(buffer);

    string::size_type pos = path.find_last_of("\\/");
	path = path.substr(0, pos);
    pos = path.find_last_of("\\/");
	path = path.substr(0, pos);
	pos = path.find_last_of("\\/");

    if (pos != std::string::npos) {
        return path.substr(0, pos);
    }
    return "";
}

Mat loadImage(const string &path) {
    string completePath = getRootPath() + path;
    return imread(completePath, IMREAD_UNCHANGED);
}

int main() {

    // extractAndMatchFeatures("C:/Users/biaso/Desktop/UFRGS/semestre4/FPI/trabFinal/Naive-Instant-3d-Photography/input/images/image1.png",
    //                         "C:/Users/biaso/Desktop/UFRGS/semestre4/FPI/trabFinal/Naive-Instant-3d-Photography/input/images/image2.png");

    // Mat K = (Mat_<float>(3, 3) << 500, 0, 320, 0, 500, 240, 0, 0, 1);  // Example intrinsic matrix
    // string depthMapPath1 = "C:/Users/biaso/Desktop/UFRGS/semestre4/FPI/trabFinal/Naive-Instant-3d-Photography/input/depthMaps/image1.png";  // Path to the first depth map
    // string depthMapPath2 = "C:/Users/biaso/Desktop/UFRGS/semestre4/FPI/trabFinal/Naive-Instant-3d-Photography/input/depthMaps/image1.png";  // Path to the second depth map

    string colorBasePath = getRootPath() + "\\input\\scene2\\images\\image";
    String depthBasePath = getRootPath() + "\\input\\scene2\\depthMaps\\image";
    int numImages = 9;

    // Load and stitch color images
    Mat colorPanorama, depthPanorama;
    vector<Mat> transformations;
    if (!loadAndStitchImages(colorBasePath, depthBasePath, numImages, colorPanorama, depthPanorama)) {
        return -1;
    }

    const string colorImagePath = getRootPath() + "\\input\\scene2\\images\\image4.jpg";
    const string depthMapPath = getRootPath() + "\\input\\scene2\\depthMaps\\image4.png";

    const string outputMeshPath = getRootPath() + "\\output\\mesh.obj";

    // Call the mesh generation method
    generateMesh(colorImagePath, depthMapPath, outputMeshPath, false);

    return 0;
}
