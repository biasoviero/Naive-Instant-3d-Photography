#ifndef FEATURE_MATCHING_HPP
#define FEATURE_MATCHING_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>
#include <vector>
#include <cmath>

using namespace std;
using namespace cv;

class FeatureExtractionMatching {
public:
    // Detect Shi-Tomasi corner features
    static vector<Point2f> detectFeatures(const Mat& image);

    // Compute descriptors at given keypoints
    static Mat computeDescriptors(const Mat& image, vector<KeyPoint>& keypoints);

    // Filter matches using the ratio test and geometric filtering
    static vector<DMatch> filterMatches(
    const Mat& image1,
    const Mat& image2,
    vector<KeyPoint>& keypoints1,
    vector<KeyPoint>& keypoints2,
    float ratioThreshold,
    float maxOffset
    );

    static bool computeCameraPose(
        const vector<Point2f>& pointsA,
        const vector<Point2f>& pointsB,
        const Mat& depthA,
        const Mat& K,
        Mat& R, Mat& t
    );

    static float robustLoss(float s);

    static float FeatureExtractionMatching::computeReprojectionError(
    const vector<Point2f>& pointsA,
    const vector<Point2f>& pointsB,
    const Mat& R,
    const Mat& t,
    const Mat& K
    );

private:
    // Helper function to calculate the median of a vector of floats
    static float calculateMedian(vector<float>& values);
};

void extractAndMatchFeatures(const string& imagePath1, const string& imagePath2);

void alignDepthMaps(const string& depthMapPath1, const string& depthMapPath2, const Mat& K);

void alignDepthMaps(const string& depthMapPath1, const string& depthMapPath2, const Mat& K);

bool loadAndStitchImages(const string& colorBasePath, const string& depthBasePath, int numImages, Mat& panorama, Mat& depthPanorama);

bool stitchDepthMaps(const vector<Mat>& depthMaps, const vector<Mat>& images, Mat& depthPanorama);

#endif // FEATURE_MATCHING_HPP