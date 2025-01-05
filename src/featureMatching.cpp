#include "featureMatching.hpp"
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/stitching.hpp>
#include <iostream>

using namespace std;
using namespace cv;

float FeatureExtractionMatching::robustLoss(float s) {
    return log(1 + s);  // Logarithmic function for outlier rejection
}

// Detect Shi-Tomasi corner features
vector<Point2f> FeatureExtractionMatching::detectFeatures(const Mat& image) {
    vector<Point2f> corners;
    int maxCorners = 1000;
    double qualityLevel = 0.01;
    double minDistance = 0.01 * sqrt(image.rows * image.rows + image.cols * image.cols);
    goodFeaturesToTrack(image, corners, maxCorners, qualityLevel, minDistance);
    return corners;
}

// Compute descriptors at given keypoints
Mat FeatureExtractionMatching::computeDescriptors(const Mat& image, vector<KeyPoint>& keypoints) {
    Mat descriptors;

    // Use ORB descriptor (substitute for DAISY in this case)
    Ptr<ORB> orb = ORB::create();
    orb->compute(image, keypoints, descriptors);

    // Ensure descriptors are of type CV_32F for FLANN matching
    if (descriptors.type() != CV_32F) {
        descriptors.convertTo(descriptors, CV_32F);
    }

    return descriptors;
}



// Filter matches using FLANN, the ratio test, and geometric filtering
vector<DMatch> FeatureExtractionMatching::filterMatches(
    const Mat& image1,
    const Mat& image2,
    vector<KeyPoint>& keypoints1,
    vector<KeyPoint>& keypoints2,
    float ratioThreshold,
    float maxOffset
) {
    // Compute descriptors for both images
    Mat descriptors1 = computeDescriptors(image1, keypoints1);
    Mat descriptors2 = computeDescriptors(image2, keypoints2);

    // Initialize the FLANN-based matcher
    FlannBasedMatcher flannMatcher;
    vector<vector<DMatch>> knnMatches;

    // Perform the FLANN matching
    flannMatcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);

    // Apply the ratio test and geometric filtering
    vector<DMatch> filteredMatches;
    
    // Apply the ratio test
    for (const auto& knnMatch : knnMatches) {
        if (knnMatch[0].distance < ratioThreshold * knnMatch[1].distance) {
            filteredMatches.push_back(knnMatch[0]);
        }
    }

    // Geometric filtering
    vector<float> offsets;
    for (const auto& match : filteredMatches) {
        auto delta = keypoints2[match.trainIdx].pt - keypoints1[match.queryIdx].pt;
        offsets.push_back(sqrt(delta.x * delta.x + delta.y * delta.y));
    }

    float medianOffset = calculateMedian(offsets);
    filteredMatches.erase(
        remove_if(filteredMatches.begin(), filteredMatches.end(), [&](const DMatch& match) {
            auto delta = keypoints2[match.trainIdx].pt - keypoints1[match.queryIdx].pt;
            float distance = sqrt(delta.x * delta.x + delta.y * delta.y);
            return distance > medianOffset + maxOffset;
        }),
        filteredMatches.end()
    );

    return filteredMatches;
}

// Helper function to calculate the median of a vector of floats
float FeatureExtractionMatching::calculateMedian(vector<float>& values) {
    if (values.empty()) {
        return 0.0f;
    }
    sort(values.begin(), values.end());
    size_t mid = values.size() / 2;
    return (values.size() % 2 == 0) ? (values[mid - 1] + values[mid]) / 2.0f : values[mid];
}

bool FeatureExtractionMatching::computeCameraPose(
    const vector<Point2f>& pointsA,
    const vector<Point2f>& pointsB,
    const Mat& depthA,
    const Mat& K,
    Mat& R, Mat& t
) {
    // Convert 2D points to 3D using depth map
    vector<Point3f> points3D_A;
    for (size_t i = 0; i < pointsA.size(); i++) {
        float depth = depthA.at<float>(pointsA[i].y, pointsA[i].x);
        if (depth > 0) {
            Point3f point3D(
                (pointsA[i].x - K.at<float>(0, 2)) * depth / K.at<float>(0, 0),
                (pointsA[i].y - K.at<float>(1, 2)) * depth / K.at<float>(1, 1),
                depth
            );
            points3D_A.push_back(point3D);
        }
    }

    // SolvePnP to estimate camera pose (R, t) using matched 2D-3D correspondences
    return solvePnP(points3D_A, pointsB, K, Mat(), R, t, false, SOLVEPNP_ITERATIVE);
}

// Function to compute the reprojection error
float FeatureExtractionMatching::computeReprojectionError(
    const vector<Point2f>& pointsA,
    const vector<Point2f>& pointsB,
    const Mat& R,
    const Mat& t,
    const Mat& K
) {
    float error = 0.0f;
    for (size_t i = 0; i < pointsA.size(); i++) {
        // Project point A to B's frame
        Point3f point3D(
            (pointsA[i].x - K.at<float>(0, 2)) * pointsA[i].x / K.at<float>(0, 0),
            (pointsA[i].y - K.at<float>(1, 2)) * pointsA[i].y / K.at<float>(1, 1),
            pointsA[i].x
        );
        Mat point3DMat = (Mat_<float>(4, 1) << point3D.x, point3D.y, point3D.z, 1);
        
        Mat projected = K * (R * point3DMat + t);
        projected /= projected.at<float>(2); // Homogenize
        
        Point2f reprojPt(projected.at<float>(0), projected.at<float>(1));
        
        error += norm(reprojPt - pointsB[i]);
    }
    return error;
}

void extractAndMatchFeatures(const string& imagePath1, const string& imagePath2) {
    // Load two images (make sure they are in the same folder as your executable or specify full paths)
    Mat image1 = imread(imagePath1, IMREAD_GRAYSCALE);
    Mat image2 = imread(imagePath2, IMREAD_GRAYSCALE);

    // Check if images are loaded correctly
    if (image1.empty() || image2.empty()) {
        cerr << "Could not open or find the images!" << endl;
        return;
    }

    // Feature extraction and matching
    FeatureExtractionMatching matcher;

    // Detect Shi-Tomasi corner features
    vector<Point2f> corners1 = matcher.detectFeatures(image1);
    vector<Point2f> corners2 = matcher.detectFeatures(image2);

    // Convert corner points to keypoints (since DAISY works with keypoints)
    vector<KeyPoint> keypoints1, keypoints2;
    for (const auto& corner : corners1) keypoints1.push_back(KeyPoint(corner, 1));
    for (const auto& corner : corners2) keypoints2.push_back(KeyPoint(corner, 1));

    // Apply feature matching with filtering (ratio threshold = 0.85, max offset = 0.02)
    float ratioThreshold = 0.85f;
    float maxOffset = 0.02f; // 2% of image diagonal
    vector<DMatch> matches = matcher.filterMatches(image1, image2, keypoints1, keypoints2, ratioThreshold, maxOffset);

    // Display the matches
    Mat imgMatches;
    drawMatches(image1, keypoints1, image2, keypoints2, matches, imgMatches);
    imshow("Matches", imgMatches);
    waitKey(0);
}

void alignDepthMaps(const string& depthMapPath1, const string& depthMapPath2, const Mat& K) {
    // Load the depth maps
    Mat depthMap1 = imread(depthMapPath1, IMREAD_GRAYSCALE);
    Mat depthMap2 = imread(depthMapPath2, IMREAD_GRAYSCALE);

    // Check if depth maps are loaded correctly
    if (depthMap1.empty() || depthMap2.empty()) {
        cerr << "Failed to load depth maps!" << endl;
        return;
    }

    // Ensure the depth maps are in CV_32FC1 format
    depthMap1.convertTo(depthMap1, CV_32FC1);
    depthMap2.convertTo(depthMap2, CV_32FC1);

    // Detect a single feature point in each depth map (e.g., the maximum depth point)
    Point point1, point2;
    double minVal, maxVal;
    minMaxLoc(depthMap1, &minVal, &maxVal, nullptr, &point1);
    minMaxLoc(depthMap2, &minVal, &maxVal, nullptr, &point2);

    // Convert Points to Point2f
    Point2f point1f(point1.x, point1.y);
    Point2f point2f(point2.x, point2.y);

    // Compute the translation vector based on the matched points
    Point2f translation = point2f - point1f;

    // Create a 2x3 transformation matrix
    Mat T = Mat::eye(2, 3, CV_32F);
    T.at<float>(0, 2) = translation.x;
    T.at<float>(1, 2) = translation.y;

    // Apply the transformation to align the second depth map
    Mat alignedDepthMap;
    warpAffine(depthMap2, alignedDepthMap, T, depthMap2.size());

    // Display the aligned depth map
    Mat alignedDisplay;
    normalize(alignedDepthMap, alignedDisplay, 0, 255, NORM_MINMAX);
    alignedDisplay.convertTo(alignedDisplay, CV_8UC1);
    imshow("Aligned Depth Map", alignedDisplay);
}

bool loadAndStitchImages(const string& basePath, int numImages, Mat& panorama, vector<Mat>& transformations) {
    vector<Mat> images;

    cout << "Loading images..." << endl;
    for (int i = 1; i <= numImages; ++i) {
        string imagePath = basePath + to_string(i) + ".jpg";
        Mat img = imread(imagePath);
        if (img.empty()) {
            cerr << "Error loading image " << imagePath << endl;
            return false;
        }
        images.push_back(img);
        cout << "Loaded image: " << imagePath << endl;
    }

    cout << "Stitching images..." << endl;
    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA);
    stitcher->setPanoConfidenceThresh(0.8);
    stitcher->setWaveCorrection(true);

    Stitcher::Status status = stitcher->stitch(images, panorama);

    if (status != Stitcher::OK) {
        cerr << "Error during stitching: " << status << endl;
        return false;
    }

    cout << "Stitching completed successfully." << endl;

    imwrite("colorPanorama.jpg", panorama);
    cout << "Saved color panorama as colorPanorama.jpg" << endl;
    return true;
}

bool loadAndStitchImages(const string& colorBasePath, const string& depthBasePath, int numImages, Mat& panorama, Mat& depthPanorama) {
    vector<Mat> images;
    vector<Mat> depthMaps;

    for (int i = 1; i <= numImages; ++i) {
        string imagePath = colorBasePath + to_string(i) + ".jpg";
        string depthPath = depthBasePath + to_string(i) + ".jpg"; // Assuming depth maps are saved as PNGs
        Mat img = imread(imagePath);
        Mat depthMap = imread(depthPath, IMREAD_UNCHANGED);  // Load depth map in original format (e.g., 16-bit)
        
        if (img.empty() || depthMap.empty()) {
            cerr << "Error loading image or depth map: " << imagePath << endl;
            return false;
        }
        
        images.push_back(img);
        depthMaps.push_back(depthMap);
    }

    // Stitch color images
    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA);
    stitcher->setPanoConfidenceThresh(0.8);
    stitcher->setWaveCorrection(true);

    Stitcher::Status status = stitcher->stitch(images, panorama);
    if (status != Stitcher::OK) {
        cerr << "Error during stitching: " << status << endl;
        return false;
    }

    imwrite("colorPanorama.jpg", panorama);

    Ptr<Stitcher> depthStitcher = Stitcher::create(Stitcher::PANORAMA);
    depthStitcher->setPanoConfidenceThresh(0.8);
    depthStitcher->setWaveCorrection(true);

    Stitcher::Status depthStatus = depthStitcher->stitch(depthMaps, depthPanorama);
    if (depthStatus != Stitcher::OK) {
        cerr << "Error during stitching: " << depthStatus << endl;
        return false;
    }

    imwrite("depthPanorama.jpg", depthPanorama);

    // // Now, stitch the depth maps using the same transformations
    // if (!stitchDepthMaps(depthMaps, images, depthPanorama)) {
    //     cerr << "Error during depth map stitching" << endl;
    //     return false;
    // }

    return true;
}

bool stitchDepthMaps(const vector<Mat>& depthMaps, const vector<Mat>& images, Mat& depthPanorama) {
    // Feature detection and matching setup
    vector<detail::ImageFeatures> features(images.size());
    vector<detail::MatchesInfo> pairwiseMatches;

    Ptr<SIFT> detector = SIFT::create(); // or another feature detector
    for (size_t i = 0; i < images.size(); ++i) {
        features[i].img_idx = i;
        detector->detectAndCompute(images[i], Mat(), features[i].keypoints, features[i].descriptors);
    }

    // Match features between images
    Ptr<detail::BestOf2NearestMatcher> matcher = makePtr<detail::BestOf2NearestMatcher>(false, 0.3f);
    matcher->operator()(features, pairwiseMatches);
    matcher->collectGarbage();

    // Estimate homographies between the images
    vector<Mat> homographies(images.size());
    homographies[0] = Mat::eye(3, 3, CV_32F);
    for (size_t i = 1; i < images.size(); ++i) {
        homographies[i] = pairwiseMatches[i - 1].H.inv();
    }

    // Apply homographies to the depth maps
    vector<Mat> alignedDepthMaps;
    for (size_t i = 0; i < depthMaps.size(); ++i) {
        Mat warpedDepth;
        // Apply the homography to warp the depth map
        warpPerspective(depthMaps[i], warpedDepth, homographies[i], depthPanorama.size());
        alignedDepthMaps.push_back(warpedDepth);
    }

    // Here you can blend the depth maps or use the stitching function
    for (const auto& depthMap : alignedDepthMaps) {
        // For now, just replace the depthPanorama with the last one
        // You may want to apply blending or averaging.
        depthPanorama = depthMap;
    }

    imwrite("depthPanorama.jpg", depthPanorama);
    return true;
}