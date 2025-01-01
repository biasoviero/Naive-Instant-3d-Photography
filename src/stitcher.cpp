#include "stitcher.hpp"
#include <limits>
#include <cmath>

// Constructor
Stitcher::Stitcher(int width, int height)
    : panoramaWidth(width), panoramaHeight(height), centerOfProjection(0.0f, 0.0f, 0.0f) {}

// Set input images, depth maps, and camera front vectors
void Stitcher::setInput(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& depthMaps, const std::vector<cv::Point3f>& cameraFrontVectors) {
    colorImages = images;
    this->depthMaps = depthMaps;
    this->cameraFrontVectors = cameraFrontVectors;
}

// Compute the center of projection
void Stitcher::computeCenterOfProjection() {
    cv::Point3f sum(0.0f, 0.0f, 0.0f);
    for (const auto& frontVector : cameraFrontVectors) {
        sum += frontVector;
    }
    centerOfProjection = sum / static_cast<float>(cameraFrontVectors.size());
}

// Render equirectangular panoramas
std::pair<cv::Mat, cv::Mat> Stitcher::renderEquirectangularPanorama() {
    cv::Mat colorPanorama = cv::Mat::zeros(panoramaHeight, panoramaWidth, CV_8UC3);
    cv::Mat depthPanorama = cv::Mat::zeros(panoramaHeight, panoramaWidth, CV_32F);

    for (size_t i = 0; i < colorImages.size(); ++i) {
        // Project each image into the equirectangular panorama
        for (int y = 0; y < colorImages[i].rows; ++y) {
            for (int x = 0; x < colorImages[i].cols; ++x) {
                // Compute the spherical coordinates
                cv::Point3f direction = cameraFrontVectors[i]; // Placeholder for actual projection logic

                // Convert spherical coordinates to panorama pixel coordinates
                int u = static_cast<int>((atan2(direction.x, direction.z) + CV_PI) / (2 * CV_PI) * panoramaWidth);
                int v = static_cast<int>((asin(direction.y) / CV_PI + 0.5) * panoramaHeight);

                // Copy color and depth information
                if (u >= 0 && u < panoramaWidth && v >= 0 && v < panoramaHeight) {
                    colorPanorama.at<cv::Vec3b>(v, u) = colorImages[i].at<cv::Vec3b>(y, x);
                    depthPanorama.at<float>(v, u) = depthMaps[i].at<float>(y, x);
                }
            }
        }
    }

    return {colorPanorama, depthPanorama};
}

// Stitch the panoramas into a seamless mosaic
cv::Mat Stitcher::stitchPanorama() {
    auto [colorPanorama, depthPanorama] = renderEquirectangularPanorama();

    cv::Mat stitchedPanorama = colorPanorama.clone();

    for (int y = 0; y < panoramaHeight; ++y) {
        for (int x = 0; x < panoramaWidth; ++x) {
            int bestSource = selectOptimalSource(cv::Point(x, y));
            if (bestSource >= 0) {
                stitchedPanorama.at<cv::Vec3b>(y, x) = colorImages[bestSource].at<cv::Vec3b>(y, x);
            }
        }
    }

    return stitchedPanorama;
}

// Compute depth consensus
double Stitcher::computeDepthConsensus(const cv::Point& p, int sourceIndex) {
    double consensus = 0.0;
    for (size_t i = 0; i < depthMaps.size(); ++i) {
        if (i != sourceIndex) {
            float depthRatio = depthMaps[i].at<float>(p) / depthMaps[sourceIndex].at<float>(p);
            if (depthRatio >= 0.9 && depthRatio <= 1.1) {
                consensus += 1.0;
            }
        }
    }
    return consensus;
}

// Compute smoothness penalty
double Stitcher::computeSmoothnessPenalty(const cv::Point& p, const cv::Point& neighborPixel) {
    double penalty = 0.0;
    for (size_t i = 0; i < colorImages.size(); ++i) {
        penalty += cv::norm(colorImages[i].at<cv::Vec3b>(p) - colorImages[i].at<cv::Vec3b>(neighborPixel));
    }
    return penalty;
}

// Select the optimal source image for a given pixel
int Stitcher::selectOptimalSource(const cv::Point& p) {
    int bestSource = -1;
    double bestScore = std::numeric_limits<double>::max();

    for (size_t i = 0; i < colorImages.size(); ++i) {
        double depthPenalty = computeDepthConsensus(p, i);
        double score = depthPenalty; // Add more penalties as required

        if (score < bestScore) {
            bestScore = score;
            bestSource = static_cast<int>(i);
        }
    }

    return bestSource;
}
