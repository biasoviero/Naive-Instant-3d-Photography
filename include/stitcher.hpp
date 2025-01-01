#ifndef STITCHER_HPP
#define STITCHER_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <utility>

class Stitcher {
public:
    /**
     * Constructor
     * @param width Width of the final panorama.
     * @param height Height of the final panorama.
     */
    Stitcher(int width, int height);

    /**
     * Set the aligned images and their associated depth maps.
     * @param images Vector of aligned color images.
     * @param depthMaps Vector of aligned depth maps.
     * @param cameraFrontVectors Vector of camera front vectors.
     */
    void setInput(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& depthMaps, const std::vector<cv::Point3f>& cameraFrontVectors);

    /**
     * Compute the center of projection for the panorama.
     */
    void computeCenterOfProjection();

    /**
     * Render equirectangular panoramas from the central viewpoint.
     * @return Pair of equirectangular panoramas (color, depth).
     */
    std::pair<cv::Mat, cv::Mat> renderEquirectangularPanorama();

    /**
     * Stitch the panoramas into a seamless mosaic.
     * @return Final stitched panorama as a color image.
     */
    cv::Mat stitchPanorama();

private:
    // Panorama dimensions
    int panoramaWidth;
    int panoramaHeight;

    // Input images, depth maps, and camera front vectors
    std::vector<cv::Mat> colorImages;
    std::vector<cv::Mat> depthMaps;
    std::vector<cv::Point3f> cameraFrontVectors;

    // Center of projection
    cv::Point3f centerOfProjection;

    // Helper functions

    /**
     * Compute the 3D center point that minimizes distance to all camera front vectors.
     */
    void computeProjectionCenter();

    /**
     * Compute depth consensus for a given pixel.
     * @param p Pixel coordinates.
     * @param sourceIndex Index of the source image.
     * @return Depth consensus penalty.
     */
    double computeDepthConsensus(const cv::Point& p, int sourceIndex);

    /**
     * Compute smoothness penalty for stitching.
     * @param p Pixel coordinates.
     * @param neighborPixel Coordinates of the neighboring pixel.
     * @return Smoothness penalty.
     */
    double computeSmoothnessPenalty(const cv::Point& p, const cv::Point& neighborPixel);

    /**
     * Select the optimal source image for a given pixel.
     * @param p Pixel coordinates.
     * @return Index of the selected source image.
     */
    int selectOptimalSource(const cv::Point& p);
};

#endif // STITCHER_HPP