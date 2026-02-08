/*
  Name: Borui Chen
  Date: 2026-02-03
  Purpose: Feature extraction implementations for CBIR system.
*/

#include "feature.h"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

// Normalize feature vector (L2 normalization)
void FeatureVector::normalize() {
    float sum = 0.0f;
    for (float val : data) {
        sum += val * val;
    }
    float magnitude = std::sqrt(sum);
    if (magnitude > 0) {
        for (float& val : data) {
            val /= magnitude;
        }
    }
}

// Convert FeatureType to string
std::string featureTypeToString(FeatureType type) {
    switch (type) {
        case FeatureType::BASELINE: return "baseline";
        case FeatureType::HISTOGRAM: return "histogram";
        case FeatureType::MULTI_HISTOGRAM: return "multi_histogram";
        case FeatureType::TEXTURE_COLOR: return "texture_color";
        case FeatureType::DNN_EMBEDDING: return "dnn_embedding";
        case FeatureType::CUSTOM: return "custom";
        default: return "unknown";
    }
}

// Convert string to FeatureType
FeatureType stringToFeatureType(const std::string& str) {
    if (str == "baseline") return FeatureType::BASELINE;
    if (str == "histogram") return FeatureType::HISTOGRAM;
    if (str == "multi_histogram") return FeatureType::MULTI_HISTOGRAM;
    if (str == "texture_color") return FeatureType::TEXTURE_COLOR;
    if (str == "dnn_embedding") return FeatureType::DNN_EMBEDDING;
    if (str == "custom") return FeatureType::CUSTOM;
    return FeatureType::BASELINE;
}

// Task 1: Baseline feature - 7x7 center square (147 dimensions)
int extractBaseline(const cv::Mat& image, FeatureVector& feature) {
    if (image.empty()) {
        return -1;
    }

    // Initialize feature vector (7x7x3 = 147 dimensions)
    feature = FeatureVector(147, FeatureType::BASELINE);

    // Calculate center region
    int centerX = image.cols / 2;
    int centerY = image.rows / 2;
    int startX = centerX - 3; // 7x7 window centered
    int startY = centerY - 3;

    // Handle edge cases where image is smaller than 7x7
    if (image.cols < 7 || image.rows < 7) {
        // For small images, still extract center but handle bounds
        startX = std::max(0, centerX - 3);
        startY = std::max(0, centerY - 3);
    }

    int idx = 0;
    for (int y = 0; y < 7; y++) {
        for (int x = 0; x < 7; x++) {
            int px = startX + x;
            int py = startY + y;

            // Clamp to image bounds
            px = std::max(0, std::min(px, image.cols - 1));
            py = std::max(0, std::min(py, image.rows - 1));

            cv::Vec3b pixel = image.at<cv::Vec3b>(py, px);
            feature[idx++] = static_cast<float>(pixel[0]); // B
            feature[idx++] = static_cast<float>(pixel[1]); // G
            feature[idx++] = static_cast<float>(pixel[2]); // R
        }
    }

    return 0;
}

// Task 2: Histogram feature - 3D RGB histogram
int extractHistogram(const cv::Mat& image, FeatureVector& feature, int binsPerChannel) {
    if (image.empty()) {
        return -1;
    }

    // Initialize histogram (binsPerChannel^3 dimensions)
    int totalBins = binsPerChannel * binsPerChannel * binsPerChannel;
    feature = FeatureVector(totalBins, FeatureType::HISTOGRAM);

    float binSize = 256.0f / binsPerChannel;
    int totalPixels = image.rows * image.cols;

    // Compute histogram
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);

            int bBin = static_cast<int>(pixel[0] / binSize);
            int gBin = static_cast<int>(pixel[1] / binSize);
            int rBin = static_cast<int>(pixel[2] / binSize);

            // Clamp to valid bin range
            bBin = std::min(bBin, binsPerChannel - 1);
            gBin = std::min(gBin, binsPerChannel - 1);
            rBin = std::min(rBin, binsPerChannel - 1);

            int idx = (rBin * binsPerChannel + gBin) * binsPerChannel + bBin;
            feature[idx] += 1.0f;
        }
    }

    // Normalize histogram
    for (size_t i = 0; i < feature.size(); i++) {
        feature[i] /= totalPixels;
    }

    return 0;
}

// Task 3: Multi-histogram feature
int extractMultiHistogram(const cv::Mat& image, FeatureVector& feature,
                          int binsPerChannel, bool splitHorizontal) {
    if (image.empty()) {
        return -1;
    }

    int binsPerRegion = binsPerChannel * binsPerChannel * binsPerChannel;
    feature = FeatureVector(binsPerRegion * 2, FeatureType::MULTI_HISTOGRAM);

    float binSize = 256.0f / binsPerChannel;

    // Define regions
    cv::Rect region1, region2;
    if (splitHorizontal) {
        // Split into top and bottom halves
        region1 = cv::Rect(0, 0, image.cols, image.rows / 2);
        region2 = cv::Rect(0, image.rows / 2, image.cols, image.rows - image.rows / 2);
    } else {
        // Split into left and right halves
        region1 = cv::Rect(0, 0, image.cols / 2, image.rows);
        region2 = cv::Rect(image.cols / 2, 0, image.cols - image.cols / 2, image.rows);
    }

    // Helper lambda to compute histogram for a region
    auto computeRegionHist = [&](const cv::Rect& roi, int offset) {
        int pixelCount = 0;
        for (int y = roi.y; y < roi.y + roi.height && y < image.rows; y++) {
            for (int x = roi.x; x < roi.x + roi.width && x < image.cols; x++) {
                cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);

                int bBin = static_cast<int>(pixel[0] / binSize);
                int gBin = static_cast<int>(pixel[1] / binSize);
                int rBin = static_cast<int>(pixel[2] / binSize);

                bBin = std::min(bBin, binsPerChannel - 1);
                gBin = std::min(gBin, binsPerChannel - 1);
                rBin = std::min(rBin, binsPerChannel - 1);

                int idx = (rBin * binsPerChannel + gBin) * binsPerChannel + bBin;
                feature[offset + idx] += 1.0f;
                pixelCount++;
            }
        }

        // Normalize region histogram
        if (pixelCount > 0) {
            for (int i = 0; i < binsPerRegion; i++) {
                feature[offset + i] /= pixelCount;
            }
        }
    };

    computeRegionHist(region1, 0);
    computeRegionHist(region2, binsPerRegion);

    return 0;
}

// Helper: Compute gradient magnitude using Sobel operators
int computeGradientMagnitude(const cv::Mat& image, cv::Mat& magnitudeImg) {
    // Convert to grayscale for texture analysis
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    // Compute Sobel gradients
    cv::Mat sobelX, sobelY;
    cv::Sobel(gray, sobelX, CV_32F, 1, 0, 3);
    cv::Sobel(gray, sobelY, CV_32F, 0, 1, 3);

    // Compute magnitude
    cv::Mat mag;
    cv::magnitude(sobelX, sobelY, mag);

    // Normalize to 0-255 range
    cv::normalize(mag, magnitudeImg, 0, 255, cv::NORM_MINMAX);
    magnitudeImg.convertTo(magnitudeImg, CV_8U);

    return 0;
}

// Helper: Compute magnitude histogram
int computeMagnitudeHistogram(const cv::Mat& magnitude, std::vector<float>& hist,
                               int bins, float maxVal) {
    hist.assign(bins, 0.0f);

    float binSize = maxVal / bins;
    int totalPixels = magnitude.rows * magnitude.cols;

    for (int y = 0; y < magnitude.rows; y++) {
        for (int x = 0; x < magnitude.cols; x++) {
            float val = static_cast<float>(magnitude.at<uchar>(y, x));
            int bin = static_cast<int>(val / binSize);
            bin = std::min(bin, bins - 1);
            hist[bin] += 1.0f;
        }
    }

    // Normalize
    if (totalPixels > 0) {
        for (float& h : hist) {
            h /= totalPixels;
        }
    }

    return 0;
}

// Task 4: Texture + Color feature
int extractTextureColor(const cv::Mat& image, FeatureVector& feature,
                        int colorBins, int textureBins) {
    if (image.empty()) {
        return -1;
    }

    int colorBinsTotal = colorBins * colorBins * colorBins;
    feature = FeatureVector(colorBinsTotal + textureBins, FeatureType::TEXTURE_COLOR);

    // Extract color histogram
    float binSize = 256.0f / colorBins;
    int totalPixels = image.rows * image.cols;

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);

            int bBin = static_cast<int>(pixel[0] / binSize);
            int gBin = static_cast<int>(pixel[1] / binSize);
            int rBin = static_cast<int>(pixel[2] / binSize);

            bBin = std::min(bBin, colorBins - 1);
            gBin = std::min(gBin, colorBins - 1);
            rBin = std::min(rBin, colorBins - 1);

            int idx = (rBin * colorBins + gBin) * colorBins + bBin;
            feature[idx] += 1.0f;
        }
    }

    // Normalize color histogram
    for (int i = 0; i < colorBinsTotal; i++) {
        feature[i] /= totalPixels;
    }

    // Extract texture features (gradient magnitude histogram)
    cv::Mat magnitudeImg;
    computeGradientMagnitude(image, magnitudeImg);

    std::vector<float> textureHist;
    computeMagnitudeHistogram(magnitudeImg, textureHist, textureBins, 255.0f);

    // Copy texture histogram to feature vector
    for (int i = 0; i < textureBins; i++) {
        feature[colorBinsTotal + i] = textureHist[i];
    }

    return 0;
}

// Task 5: Load DNN embeddings from CSV
int extractDNNFromCSV(const std::string& csvPath, const std::string& imageName,
                      FeatureVector& feature) {
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open DNN CSV file: " << csvPath << std::endl;
        return -1;
    }

    std::string line;
    bool found = false;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;

        // First column is filename
        std::getline(ss, token, ',');

        // Check if this is the image we're looking for
        if (token.find(imageName) != std::string::npos ||
            imageName.find(token) != std::string::npos) {
            feature = FeatureVector(512, FeatureType::DNN_EMBEDDING);
            feature.imagePath = imageName;

            int idx = 0;
            while (std::getline(ss, token, ',') && idx < 512) {
                feature[idx++] = std::stof(token);
            }

            found = true;
            break;
        }
    }

    file.close();

    if (!found) {
        std::cerr << "Error: Image " << imageName << " not found in DNN CSV" << std::endl;
        return -1;
    }

    return 0;
}

// Load all DNN embeddings from CSV
int loadDNNEmbeddings(const std::string& csvPath,
                      std::vector<std::string>& imageNames,
                      std::vector<FeatureVector>& features) {
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open DNN CSV file: " << csvPath << std::endl;
        return -1;
    }

    std::string line;
    int lineCount = 0;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;

        // First column is filename
        if (!std::getline(ss, token, ',')) {
            continue;
        }

        imageNames.push_back(token);

        FeatureVector feature(512, FeatureType::DNN_EMBEDDING);
        feature.imagePath = token;

        int idx = 0;
        while (std::getline(ss, token, ',') && idx < 512) {
            try {
                feature[idx++] = std::stof(token);
            } catch (...) {
                feature[idx++] = 0.0f;
            }
        }

        features.push_back(feature);
        lineCount++;
    }

    file.close();
    std::cout << "Loaded " << lineCount << " DNN embeddings from " << csvPath << std::endl;

    return lineCount;
}

// Task 7: Custom feature - Blue Sky Detector
// Feature vector design (30 dimensions):
// 1. Blue color histogram (16 dims): HSV H-channel 100-140° (blue range)
// 2. Spatial distribution (8 dims): Blue pixel ratio in top/bottom halves, 4 bins each
// 3. Brightness features (4 dims): High brightness (>150) distribution, 2 bins per half
// 4. Sky position features (2 dims): Blue color concentration in upper region
int extractCustom(const cv::Mat& image, FeatureVector& feature) {
    if (image.empty()) {
        return -1;
    }

    // Initialize 30-dimensional feature vector
    feature = FeatureVector(30, FeatureType::CUSTOM);
    std::fill(feature.data.begin(), feature.data.end(), 0.0f);

    // Convert to HSV for better color analysis
    cv::Mat hsvImage;
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

    int rows = hsvImage.rows;
    int cols = hsvImage.cols;
    int halfRow = rows / 2;

    // Define regions: top half (sky) and bottom half
    cv::Rect topRegion(0, 0, cols, halfRow);
    cv::Rect bottomRegion(0, halfRow, cols, rows - halfRow);

    // Feature indices:
    // 0-15: Blue color histogram (16 bins for H: 100-140°)
    // 16-23: Spatial distribution (4 bins top + 4 bins bottom)
    // 24-27: Brightness features (2 bins top + 2 bins bottom)
    // 28-29: Sky position features (blue concentration in upper region)

    // Blue hue range in OpenCV HSV: 100-140 degrees (OpenCV H: 0-179 maps to 0-358)
    // In OpenCV, H value is 0-179, so blue is approximately 50-70
    const float BLUE_HUE_MIN_OPENCV = 50.0f;   // ~100 degrees
    const float BLUE_HUE_MAX_OPENCV = 70.0f;   // ~140 degrees
    const float BLUE_HUE_RANGE = BLUE_HUE_MAX_OPENCV - BLUE_HUE_MIN_OPENCV;  // 20
    const int BLUE_HIST_BINS = 16;
    const int SPATIAL_BINS = 4;
    const float BRIGHTNESS_THRESHOLD = 150.0f;  // Sky is usually bright
    const float SATURATION_MIN = 50.0f;         // Minimum saturation for sky blue

    int bluePixelCountTop = 0;
    int bluePixelCountBottom = 0;
    int totalPixelsTop = 0;
    int totalPixelsBottom = 0;
    float blueSumY = 0.0f;
    int bluePixelCount = 0;

    // Process top region (sky)
    for (int y = topRegion.y; y < topRegion.y + topRegion.height; y++) {
        for (int x = topRegion.x; x < topRegion.x + topRegion.width; x++) {
            cv::Vec3b pixel = hsvImage.at<cv::Vec3b>(y, x);
            float h = pixel[0];          // H: 0-179 (OpenCV)
            float s = pixel[1];          // S: 0-255
            float v = pixel[2];          // V: 0-255 (brightness)

            totalPixelsTop++;

            // Check if blue color (Hue in blue range, sufficient saturation, not too dark)
            if (h >= BLUE_HUE_MIN_OPENCV && h <= BLUE_HUE_MAX_OPENCV &&
                s >= SATURATION_MIN && v > 50) {

                // Map blue hue to bin (0-15)
                float normalizedHue = (h - BLUE_HUE_MIN_OPENCV) / BLUE_HUE_RANGE;
                int bin = static_cast<int>(normalizedHue * BLUE_HIST_BINS);
                bin = std::min(bin, BLUE_HIST_BINS - 1);
                feature[bin] += 1.0f;

                bluePixelCountTop++;
                blueSumY += y;
                bluePixelCount++;

                // Spatial distribution (divide top region into 4 horizontal strips)
                int spatialBin = (y * SPATIAL_BINS) / halfRow;
                spatialBin = std::min(spatialBin, SPATIAL_BINS - 1);
                feature[16 + spatialBin] += 1.0f;
            }

            // Brightness feature (sky is usually bright)
            if (v > BRIGHTNESS_THRESHOLD) {
                int brightBin = (v > 200) ? 1 : 0;  // High vs very high brightness
                feature[24 + brightBin] += 1.0f;
            }
        }
    }

    // Process bottom region
    for (int y = bottomRegion.y; y < bottomRegion.y + bottomRegion.height; y++) {
        for (int x = bottomRegion.x; x < bottomRegion.x + bottomRegion.width; x++) {
            cv::Vec3b pixel = hsvImage.at<cv::Vec3b>(y, x);
            float h = pixel[0];
            float s = pixel[1];
            float v = pixel[2];

            totalPixelsBottom++;

            // Check if blue color
            if (h >= BLUE_HUE_MIN_OPENCV && h <= BLUE_HUE_MAX_OPENCV &&
                s >= SATURATION_MIN && v > 50) {

                float normalizedHue = (h - BLUE_HUE_MIN_OPENCV) / BLUE_HUE_RANGE;
                int bin = static_cast<int>(normalizedHue * BLUE_HIST_BINS);
                bin = std::min(bin, BLUE_HIST_BINS - 1);
                feature[bin] += 1.0f;

                bluePixelCountBottom++;
                blueSumY += y;
                bluePixelCount++;

                // Spatial distribution (divide bottom region into 4 horizontal strips)
                int relativeY = y - halfRow;
                int spatialBin = (relativeY * SPATIAL_BINS) / (rows - halfRow);
                spatialBin = std::min(spatialBin, SPATIAL_BINS - 1);
                feature[20 + spatialBin] += 1.0f;
            }

            // Brightness feature
            if (v > BRIGHTNESS_THRESHOLD) {
                int brightBin = (v > 200) ? 1 : 0;
                feature[26 + brightBin] += 1.0f;
            }
        }
    }

    // Normalize blue color histogram (indices 0-15)
    int totalBluePixels = bluePixelCountTop + bluePixelCountBottom;
    if (totalBluePixels > 0) {
        for (int i = 0; i < BLUE_HIST_BINS; i++) {
            feature[i] /= totalBluePixels;
        }
    }

    // Normalize spatial distribution (indices 16-23)
    if (totalPixelsTop > 0) {
        for (int i = 0; i < SPATIAL_BINS; i++) {
            feature[16 + i] /= totalPixelsTop;
        }
    }
    if (totalPixelsBottom > 0) {
        for (int i = 0; i < SPATIAL_BINS; i++) {
            feature[20 + i] /= totalPixelsBottom;
        }
    }

    // Normalize brightness features (indices 24-27)
    if (totalPixelsTop > 0) {
        feature[24] /= totalPixelsTop;
        feature[25] /= totalPixelsTop;
    }
    if (totalPixelsBottom > 0) {
        feature[26] /= totalPixelsBottom;
        feature[27] /= totalPixelsBottom;
    }

    // Sky position features (indices 28-29)
    // Feature 28: Ratio of blue pixels in top half vs total blue pixels
    // For blue sky images, most blue should be in the top half
    if (totalBluePixels > 0) {
        feature[28] = static_cast<float>(bluePixelCountTop) / totalBluePixels;
    }

    // Feature 29: Normalized average Y position of blue pixels
    // (0 = top, 1 = bottom) - sky blue should have low values
    if (bluePixelCount > 0) {
        feature[29] = (blueSumY / bluePixelCount) / rows;
    }

    return 0;
}

// Generic feature extraction dispatcher
int extractFeature(const cv::Mat& image, FeatureVector& feature, FeatureType type) {
    switch (type) {
        case FeatureType::BASELINE:
            return extractBaseline(image, feature);
        case FeatureType::HISTOGRAM:
            return extractHistogram(image, feature, 16);
        case FeatureType::MULTI_HISTOGRAM:
            return extractMultiHistogram(image, feature, 8, true);
        case FeatureType::TEXTURE_COLOR:
            return extractTextureColor(image, feature, 8, 8);
        case FeatureType::CUSTOM:
            return extractCustom(image, feature);
        default:
            std::cerr << "Error: Unknown feature type" << std::endl;
            return -1;
    }
}
