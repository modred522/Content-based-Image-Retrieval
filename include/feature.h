/*
  Name: Borui Chen
  Date: 2026-02-03
  Purpose: Feature extraction interface and function declarations for CBIR system.
*/

#ifndef FEATURE_H
#define FEATURE_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// Feature type enumeration for different CBIR tasks
enum class FeatureType {
    BASELINE,        // Task 1: 7x7 center square
    HISTOGRAM,       // Task 2: Color histogram
    MULTI_HISTOGRAM, // Task 3: Multi-region histogram
    TEXTURE_COLOR,   // Task 4: Texture + Color features
    DNN_EMBEDDING,   // Task 5: ResNet18 embeddings from CSV
    CUSTOM           // Task 7: Custom design (placeholder)
};

// Feature vector structure to store extracted features
struct FeatureVector {
    std::vector<float> data;
    std::string imagePath;
    FeatureType type;

    FeatureVector() : type(FeatureType::BASELINE) {}
    FeatureVector(size_t size, FeatureType t) : data(size, 0.0f), type(t) {}

    size_t size() const { return data.size(); }
    float& operator[](size_t idx) { return data[idx]; }
    const float& operator[](size_t idx) const { return data[idx]; }
    void normalize();
};

// Convert FeatureType to string
std::string featureTypeToString(FeatureType type);
FeatureType stringToFeatureType(const std::string& str);

// Task 1: Baseline feature - 7x7 center square (147 dimensions)
int extractBaseline(const cv::Mat& image, FeatureVector& feature);

// Task 2: Histogram feature - 2D rg-chromaticity or 3D RGB histogram
// binsPerChannel: number of bins per channel (default 16)
int extractHistogram(const cv::Mat& image, FeatureVector& feature, int binsPerChannel = 16);

// Task 3: Multi-histogram feature - split image into regions and compute histograms
// splitHorizontal: if true, split into top/bottom halves; else left/right
int extractMultiHistogram(const cv::Mat& image, FeatureVector& feature,
                          int binsPerChannel = 8, bool splitHorizontal = true);

// Task 4: Texture + Color feature
// Combines color histogram with gradient magnitude histogram
int extractTextureColor(const cv::Mat& image, FeatureVector& feature,
                        int colorBins = 8, int textureBins = 8);

// Task 5: DNN Embedding - read from CSV file
// Returns feature for a single image from pre-computed CSV
int extractDNNFromCSV(const std::string& csvPath, const std::string& imageName,
                      FeatureVector& feature);

// Load all DNN embeddings from CSV
int loadDNNEmbeddings(const std::string& csvPath,
                      std::vector<std::string>& imageNames,
                      std::vector<FeatureVector>& features);

// Task 7: Custom feature (placeholder)
int extractCustom(const cv::Mat& image, FeatureVector& feature);

// Generic feature extraction dispatcher
int extractFeature(const cv::Mat& image, FeatureVector& feature, FeatureType type);

// Helper functions for texture features
int computeGradientMagnitude(const cv::Mat& image, cv::Mat& magnitude);
int computeMagnitudeHistogram(const cv::Mat& magnitude, std::vector<float>& hist,
                               int bins, float maxVal = 255.0f);

#endif // FEATURE_H
