/*
  Name: Borui Chen
  Date: 2026-02-03
  Purpose: CBIR system main interface.
*/

#ifndef CBIR_H
#define CBIR_H

#include "feature.h"
#include "distance.h"
#include <vector>
#include <string>
#include <utility>
#include <map>

// Result structure for query matches
struct MatchResult {
    std::string imagePath;
    float distance;

    MatchResult() : distance(0.0f) {}
    MatchResult(const std::string& path, float dist) : imagePath(path), distance(dist) {}

    // For sorting by distance (ascending)
    bool operator<(const MatchResult& other) const {
        return distance < other.distance;
    }
};

// CBIR System class
class CBIRSystem {
private:
    std::vector<std::string> imagePaths;
    std::vector<FeatureVector> features;
    FeatureType currentFeatureType;
    std::string dnnCsvPath;  // Path to DNN embeddings CSV

    // Map for quick DNN feature lookup (filename -> feature index)
    std::map<std::string, size_t> dnnFeatureMap;

public:
    CBIRSystem();
    ~CBIRSystem();

    // Set DNN CSV path for Task 5
    void setDNNCsvPath(const std::string& path);

    // Build feature database from image directory
    // Returns number of images processed, or -1 on error
    int buildDatabase(const std::string& imageDir, FeatureType type);

    // Save features to CSV file
    int saveFeatures(const std::string& filename);

    // Load features from CSV file
    int loadFeatures(const std::string& filename);

    // Query for similar images
    // Returns top N matches sorted by distance
    std::vector<MatchResult> query(const std::string& targetImage, int topN);

    // Query using pre-computed feature vector
    std::vector<MatchResult> query(const FeatureVector& targetFeature, int topN);

    // Getters
    size_t getDatabaseSize() const { return features.size(); }
    FeatureType getFeatureType() const { return currentFeatureType; }
    const std::vector<std::string>& getImagePaths() const { return imagePaths; }

    // Clear database
    void clear();

private:
    // Helper to get filename from path
    std::string getFilename(const std::string& path);

    // Helper to check if file is an image
    bool isImageFile(const std::string& filename);
};

#endif // CBIR_H
