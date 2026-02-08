/*
  Name: Borui Chen
  Date: 2026-02-03
  Purpose: CBIR system core implementation.
*/

#include "cbir.h"
#include <dirent.h>
#include <sys/stat.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

CBIRSystem::CBIRSystem() : currentFeatureType(FeatureType::BASELINE) {}

CBIRSystem::~CBIRSystem() {}

void CBIRSystem::setDNNCsvPath(const std::string& path) {
    dnnCsvPath = path;
}

std::string CBIRSystem::getFilename(const std::string& path) {
    size_t lastSlash = path.find_last_of("/\\");
    if (lastSlash != std::string::npos) {
        return path.substr(lastSlash + 1);
    }
    return path;
}

bool CBIRSystem::isImageFile(const std::string& filename) {
    std::string lower = filename;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    return (lower.find(".jpg") != std::string::npos ||
            lower.find(".jpeg") != std::string::npos ||
            lower.find(".png") != std::string::npos ||
            lower.find(".ppm") != std::string::npos ||
            lower.find(".tif") != std::string::npos ||
            lower.find(".tiff") != std::string::npos ||
            lower.find(".bmp") != std::string::npos);
}

int CBIRSystem::buildDatabase(const std::string& imageDir, FeatureType type) {
    currentFeatureType = type;
    imagePaths.clear();
    features.clear();
    dnnFeatureMap.clear();

    // Special handling for DNN embeddings
    if (type == FeatureType::DNN_EMBEDDING) {
        if (dnnCsvPath.empty()) {
            std::cerr << "Error: DNN CSV path not set. Use setDNNCsvPath() first." << std::endl;
            return -1;
        }

        // Load all DNN embeddings
        int count = loadDNNEmbeddings(dnnCsvPath, imagePaths, features);
        if (count < 0) {
            return -1;
        }

        // Build lookup map
        for (size_t i = 0; i < imagePaths.size(); i++) {
            dnnFeatureMap[imagePaths[i]] = i;
        }

        return count;
    }

    // Open directory
    DIR* dirp = opendir(imageDir.c_str());
    if (dirp == nullptr) {
        std::cerr << "Error: Cannot open directory " << imageDir << std::endl;
        return -1;
    }

    struct dirent* dp;
    int count = 0;

    while ((dp = readdir(dirp)) != nullptr) {
        std::string filename = dp->d_name;

        // Check if it's an image file
        if (!isImageFile(filename)) {
            continue;
        }

        // Build full path
        std::string fullPath = imageDir + "/" + filename;

        // Load image
        cv::Mat image = cv::imread(fullPath);
        if (image.empty()) {
            std::cerr << "Warning: Cannot load image " << fullPath << std::endl;
            continue;
        }

        // Extract feature
        FeatureVector feature;
        if (extractFeature(image, feature, type) != 0) {
            std::cerr << "Warning: Failed to extract feature from " << fullPath << std::endl;
            continue;
        }

        feature.imagePath = fullPath;
        imagePaths.push_back(fullPath);
        features.push_back(feature);
        count++;

        if (count % 100 == 0) {
            std::cout << "Processed " << count << " images..." << std::endl;
        }
    }

    closedir(dirp);

    std::cout << "Built database with " << count << " images" << std::endl;
    return count;
}

int CBIRSystem::saveFeatures(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << filename << std::endl;
        return -1;
    }

    // Write header with feature type
    file << "# CBIR Feature Database\n";
    file << "# Feature Type: " << featureTypeToString(currentFeatureType) << "\n";
    file << "# Feature Dimension: " << (features.empty() ? 0 : features[0].size()) << "\n";
    file << "# Number of Images: " << features.size() << "\n";

    // Write features
    for (size_t i = 0; i < features.size(); i++) {
        file << getFilename(imagePaths[i]);
        for (size_t j = 0; j < features[i].size(); j++) {
            file << "," << features[i][j];
        }
        file << "\n";
    }

    file.close();
    std::cout << "Saved " << features.size() << " features to " << filename << std::endl;
    return 0;
}

int CBIRSystem::loadFeatures(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for reading: " << filename << std::endl;
        return -1;
    }

    imagePaths.clear();
    features.clear();

    std::string line;
    int lineCount = 0;
    int featureDim = -1;

    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') {
            // Parse header info
            if (line.find("Feature Type:") != std::string::npos) {
                size_t pos = line.find(":");
                if (pos != std::string::npos) {
                    std::string typeStr = line.substr(pos + 2);
                    currentFeatureType = stringToFeatureType(typeStr);
                }
            }
            if (line.find("Feature Dimension:") != std::string::npos) {
                size_t pos = line.find(":");
                if (pos != std::string::npos) {
                    featureDim = std::stoi(line.substr(pos + 2));
                }
            }
            continue;
        }

        std::stringstream ss(line);
        std::string token;

        // First column is filename
        if (!std::getline(ss, token, ',')) {
            continue;
        }

        imagePaths.push_back(token);

        // Read feature values
        FeatureVector feature;
        feature.imagePath = token;
        feature.type = currentFeatureType;

        while (std::getline(ss, token, ',')) {
            try {
                feature.data.push_back(std::stof(token));
            } catch (...) {
                feature.data.push_back(0.0f);
            }
        }

        if (featureDim < 0) {
            featureDim = feature.size();
        }

        features.push_back(feature);
        lineCount++;
    }

    file.close();
    std::cout << "Loaded " << lineCount << " features from " << filename << std::endl;
    return lineCount;
}

std::vector<MatchResult> CBIRSystem::query(const std::string& targetImage, int topN) {
    std::vector<MatchResult> results;

    // Load target image
    cv::Mat image = cv::imread(targetImage);
    if (image.empty()) {
        std::cerr << "Error: Cannot load target image " << targetImage << std::endl;
        return results;
    }

    // Extract feature
    FeatureVector targetFeature;

    // Special handling for DNN embeddings
    if (currentFeatureType == FeatureType::DNN_EMBEDDING) {
        std::string filename = getFilename(targetImage);

        // Check if we have this image in our DNN database
        auto it = dnnFeatureMap.find(filename);
        if (it != dnnFeatureMap.end()) {
            targetFeature = features[it->second];
        } else {
            // Try to extract from CSV
            if (extractDNNFromCSV(dnnCsvPath, filename, targetFeature) != 0) {
                std::cerr << "Error: Target image not found in DNN database" << std::endl;
                return results;
            }
        }
    } else {
        if (extractFeature(image, targetFeature, currentFeatureType) != 0) {
            std::cerr << "Error: Failed to extract feature from target image" << std::endl;
            return results;
        }
    }

    targetFeature.imagePath = targetImage;
    targetFeature.type = currentFeatureType;

    return query(targetFeature, topN);
}

std::vector<MatchResult> CBIRSystem::query(const FeatureVector& targetFeature, int topN) {
    std::vector<MatchResult> results;

    if (features.empty()) {
        std::cerr << "Error: Database is empty" << std::endl;
        return results;
    }

    // Compute distances to all images
    for (size_t i = 0; i < features.size(); i++) {
        float dist = computeDistance(targetFeature, features[i], currentFeatureType);
        results.push_back(MatchResult(imagePaths[i], dist));
    }

    // Sort by distance (ascending)
    std::sort(results.begin(), results.end());

    // Return top N
    if (results.size() > static_cast<size_t>(topN)) {
        results.resize(topN);
    }

    return results;
}

void CBIRSystem::clear() {
    imagePaths.clear();
    features.clear();
    dnnFeatureMap.clear();
}
