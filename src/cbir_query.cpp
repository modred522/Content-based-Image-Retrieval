/*
  Name: Borui Chen
  Date: 2026-02-03
  Purpose: Query similar images from CBIR database.
  Usage: ./cbir_query -t <target_image> -f <feature_type> -i <features.csv> -n <num_results> [-c <dnn_csv>]
*/

#include "cbir.h"
#include "feature.h"
#include <iostream>
#include <cstring>
#include <cstdlib>

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " -t <target_image> -f <feature_type> -i <features.csv> -n <num_results> [-c <dnn_csv>]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -t <target_image>   Target image to query" << std::endl;
    std::cout << "  -f <feature_type>   Feature type:" << std::endl;
    std::cout << "                        baseline        - 7x7 center square (Task 1)" << std::endl;
    std::cout << "                        histogram       - Color histogram (Task 2)" << std::endl;
    std::cout << "                        multi_histogram - Multi-region histogram (Task 3)" << std::endl;
    std::cout << "                        texture_color   - Texture + Color (Task 4)" << std::endl;
    std::cout << "                        dnn_embedding   - ResNet18 embeddings (Task 5)" << std::endl;
    std::cout << "                        custom          - Custom features (Task 7)" << std::endl;
    std::cout << "  -i <features.csv>   Input feature database file" << std::endl;
    std::cout << "  -n <num_results>    Number of top matches to return" << std::endl;
    std::cout << "  -c <dnn_csv>        Path to DNN embeddings CSV (required for dnn_embedding)" << std::endl;
    std::cout << "  -h                  Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << programName << " -t data/olympus/pic.1016.jpg -f baseline -i features_baseline.csv -n 3" << std::endl;
    std::cout << "  " << programName << " -t data/olympus/pic.0164.jpg -f histogram -i features_hist.csv -n 5" << std::endl;
    std::cout << "  " << programName << " -t data/olympus/pic.0893.jpg -f dnn_embedding -i features_dnn.csv -c resnet18_features.csv -n 3" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string targetImage;
    std::string featureTypeStr;
    std::string featuresFile;
    std::string dnnCsvPath;
    int numResults = 3;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            targetImage = argv[++i];
        } else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            featureTypeStr = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            featuresFile = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numResults = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            dnnCsvPath = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << argv[i] << std::endl;
            printUsage(argv[0]);
            return -1;
        }
    }

    // Validate arguments
    if (targetImage.empty() || featureTypeStr.empty() || featuresFile.empty()) {
        std::cerr << "Error: Missing required arguments" << std::endl;
        printUsage(argv[0]);
        return -1;
    }

    // Convert feature type string to enum
    FeatureType featureType = stringToFeatureType(featureTypeStr);

    // Special check for DNN embeddings
    if (featureType == FeatureType::DNN_EMBEDDING && dnnCsvPath.empty()) {
        std::cerr << "Error: DNN embeddings require -c <dnn_csv> option" << std::endl;
        printUsage(argv[0]);
        return -1;
    }

    std::cout << "CBIR Query Tool" << std::endl;
    std::cout << "===============" << std::endl;
    std::cout << "Target image: " << targetImage << std::endl;
    std::cout << "Feature type: " << featureTypeToString(featureType) << std::endl;
    std::cout << "Features file: " << featuresFile << std::endl;
    std::cout << "Number of results: " << numResults << std::endl;
    if (!dnnCsvPath.empty()) {
        std::cout << "DNN CSV: " << dnnCsvPath << std::endl;
    }
    std::cout << std::endl;

    // Create CBIR system
    CBIRSystem cbir;

    if (!dnnCsvPath.empty()) {
        cbir.setDNNCsvPath(dnnCsvPath);
    }

    // Load feature database
    if (cbir.loadFeatures(featuresFile) <= 0) {
        std::cerr << "Error: Failed to load feature database" << std::endl;
        return -1;
    }

    // Verify feature type matches
    if (cbir.getFeatureType() != featureType) {
        std::cout << "Warning: Feature type mismatch. Database uses "
                  << featureTypeToString(cbir.getFeatureType())
                  << ", query uses " << featureTypeToString(featureType) << std::endl;
        std::cout << "Using database feature type for query." << std::endl << std::endl;
    }

    // Perform query
    std::cout << "Querying..." << std::endl;
    std::vector<MatchResult> results = cbir.query(targetImage, numResults);

    if (results.empty()) {
        std::cerr << "Error: Query returned no results" << std::endl;
        return -1;
    }

    // Print results
    std::cout << std::endl;
    std::cout << "Top " << results.size() << " matches for " << targetImage << ":" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    for (size_t i = 0; i < results.size(); i++) {
        std::cout << i + 1 << ". " << results[i].imagePath
                  << " (distance: " << results[i].distance << ")" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Query completed successfully." << std::endl;

    return 0;
}
