/*
  Name: Borui Chen
  Date: 2026-02-03
  Purpose: Build feature database for CBIR system.
  Usage: ./cbir_build -d <image_dir> -f <feature_type> -o <output.csv> [-c <dnn_csv>]
*/

#include "cbir.h"
#include "feature.h"
#include <iostream>
#include <cstring>
#include <cstdlib>

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " -d <image_dir> -f <feature_type> -o <output.csv> [-c <dnn_csv>]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -d <image_dir>     Directory containing images" << std::endl;
    std::cout << "  -f <feature_type>  Feature type:" << std::endl;
    std::cout << "                       baseline        - 7x7 center square (Task 1)" << std::endl;
    std::cout << "                       histogram       - Color histogram (Task 2)" << std::endl;
    std::cout << "                       multi_histogram - Multi-region histogram (Task 3)" << std::endl;
    std::cout << "                       texture_color   - Texture + Color (Task 4)" << std::endl;
    std::cout << "                       dnn_embedding   - ResNet18 embeddings (Task 5)" << std::endl;
    std::cout << "                       custom          - Custom features (Task 7)" << std::endl;
    std::cout << "  -o <output.csv>    Output feature database file" << std::endl;
    std::cout << "  -c <dnn_csv>       Path to DNN embeddings CSV (required for dnn_embedding)" << std::endl;
    std::cout << "  -h                 Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << programName << " -d data/olympus -f baseline -o features_baseline.csv" << std::endl;
    std::cout << "  " << programName << " -d data/olympus -f histogram -o features_hist.csv" << std::endl;
    std::cout << "  " << programName << " -d data/olympus -f dnn_embedding -c resnet18_features.csv -o features_dnn.csv" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string imageDir;
    std::string featureTypeStr;
    std::string outputFile;
    std::string dnnCsvPath;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            imageDir = argv[++i];
        } else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            featureTypeStr = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            outputFile = argv[++i];
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
    if (imageDir.empty() || featureTypeStr.empty() || outputFile.empty()) {
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

    std::cout << "CBIR Build Tool" << std::endl;
    std::cout << "===============" << std::endl;
    std::cout << "Image directory: " << imageDir << std::endl;
    std::cout << "Feature type: " << featureTypeToString(featureType) << std::endl;
    std::cout << "Output file: " << outputFile << std::endl;
    if (!dnnCsvPath.empty()) {
        std::cout << "DNN CSV: " << dnnCsvPath << std::endl;
    }
    std::cout << std::endl;

    // Create CBIR system
    CBIRSystem cbir;

    if (!dnnCsvPath.empty()) {
        cbir.setDNNCsvPath(dnnCsvPath);
    }

    // Build database
    int count = cbir.buildDatabase(imageDir, featureType);
    if (count < 0) {
        std::cerr << "Error: Failed to build database" << std::endl;
        return -1;
    }

    // Save features
    if (cbir.saveFeatures(outputFile) != 0) {
        std::cerr << "Error: Failed to save features" << std::endl;
        return -1;
    }

    std::cout << std::endl;
    std::cout << "Successfully built feature database with " << count << " images" << std::endl;

    return 0;
}
