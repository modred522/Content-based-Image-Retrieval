/*
  Name: Borui Chen
  Date: 2026-02-03
  Purpose: Distance metric implementations for CBIR system.
*/

#include "distance.h"
#include <cmath>
#include <algorithm>

// Task 1: Sum of Squared Difference (SSD)
// d = sum((a[i] - b[i])^2)
float sumSquaredDifference(const FeatureVector& a, const FeatureVector& b) {
    if (a.size() != b.size()) {
        return -1.0f;
    }

    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// Task 2,3: Histogram Intersection (similarity)
// similarity = sum(min(a[i], b[i]))
float histogramIntersection(const FeatureVector& a, const FeatureVector& b) {
    if (a.size() != b.size()) {
        return -1.0f;
    }

    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        sum += std::min(a[i], b[i]);
    }
    return sum;
}

// Histogram Intersection Distance
// Returns negative similarity (so lower is better, like distance)
float histogramIntersectionDistance(const FeatureVector& a, const FeatureVector& b) {
    float intersection = histogramIntersection(a, b);
    // Return negative intersection so that smaller values are better matches
    return -intersection;
}

// Task 5: Cosine Similarity
// cos_theta = dot(a, b) / (||a|| * ||b||)
float cosineSimilarity(const FeatureVector& a, const FeatureVector& b) {
    if (a.size() != b.size()) {
        return -1.0f;
    }

    float dotProduct = 0.0f;
    float normA = 0.0f;
    float normB = 0.0f;

    for (size_t i = 0; i < a.size(); i++) {
        dotProduct += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    normA = std::sqrt(normA);
    normB = std::sqrt(normB);

    if (normA == 0.0f || normB == 0.0f) {
        return 0.0f;
    }

    return dotProduct / (normA * normB);
}

// Task 5: Cosine Distance
// distance = 1 - cos_theta
float cosineDistance(const FeatureVector& a, const FeatureVector& b) {
    return 1.0f - cosineSimilarity(a, b);
}

// Weighted distance for combining multiple features
float weightedDistance(const std::vector<float>& distances,
                       const std::vector<float>& weights) {
    if (distances.size() != weights.size()) {
        return -1.0f;
    }

    float sum = 0.0f;
    float weightSum = 0.0f;

    for (size_t i = 0; i < distances.size(); i++) {
        sum += distances[i] * weights[i];
        weightSum += weights[i];
    }

    if (weightSum > 0) {
        return sum / weightSum;
    }
    return sum;
}

// L1 Distance (Manhattan)
float l1Distance(const FeatureVector& a, const FeatureVector& b) {
    if (a.size() != b.size()) {
        return -1.0f;
    }

    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        sum += std::abs(a[i] - b[i]);
    }
    return sum;
}

// L2 Distance (Euclidean)
float l2Distance(const FeatureVector& a, const FeatureVector& b) {
    if (a.size() != b.size()) {
        return -1.0f;
    }

    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Generic distance function dispatcher based on feature type
float computeDistance(const FeatureVector& a, const FeatureVector& b, FeatureType type) {
    switch (type) {
        case FeatureType::BASELINE:
            // Use SSD for baseline
            return sumSquaredDifference(a, b);

        case FeatureType::HISTOGRAM:
            // Use histogram intersection distance
            return histogramIntersectionDistance(a, b);

        case FeatureType::MULTI_HISTOGRAM: {
            // Split features into two halves and compute weighted intersection
            size_t halfSize = a.size() / 2;
            FeatureVector region1_a(halfSize, FeatureType::HISTOGRAM);
            FeatureVector region1_b(halfSize, FeatureType::HISTOGRAM);
            FeatureVector region2_a(halfSize, FeatureType::HISTOGRAM);
            FeatureVector region2_b(halfSize, FeatureType::HISTOGRAM);

            for (size_t i = 0; i < halfSize; i++) {
                region1_a[i] = a[i];
                region1_b[i] = b[i];
                region2_a[i] = a[i + halfSize];
                region2_b[i] = b[i + halfSize];
            }

            float dist1 = histogramIntersectionDistance(region1_a, region1_b);
            float dist2 = histogramIntersectionDistance(region2_a, region2_b);

            // Equal weight for both regions
            return (dist1 + dist2) / 2.0f;
        }

        case FeatureType::TEXTURE_COLOR: {
            // Split into color and texture parts
            // Assuming 8x8x8 = 512 color bins and 8 texture bins
            size_t colorBins = 512;  // 8*8*8
            size_t textureBins = a.size() - colorBins;

            FeatureVector color_a(colorBins, FeatureType::HISTOGRAM);
            FeatureVector color_b(colorBins, FeatureType::HISTOGRAM);
            FeatureVector texture_a(textureBins, FeatureType::HISTOGRAM);
            FeatureVector texture_b(textureBins, FeatureType::HISTOGRAM);

            for (size_t i = 0; i < colorBins; i++) {
                color_a[i] = a[i];
                color_b[i] = b[i];
            }
            for (size_t i = 0; i < textureBins; i++) {
                texture_a[i] = a[i + colorBins];
                texture_b[i] = b[i + colorBins];
            }

            float colorDist = histogramIntersectionDistance(color_a, color_b);
            float textureDist = histogramIntersectionDistance(texture_a, texture_b);

            // Equal weight for color and texture
            return (colorDist + textureDist) / 2.0f;
        }

        case FeatureType::DNN_EMBEDDING:
            // Use cosine distance for DNN embeddings
            return cosineDistance(a, b);

        case FeatureType::CUSTOM: {
            // Blue Sky detector distance: weighted combination of different feature components
            // Feature layout: 0-15 blue hist, 16-23 spatial, 24-27 brightness, 28-29 sky position

            // Blue color histogram distance (histogram intersection)
            FeatureVector blueA(16, FeatureType::HISTOGRAM);
            FeatureVector blueB(16, FeatureType::HISTOGRAM);
            for (int i = 0; i < 16; i++) {
                blueA[i] = a[i];
                blueB[i] = b[i];
            }
            float blueDist = histogramIntersectionDistance(blueA, blueB);

            // Spatial distribution distance (weighted SSD, much higher weight for top region)
            // For blue sky, we expect most blue to be in the top half
            float spatialDist = 0.0f;
            float spatialWeightSum = 0.0f;
            for (int i = 16; i < 24; i++) {
                // Much higher weight for top region (bins 16-19) since sky is usually at top
                float weight = (i < 20) ? 3.0f : 1.0f;
                float diff = a[i] - b[i];
                spatialDist += weight * diff * diff;
                spatialWeightSum += weight;
            }
            spatialDist /= spatialWeightSum;

            // Brightness distance (histogram intersection)
            // Sky is usually bright
            FeatureVector brightA(4, FeatureType::HISTOGRAM);
            FeatureVector brightB(4, FeatureType::HISTOGRAM);
            for (int i = 0; i < 4; i++) {
                brightA[i] = a[24 + i];
                brightB[i] = b[24 + i];
            }
            float brightDist = histogramIntersectionDistance(brightA, brightB);

            // Sky position distance (absolute difference)
            // Feature 28: blue ratio in top half, Feature 29: average Y position of blue
            float skyPosDist = 0.0f;
            for (int i = 28; i < 30; i++) {
                skyPosDist += std::abs(a[i] - b[i]);
            }
            skyPosDist /= 2.0f;

            // Weighted combination (blue color and sky position are most important for blue sky)
            // Weights: blue 0.35, spatial 0.25, brightness 0.2, sky position 0.2
            return 0.35f * blueDist + 0.25f * spatialDist + 0.2f * brightDist + 0.2f * skyPosDist;
        }

        default:
            return sumSquaredDifference(a, b);
    }
}
