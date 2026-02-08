/*
  Name: Borui Chen
  Date: 2026-02-03
  Purpose: Distance metric functions for CBIR system.
*/

#ifndef DISTANCE_H
#define DISTANCE_H

#include "feature.h"
#include <vector>
#include <cmath>
#include <algorithm>

// Task 1: Sum of Squared Difference (SSD)
// d = sum((a[i] - b[i])^2)
float sumSquaredDifference(const FeatureVector& a, const FeatureVector& b);

// Task 2,3: Histogram Intersection
// similarity = sum(min(a[i], b[i]))
// distance = 1 - similarity (if normalized) or -similarity (if not)
float histogramIntersection(const FeatureVector& a, const FeatureVector& b);
float histogramIntersectionDistance(const FeatureVector& a, const FeatureVector& b);

// Task 5: Cosine Distance
// cos_theta = dot(a_norm, b_norm)
// distance = 1 - cos_theta
float cosineDistance(const FeatureVector& a, const FeatureVector& b);
float cosineSimilarity(const FeatureVector& a, const FeatureVector& b);

// Weighted distance for combining multiple features
// Used in Task 3, 4 for multi-feature matching
float weightedDistance(const std::vector<float>& distances,
                       const std::vector<float>& weights);

// L1 Distance (Manhattan)
float l1Distance(const FeatureVector& a, const FeatureVector& b);

// L2 Distance (Euclidean)
float l2Distance(const FeatureVector& a, const FeatureVector& b);

// Generic distance function dispatcher based on feature type
float computeDistance(const FeatureVector& a, const FeatureVector& b, FeatureType type);

#endif // DISTANCE_H
