#ifndef LIBF_ESTIMATORS_H
#define LIBF_ESTIMATORS_H

/**
 * This file contains the data structures for the classifiers. There are 
 * basically two kinds ot classifiers:
 * 1. Decision trees
 * 2. Random forests
 */

#include <iostream>
#include <vector>
#include <Eigen/Dense>

#include "util.h"
#include "data.h"
#include "classifiers.h"

namespace libf {
    /**
     * Forward declarations to speed up compiling
     */
    class DataStorage;
    class DataPoint;
    
    class Estimator {
    public:
        /**
         * Estimate the probability of a datapoint.
         */
        virtual float estimate(const DataPoint* x) = 0;
        
        /**
         * Sample from the estimator.
         */
        virtual DataPoint* sample() = 0;
    };
    
    /**
     * A simple Gaussian distribution represented by mean and covariance matrix.
     */
    class Gaussian {
    public:
        /**
         * Default Gaussian with zero mean and identity covariance.
         */
        Gaussian() :
                cachedInverse(false),
                cachedDeterminant(false) {};
        
        /**
         * Gaussian with given mean and covariance.
         */
        Gaussian(Eigen::VectorXf _mean, Eigen::MatrixXf _covariance);
                
        /**
         * Destructor.
         */
        ~Gaussian() {};
        
//        Gaussian operator=(const Gaussian & other)
//        {
//            mean = other.mean;
//            covariance = other.covariance;
//            
//            transform = other.transform;
//            
//            cachedInverse = false;
//            cachedDeterminant = false;
//            
//            return *this;
//        }
        
        /**
         * Get probability of the given data point.
         */
        float evaluate(const DataPoint* x);
        
        /**
         * Sample a point from the Gaussian.
         */
        DataPoint* sample();
        
        /**
         * Get dimensionality of gaussian.
         */
        int getDimensionality()
        {
            return mean.rows();
        }
        
        /**
         * Sets the mean.
         */
        void setMean(const Eigen::VectorXf _mean)
        {
            mean = _mean;
        }
        
        /**
         * Returns the mean.
         */
        Eigen::VectorXf & getMean()
        {
            return mean;
        }
        
        /**
         * Sets the covariance matrix.
         */
        void setCovariance(const Eigen::MatrixXf _covariance);
        
        /**
         * Returns the covariance matrix.
         */
        Eigen::MatrixXf getCovariance()
        {
            return covariance;
        }
        
    private:
        Eigen::VectorXf asEigenVector(const DataPoint* x);
        DataPoint* asDataPoint(const Eigen::VectorXf & y);
        
        /**
         * Mean of Gaussian.
         */
        Eigen::VectorXf mean;
        /**
         * Covariance of Gaussian.
         */
        Eigen::MatrixXf covariance;
        /**
         * Inverse covariance can be cached.
         */
        bool cachedInverse = false;
        /**
         * Covariance determinant can be cached.
         */
        bool cachedDeterminant = false;
        /**
         * Cached covariance inverse.
         */
        Eigen::MatrixXf covarianceInverse;
        /**
         * Cached determinant.
         */
        float covarianceDeterminant;
        /**
         * Eigenvector and eigenvalue transformation for sampling.
         */
        Eigen::MatrixXf transform;
    };
    
    /**
     * Density decision tree for unsupervised learning.
     */
    class DensityTree :public Estimator {
    public:
        
        /**
         * Creates an empty density tree.
         */
        DensityTree();
        
        /**
         * Destructor.
         */
        virtual ~DensityTree() {};
        
        /**
         * Splits a child node and returns the index of the left child. 
         */
        int splitNode(int node);
        
        /**
         * Returns the leaf node for a specific data point
         */
        int findLeafNode(const DataPoint* x) const;
        
        /**
         * Sets the split feature for a node
         */
        void setSplitFeature(int node, int feature)
        {
            splitFeatures[node] = feature;
        }
        
        /**
         * Returns the split feature for a node
         */
        int getSplitFeature(int node) const
        {
            return splitFeatures[node];
        }
        
        /**
         * Sets the threshold for a node
         */
        void setThreshold(int node, float threshold)
        {
            thresholds[node] = threshold;
        }
        
        /**
         * Returns the threshold for a node
         */
        float getThreshold(int node) const
        {
            return thresholds[node];
        }
        
        /**
         * Returns the total number of nodes
         */
        int getNumNodes() const
        {
            return static_cast<int>(leftChild.size());
        }
        
        /**
         * Returns true if the given node is a leaf node
         */
        bool isLeafNode(int node) const 
        {
            assert(node >= 0 && node <= static_cast<int>(leftChild.size()));
            return leftChild[node] == 0;
        }
        
        /**
         * Returns the left child for a node
         */
        int getLeftChild(int node) const
        {
            assert(node >= 0 && node <= static_cast<int>(leftChild.size()));
            return leftChild[node];
        }
        
        /**
         * Get depth of a node.
         */
        int getDepth(int node)
        {
            assert(node >= 0 && node <= static_cast<int>(depths.size()));
            return depths[node];
        }
        
        /**
         * Get the Gaussian of a specific leaf.
         */
        Gaussian & getGaussian(const int node)
        {
            return gaussians[node];
        }
        
        /**
         * Estimate the probability of a datapoint.
         */
        virtual float estimate(const DataPoint* x);
        
        /**
         * Sample from the model.
         */
        virtual DataPoint* sample();
        
    private:
        /**
         * Adds a plain new node.
         */
        virtual void addNode(int depth);
        
        /**
         * The depth of each node.
         */
        std::vector<int> depths;
        /**
         * The split feature at each node. 
         */
        std::vector<int> splitFeatures;
        /**
         * The threshold at each node
         */
        std::vector<float> thresholds;
        /**
         * The left child node of each node. If the left child node is 0, then 
         * this is a leaf node. The right child node is left + 1. 
         */
        std::vector<int> leftChild;
        
        /**
         * The Gaussians at the leafs.
         */
        std::vector<Gaussian> gaussians;
        
    };
}
#endif