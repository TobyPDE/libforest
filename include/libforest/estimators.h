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
    
    /**
     * An estimator can be used to estimate the probability of a data point.
     */
    class Estimator {
    public:
        /**
         * Estimate the probability of a datapoint.
         */
        virtual float estimate(const DataPoint* x) = 0;
        
    };
    
    /**
     * A generator can be used to sample from a distribution.
     */
    class Generator {
    public:
        /**
         * Sample from the estimator.
         */
        virtual DataPoint* sample() = 0;
        
    };
    
    /**
     * Kernel interface.
     */
    class Kernel {
    public:
        Kernel() {};
        virtual ~Kernel() {};
        
        /**
         * Evaluate the kernel.
         */
        virtual float evaluate(DataPoint* x) = 0;
    };
    
    /**
     * Simple Gaussian kernel for density estimation.
     */
    class GaussianKernel : public Kernel {
    public:
        /**
         * Evaluate kernel for a given datapoint.
         */
        virtual float evaluate(DataPoint* x)
        {
            const int D = x->getDimensionality();
            
            float inner = 0;
            for (int i = 0; i < D; i++)
            {
                inner += x->at(i)*x->at(i);
            }
            
            return std::sqrt(std::pow(2*M_PI, D)) * std::exp(- 1./2. * inner);
        }
    };
    
    /**
     * Simple kernel density estimation.
     */
    class KernelDensityEstimator : public Estimator {
    public:
        
        /**
         * E.g. see: 
         * 
         *  B. A. Turlach.
         *  Bandwidth Selection in Kernel Density Estimation: A Review.
         *  CORE and Institut de Statistique.
         */
        static const int BANDWIDTH_RULE_OF_THUMB = 0;
        static const int BANDWIDTH_RULE_OF_THUMB_INTERQUARTILE = 1;
        /**
         * Constructs a kernel density estimator given the data with default
         * Gaussian kernel.
         */
        KernelDensityEstimator(UnlabeledDataStorage* _storage) : 
                kernel(new GaussianKernel()),
                bandwidthSelectionMethod(BANDWIDTH_RULE_OF_THUMB)
        {
            storage = _storage;
        }
        
        /**
         * Create a kernel density estimator with the given data and kernel.
         */
        KernelDensityEstimator(UnlabeledDataStorage* _storage, Kernel* _kernel) : 
                kernel(_kernel),
                bandwidthSelectionMethod(BANDWIDTH_RULE_OF_THUMB)
        {
            storage = _storage;
        }
        
        /**
         * Sets bandwidth selection method.
         */
        void setBandwidthSelectionMethod(int _bandwidthSelectionMethod)
        {
            switch (_bandwidthSelectionMethod)
            {
                case BANDWIDTH_RULE_OF_THUMB:
                    bandwidthSelectionMethod = _bandwidthSelectionMethod;
                    break;
                default:
                    bandwidthSelectionMethod = BANDWIDTH_RULE_OF_THUMB;
                    break;
            }
        }
        
        /**
         * Returns the bandwidth selection method.
         */
        int getBandwidthSelectionMethod()
        {
            return bandwidthSelectionMethod;
        }
        
        /**
         * Destructor.
         */
        virtual ~KernelDensityEstimator() {};
        
        /**
         * Estimate the probability of a datapoint.
         */
        virtual float estimate(const DataPoint* x) = 0;
        
    private:
        /**
         * Calculate the variance of the datainoen dimension.
         */
        float calculateVariance(int d);
        
        /**
         * Select the bandwidth using the naive Gaussian method, see:
         * 
         *  R. J. Hyndman, X Zhang, M. L. King.
         *  Bandwidth Selection for Multivariate Kernel Density Estimation Using MCMC.
         *  Econometric Society Australasian Meetings, 2004.
         */
        Eigen::VectorXf selectBandwidthRuleOfThumb();
        
        /**
         * Used kernel.
         */
        Kernel* kernel;
        
        /**
         * Bandwidth selection method.
         */
        int bandwidthSelectionMethod;
        /**
         * Data storage to base estimation on.
         */
        UnlabeledDataStorage* storage;
        
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
                cachedDeterminant(false),
                dataSupport(0) {};
        
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
        
        /**
         * Returns the number of samples used to estimate this gaussian.
         */
        int getDataSupport()
        {
            return dataSupport;
        }
        
        /**
         * Set number of samples used to estimate this gaussian.
         */
        void setDataSupport(int _dataSupport)
        {
            dataSupport = _dataSupport;
        }
        
    private:
        /**
         * Get a data point as vector.
         */
        Eigen::VectorXf asEigenVector(const DataPoint* x);
        
        /**
         * A vector as data point.
         */
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
        /**
         * Number of data points used for this gaussian.
         */
        int dataSupport;
        
    };
    
    /**
     * Density decision tree for unsupervised learning.
     */
    class DensityTree : public Estimator, public Generator {
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
         * Compute and cache the partition function.
         */
        float getPartitionFunction(int D);
        
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
        /**
         * Partition function.
         */
        float partitionFunction;
        /**
         * The partition function is cached.
         */
        bool cachedPartitionFunction;
        
    };
    
    /**
     * A density forest consisting of several density trees.
     */
    class DensityForest : public Estimator {
    public:
        /**
         * Constructor.
         */
        DensityForest() {}
        
        /**
         * Destructor.
         */
        virtual ~DensityForest() {};
        
        /**
         * Adds a tree to the ensemble
         */
        void addTree(DensityTree* tree)
        {
            trees.push_back(tree);
        }
        
        /**
         * Returns the number of trees
         */
        int getSize() const
        {
            return static_cast<int>(trees.size());
        }
        
        /**
         * Returns the i-th tree
         */
        DensityTree* getTree(int i)
        {
            return trees[i];
        }
        
        /**
         * Removes the i-th tree
         */
        void removeTree(int i)
        {
            // Delete the tree
            delete trees[i];
            // Remove it from the array
            trees.erase(trees.begin() + i);
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
         * The individual decision trees. 
         */
        std::vector<DensityTree*> trees;
    };
}
#endif