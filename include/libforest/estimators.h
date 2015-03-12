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

// For gaussian sampling
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>

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
                cachedDeterminant(false),
                randN(rng, norm) {};
        
        /**
         * Gaussian with given mean and covariance.
         */
        Gaussian(Eigen::VectorXf _mean, Eigen::MatrixXf _covariance);
        
        /**
         * Gaussian with given mean and covariance and cached determinant.
         */
        Gaussian(Eigen::VectorXf _mean, Eigen::MatrixXf _covariance, float _covarianceDeterminant);
                
        /**
         * Destructor.
         */
        ~Gaussian() {};
        
        Gaussian operator=(const Gaussian & other)
        {
//            mean = other.mean;
            covariance = other.covariance;

            transform = other.transform;
            
            cachedInverse = false;
            cachedDeterminant = false;
            
            return *this;
        }
        
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
         * Uniform pseudo random generator.
         */
        boost::mt19937 rng;
        /**
         * Scalar Gaussian distribution.
         */
        boost::normal_distribution<float> norm;
        /**
         * Zero mean and unit variance Gaussian distribution.
         */
        boost::variate_generator<boost::mt19937&, boost::normal_distribution<float> > randN;
        /**
         * Eigenvector and eigenvalue transformation for sampling.
         */
        Eigen::MatrixXf transform;
    };
    
    /**
     * Density decision tree for unsupervised learning.
     */
    class DensityTree : public Tree, public Estimator {
    public:
        /**
         * Creates an empty density tree.
         */
        DensityTree() : Tree() {};
        
        /**
         * Destructor.
         */
        ~DensityTree() {};
        
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
         * The Gaussians at the leafs.
         */
        std::vector<Gaussian> gaussians;
        
    };
}
#endif