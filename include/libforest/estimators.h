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
     * Univariate kernel interface.
     */
    class Kernel {
    public:
        /**
         * Constructor.
         */
        Kernel() {};
        
        /**
         * Destructor.
         */
        virtual ~Kernel() {};
        
        /**
         * Evaluate the kernel in the given dimension.
         */
        virtual float evaluate(DataPoint* x, int d) = 0;
        
        /**
         * Returns the coefficient for the rule of thumb bandwidth selection
         * calculated as
         * 
         *  2 ((pi^(1/2) * (v!)^3 * R(k))/(2*v * (2*v)! kappa_v^2(k)))^(1/(2*2 + 1))
         * 
         * where v is the order (int this case v = 2), kappa_v(k) = int(x^2 k(x) dx),
         * and k is the kernel.
         */
        virtual float getRuleOfThumbCoefficient() = 0;
    };
    
    /**
     * Univariate Gaussian kernel.
     */
    class GaussianKernel : public Kernel {
    public:
        /**
         * Evaluate the kernel in the given dimension.
         */
        virtual float evaluate(DataPoint* x, int d)
        {
            assert(d >= 0 && d < x->getDimensionality());
            return std::sqrt(2*M_PI) * std::exp(- 1.f/2.f * x->at(d)*x->at(d));
        }
        
        /**
         * Get coefficient for rule of thumb bandwidth selection.
         */
        float getRuleOfThumbCoefficient()
        {
            // See table 4 in:
            //  B. E. Hansen.
            //  Lecture Notes on Nonparametrics.
            //  University of Wisconsin
            return 1.06f;
        }
    };
    
    /**
     * Univariate Epanechnikov kernel.
     */
    class EpanechnikovKernel : public Kernel {
    public:
        /**
         * Evaluate the kernel in the given dimension.
         */
        virtual float evaluate(DataPoint* x, int d)
        {
            assert(d >= 0 && d < x->getDimensionality());
            
            float K_x = 1 - x->at(d)*x->at(d);
            if (K_x > 0)
            {
                return 3.f/4.f * K_x;
            }
            else
            {
                return 0;
            }
        }
        
        /**
         * Get coefficient for rule of thumb bandwidth selection.
         */
        float getRuleOfThumbCoefficient()
        {
            // See table 4 in:
            //  B. E. Hansen.
            //  Lecture Notes on Nonparametrics.
            //  University of Wisconsin
            return 2.34f;
        }
    };
    
    /**
     * Univariate biweight kernel.
     */
    class BiweightKernel : public Kernel {
    public:
        /**
         * Evaluate the kernel in the given dimension.
         */
        virtual float evaluate(DataPoint* x, int d)
        {
            assert(d >= 0 && d < x->getDimensionality());
            
            float K_x = 1 - x->at(d)*x->at(d);
            if (K_x > 0)
            {
                return 15.f/16.f * K_x*K_x;
            }
            else
            {
                return 0;
            }
        }
        
        /**
         * Get coefficient for rule of thumb bandwidth selection.
         */
        float getRuleOfThumbCoefficient()
        {
            // See table 4 in:
            //  B. E. Hansen.
            //  Lecture Notes on Nonparametrics.
            //  University of Wisconsin
            return 2.78f;
        }
    };
    
    /**
     * Univariate triweight kernel.
     */
    class TriweightKernel : public Kernel {
    public:
        /**
         * Evaluate the kernel in the given dimension.
         */
        virtual float evaluate(DataPoint* x, int d)
        {
            assert(d >= 0 && d < x->getDimensionality());
            
            float K_x = 1 - x->at(d)*x->at(d);
            if (K_x > 0)
            {
                return 35.f/32.f * K_x*K_x*K_x;
            }
            else
            {
                return 0;
            }
        }
        
        /**
         * Get coefficient for rule of thumb bandwidth selection.
         */
        float getRuleOfThumbCoefficient()
        {
            // See table 4 in:
            //  B. E. Hansen.
            //  Lecture Notes on Nonparametrics.
            //  University of Wisconsin
            return 3.15f;
        }
    };
    
    /**
     * Multivariate kernel interface.
     */
    class MultivariateKernel {
    public:
        /**
         * Constructor.
         */
        MultivariateKernel() {};
        
        /**
         * Destructor.
         */
        virtual ~MultivariateKernel() {};
        
        /**
         * Evaluate the kernel.
         */
        virtual float evaluate(DataPoint* x) = 0;
        
        /**
         * Returns the coefficient for the rule of thumb bandwidth selection
         * calculated as
         * 
         *  2 ((pi^(1/2) * (v!)^3 * R(k))/(2*v * (2*v)! kappa_v^2(k)))^(1/(2*2 + 1))
         * 
         * where v is the order (int this case v = 2), kappa_v(k) = int(x^2 k(x) dx),
         * and k is the kernel.
         */
        virtual float getRuleOfThumbCoefficient() = 0;
        
        /**
         * For Maximal Smoothing Principle bandwidth selection we compute
         * 
         *  R(k) = int(k^2(x) dx)
         * 
         * using Monte Carlo integration with Gaussian proposal distribution.
         */
        float calculateSquareIntegral(int D);
        
        /**
         * For Maximal Smoothing Principle bandwidth selection we compute
         * 
         *  kappa_2(k) = int(x^2 k(x) dx)
         * 
         * using Monte Carlo integration with Gaussian proposal distribution.
         */
        float calculateSecondMoment(int D);
    };
    
    /**
     * Construct a multivariate kernel as product of univariate kernels.
     */
    class ProductKernel : public MultivariateKernel {
    public:
        /**
         * Constructs a product kernel given the univariate kernel.
         */
        ProductKernel(Kernel* _kernel) :
                kernel(_kernel) {};
        
        /**
         * Destructor.
         */
        ~ProductKernel()
        {
            delete kernel;
        }
        
        /**
         * Evaluate the kernel.
         */
        virtual float evaluate(DataPoint* x)
        {
            const int D = x->getDimensionality();
            
            float K_x = 1;
            for (int d = 0; d < D; d++)
            {
                K_x *= kernel->evaluate(x, d);
            }
            
            return K_x;
        }
        
        /**
         * Returns the coefficient for the rule of thumb bandwidth selection.
         */
        virtual float getRuleOfThumbCoefficient()
        {
            return kernel->getRuleOfThumbCoefficient();
        }
        
    private:
        /**
         * Underlying univariate kernel.
         */
        Kernel* kernel;
    };
    
    /**
     * Multivariate Gaussian kernel for density estimation.
     */
    class MultivariateGaussianKernel : public MultivariateKernel {
    public:
        /**
         * Evaluate kernel for a given data point.
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
        
        /**
         * Get coefficient for rule of thumb bandwidth selection.
         */
        float getRuleOfThumbCoefficient()
        {
            // See table 4 in:
            //  B. E. Hansen.
            //  Lecture Notes on Nonparametrics.
            //  University of Wisconsin
            return 1.06f;
        }
    };
    
    /**
     * Multivariate Epanechnikov Kernel as described in
     * 
     *  P. B. Stark.
     *  Statistics 240 Lecture Notes, Part 10: Density Estimation.
     *  University of California Berkeley, 2008.
     */
    class MultivariateEpanechnikovKernel : public MultivariateKernel {
    public:
        /**
         * Constructor.
         */
        MultivariateEpanechnikovKernel() : MultivariateKernel(),
                cachedSphereVolume(false),
                sphereVolume(0),
                cachedD(0) {};
        
        /**
         * Evaluate kernel for a given data point.
         */
        virtual float evaluate(DataPoint* x)
        {
            const int D = x->getDimensionality();
            
            float inner = 0;
            for (int i = 0; i < D; i++)
            {
                inner += x->at(i)*x->at(i);
            }
            
            float K_x = 1 - inner;
            if (K_x > 0)
            {
                // The volume of a n dim unit ball.
                // @see http://en.wikipedia.org/wiki/Volume_of_an_n-ball
                float V_D = 0;
                
                if (cachedSphereVolume && cachedD == D)
                {
                    V_D = sphereVolume;
                }
                else
                {
                    if (D%2 == 0)
                    {
                        const int k = D/2;
                        V_D = pow(M_PI, k)/Util::factorial(k);
                    }
                    else
                    {
                        const int k = D/2; // Note that this rounds off!
                        V_D = (pow(2, k + 1) * pow(M_PI, k))/(Util::doubleFactorial(D));
                    }
                    
                    cachedSphereVolume = true;
                    sphereVolume = V_D;
                    cachedD = D;
                }
                
                assert(V_D > 0);
                
                return (D + 2)/(2.*V_D) * K_x;
            }
            else
            {
                return 0;
            }
        }
        
        /**
         * Get coefficient for rule of thumb bandwidth selection.
         */
        float getRuleOfThumbCoefficient()
        {
            // See table 4 in:
            //  B. E. Hansen.
            //  Lecture Notes on Nonparametrics.
            //  University of Wisconsin
            return 2.34f;
        }
        
    private:
        /**
         * The volume of the D-dimensional sphere is cached.
         */
        bool cachedSphereVolume;
        /**
         * The cached volume of a cachedD-dim sphere.
         */
        float sphereVolume;
        /**
         * Dimension of the cached sphere volume.
         */
        int cachedD;
        
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
         *  CORE and Institut de Statistique, Universite Catholique the Louvain.
         * 
         * Or:
         * 
         *  M. C. Jones, J. S. Marron, S. J. Sheather.
         *  A Brief Survey of Bandwidth Selection for Density Estimation.
         *  Journal of the American Statistical Association, 91(433), 1996.
         */
        static const int BANDWIDTH_RULE_OF_THUMB = 0;
        static const int BANDWIDTH_RULE_OF_THUMB_INTERQUARTILE = 1;
        static const int BANDWIDTH_MAXIMAL_SMOOTHING_PRINCIPLE = 2;
        
        /**
         * Constructor.
         */
        KernelDensityEstimator() : 
                kernel(new MultivariateGaussianKernel()),
                bandwidthSelectionMethod(BANDWIDTH_RULE_OF_THUMB),
                bandwidth(1),
                storage(0) {};
        
        /**
         * Contructs a kernel density estimator using the given kernel
         * and an empty data storage.
         */
        KernelDensityEstimator(MultivariateKernel* _kernel) : 
                kernel(_kernel),
                bandwidthSelectionMethod(BANDWIDTH_RULE_OF_THUMB),
                bandwidth(1),
                storage(0) {};
        
        /**
         * Constructs a kernel density estimator given the data with default
         * Gaussian kernel.
         */
        KernelDensityEstimator(UnlabeledDataStorage* _storage) : 
                kernel(new MultivariateGaussianKernel()),
                bandwidthSelectionMethod(BANDWIDTH_RULE_OF_THUMB),
                bandwidth(1),
                storage(_storage) {};
        
        /**
         * Create a kernel density estimator with the given data and kernel.
         */
        KernelDensityEstimator(UnlabeledDataStorage* _storage, MultivariateKernel* _kernel) : 
                kernel(_kernel),
                bandwidthSelectionMethod(BANDWIDTH_RULE_OF_THUMB), 
                bandwidth(1),
                storage(_storage) {};
        
        /**
         * Destructor.
         */
        virtual ~KernelDensityEstimator() {};
                
        /**
         * Sets the kernel to use for density estimation.
         */
        void setKernel(MultivariateKernel* _kernel)
        {
            kernel = _kernel;
        }

        /**
         * Returns the kernel used for density estimation.
         */
        MultivariateKernel* getKernel()
        {
            return kernel;
        }
                
        /**
         * Sets a single bandwidth for all dimensions.
         */
        void setBandwidth(float _bandwidth)
        {
            const int D = storage->getDimensionality();
            
            bandwidth = Eigen::VectorXf(D);
            for (int d = 0; d < D; d++)
            {
                bandwidth(d) = _bandwidth;
            }
        }
        
        /**
         * Sets the bandwidth vector.
         */
        void setBandwidth(Eigen::VectorXf _bandwidth)
        {
            assert(_bandwidth.rows() == storage->getDimensionality());
            bandwidth = _bandwidth;
        }
        
        /**
         * Returns the bandwidth vector.
         */
        Eigen::VectorXf getBandwidth()
        {
            return bandwidth;
        }
        
        /**
         * Selects the bandwidth according to the given bandwidth selection method.
         */
        Eigen::VectorXf selectBandwidth(int bandWidthSelectionMethod)
        {
            bandwidth = Eigen::VectorXf(storage->getDimensionality());
            switch (bandWidthSelectionMethod)
            {
                case BANDWIDTH_RULE_OF_THUMB:
                    selectBandwidthRuleOfThumb();
                    break;
                case BANDWIDTH_RULE_OF_THUMB_INTERQUARTILE:
                    selectBandwidthRuleOfThumbInterquartile();
                    break;
                case BANDWIDTH_MAXIMAL_SMOOTHING_PRINCIPLE:
                    selectBandwidthMaximalSmoothingPrinciple();
                    break;
                default:
                    assert(false);
                    break;
            }
            
            return bandwidth;
        }
        
        /**
         * Sets the data storage to use for density estimation.
         */
        void setDataStorage(UnlabeledDataStorage* _storage)
        {
            storage = _storage;
        }
        
        /**
         * Returns the data storage to use for density estimation.
         */
        UnlabeledDataStorage* getDataStorage()
        {
            return storage;
        }
        
        /**
         * Estimate the probability of a datapoint.
         */
        virtual float estimate(const DataPoint* x);
        
    private:
        /**
         * Calculate the variance of the data in one dimension.
         */
        float calculateVariance(int d);
        
        /**
         * Calculate the interquartile range along a dimension.
         */
        float calculateInterquartileRange(int d);
        
        /**
         * Select the bandwidth using the naive Gaussian method, see:
         * 
         *  B. A. Turlach.
         *  Bandwidth Selection in Kernel Density Estimation: A Review.
         *  CORE and Institut de Statistique, Universite Catholique the Louvain.
         */
        void selectBandwidthRuleOfThumb();
        
        /**
         * Select the bandwidth using the naive Gaussian rule of thumb, but using
         * the interquartile range isntead of the variance as measure of spread,
         * see:
         * 
         *  B. A. Turlach.
         *  Bandwidth Selection in Kernel Density Estimation: A Review.
         *  CORE and Institut de Statistique, Universite Catholique the Louvain.
         */
        void selectBandwidthRuleOfThumbInterquartile();
        
        /**
         * Select the bandwidths according to the Maximal Smoothing Priciple:
         * 
         *  B. A. Turlach.
         *  Bandwidth Selection in Kernel Density Estimation: A Review.
         *  CORE and Institut de Statistique, Universite Catholique the Louvain.
         */
        void selectBandwidthMaximalSmoothingPrinciple();
        
        /**
         * Used kernel.
         */
        MultivariateKernel* kernel;
        /**
         * Bandwidth selection method.
         */
        int bandwidthSelectionMethod;
        /**
         * The used bandwidth in each dimension.
         */
        Eigen::VectorXf bandwidth;
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
    class DensityTree : public Tree, public Estimator, public Generator {
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
        
    protected:
        /**
         * Adds a plain new node.
         */
        virtual void addNodeDerived(int depth);;
        
    private:
        /**
         * Compute and cache the partition function.
         */
        float getPartitionFunction(int D);
        
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
    
    /**
     * A kernel density tree.
     */
    class KernelDensityTree : public Tree, public Estimator {
    public:
        /**
         * Constructor.
         */
        KernelDensityTree();
        
        /**
         * Destructor.
         */
        ~KernelDensityTree() {};
        
        Gaussian & getGaussian(int node)
        {
            assert(node >= 0 && node < static_cast<int>(leftChild.size()));
            return gaussians[node];
        }
        
        KernelDensityEstimator & getEstimator(int node)
        {
            assert(node >= 0 && node < static_cast<int>(leftChild.size()));
            return estimators[node];
        }
        
        /**
         * Estimate probability of a point.
         */
        virtual float estimate(const DataPoint*);
        
    protected:
        /**
         * Add a node.
         */
        virtual void addNodeDerived(int depth);
        
    private:
        /**
         * Get the partition function.
         */
        float getPartitionFunction(int D);
        
        /**
         * The Gaussians at the leafs.
         */
        std::vector<Gaussian> gaussians;
        /**
         * The leaf node kernel density estimators.
         */
        std::vector<KernelDensityEstimator> estimators;
        /**
         * Kernel used for kernel density estimation.
         */
        MultivariateKernel* kernel;
        /**
         * Partition function.
         */
        float partitionFunction;
        /**
         * The partition function is cached.
         */
        bool cachedPartitionFunction;
        
    };
}
#endif