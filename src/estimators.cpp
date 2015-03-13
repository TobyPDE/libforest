#include "libforest/classifiers.h"
#include "libforest/estimators.h"
#include "libforest/data.h"
#include "libforest/io.h"
#include "libforest/util.h"
#include "libforest/fastlog.h"
#include <ios>
#include <iostream>
#include <string>
#include <cmath>
#include <Eigen/LU>

// For gaussian sampling
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>

/**
 * Uniform pseudo random generator.
 */
boost::mt19937 rng;
/**
 * Scalar Gaussian distribution.
 */
boost::normal_distribution<float> norm;
/**
 * Gaussian generator.
 */
boost::variate_generator< boost::mt19937&, boost::normal_distribution<float> > randN(rng, norm);

using namespace libf;

////////////////////////////////////////////////////////////////////////////////
/// KernelDensityEstimator
////////////////////////////////////////////////////////////////////////////////

float MultivariateKernel::calculateSquareIntegral(int D)
{
    const int N = 100 * pow(10, D);
    Gaussian gaussian(Eigen::VectorXf::Zero(D), Eigen::MatrixXf::Identity(D, D));
    
    float expectation = 0;
    for (int n = 0; n < N; n++)
    {
        DataPoint* x = gaussian.sample();
        float p_x = gaussian.evaluate(x);
        float k_x = evaluate(x);
        
        expectation += k_x*k_x/p_x;
    }
    
    return expectation/N;
}

float MultivariateKernel::calculateSecondMoment(int D)
{
    const int N = 100 * pow(10, D);
    Gaussian gaussian(Eigen::VectorXf::Zero(D), Eigen::MatrixXf::Identity(D, D));
    
    float expectation = 0;
    for (int n = 0; n < N; n++)
    {
        DataPoint* x = gaussian.sample();
        
        float p_x = gaussian.evaluate(x);
        float k_x = evaluate(x);
        
        float inner = 0;
        for (int d = 0; d < D; d++)
        {
            inner += x->at(d)*x->at(d);
        }
        
        expectation += inner*k_x/p_x;
    }
    
    return expectation/N;
}

////////////////////////////////////////////////////////////////////////////////
/// KernelDensityEstimator
////////////////////////////////////////////////////////////////////////////////

float KernelDensityEstimator::estimate(const DataPoint* x)
{
    const int D = storage->getDimensionality();
    const int N = storage->getSize();
    
    assert(bandwidth.rows() == D);
    
    float p_x = 0;
    DataPoint x_bar(D);
    
    for (int n = 0; n < N; n++)
    {
        for (int d = 0; d < D; d++)
        {
            x_bar.at(d) = (storage->getDataPoint(n)->at(d) - x->at(d))/bandwidth(d);
        }
        
        p_x += kernel->evaluate(&x_bar);
    }
    
    float H = 1;
    for (int d = 0; d < D; d++)
    {
        H *= bandwidth(d);
    }
    
    return p_x/(N*H);
}

float KernelDensityEstimator::calculateVariance(int d)
{
    const int N = storage->getSize();
    const int D = storage->getDimensionality();
    
    assert(d >= 0 && d < D);
    
    float mean = 0;
    float variance = 0;
    
    for (int n = 0; n < N; n++)
    {
        float value = storage->getDataPoint(n)->at(d);
        
        mean += value;
        variance += value*value;
    }
    
    mean /= N;
    variance /= N;
    variance -= mean*mean;
    
    return variance;
}

float KernelDensityEstimator::calculateInterquartileRange(int d)
{
    const float N = storage->getSize();
    
    std::vector<float> points(N);
    for (int n = 0; n < N; n++)
    {
        points[n] = storage->getDataPoint(n)->at(d);
    }
    
    std::sort(points.begin(), points.end());
    
    float Q_1 = points[std::floor(points.size()*1./4.)];
    float Q_3 = points[std::floor(points.size()*3./4.)];
    
    assert(Q_3 > Q_1);
    
    return Q_3 - Q_1;
}

void KernelDensityEstimator::selectBandwidthRuleOfThumb()
{
    const float C = kernel->getRuleOfThumbCoefficient();
    const int N = storage->getSize();
    const int D = storage->getDimensionality();
    
    bandwidth = Eigen::VectorXf(D);
    for (int d = 0; d < D; d++)
    {
        bandwidth(d) = C * std::sqrt(calculateVariance(d)) * std::pow(N, -1.f/5.f);
    }
}

void KernelDensityEstimator::selectBandwidthRuleOfThumbInterquartile()
{
    const float C = kernel->getRuleOfThumbCoefficient();
    const int N = storage->getSize();
    const int D = storage->getDimensionality();
    
    bandwidth = Eigen::VectorXf(D);
    for (int d = 0; d < D; d++)
    {
        bandwidth(d) = C * std::min(std::sqrt(calculateVariance(d)), calculateInterquartileRange(d)/1.34f) 
                * std::pow(N, -1.f/5.f);
    }
}

void KernelDensityEstimator::selectBandwidthMaximalSmoothingPrinciple()
{
    const int N = storage->getSize();
    const int D = storage->getDimensionality();
    
    float R = kernel->calculateSquareIntegral(D);
    float kappa = kernel->calculateSecondMoment(D);
    
    bandwidth = Eigen::VectorXf(D);
    for (int d = 0; d < D; d++)
    {
        bandwidth(d) = 3 * pow(35.f, -1.f/5.f) * std::sqrt(calculateVariance(d))
                * std::pow(R/(kappa*kappa), 1.f/5.f) * std::pow(N, -1.f/5.f);
    }
}

////////////////////////////////////////////////////////////////////////////////
/// Gaussian
////////////////////////////////////////////////////////////////////////////////

Gaussian::Gaussian(Eigen::VectorXf _mean, Eigen::MatrixXf _covariance) :
        mean(_mean), 
        covariance(_covariance),
        cachedInverse(false),
        cachedDeterminant(false),
        dataSupport(0) 
{
    const int rows = _covariance.rows();
    const int cols = _covariance.cols();
    
    assert(rows == cols);
    assert(rows == _mean.rows());
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(covariance);
    transform = solver.eigenvectors()*solver.eigenvalues().cwiseSqrt().asDiagonal();
}

void Gaussian::setCovariance(const Eigen::MatrixXf _covariance)
{
    const int rows = _covariance.rows();
    const int cols = _covariance.cols();
    
    assert(rows == cols);
    
    covariance = _covariance;
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(covariance);
    transform = solver.eigenvectors()*solver.eigenvalues().cwiseSqrt().asDiagonal();
    
    cachedInverse = false;
    cachedDeterminant = false;
}

DataPoint* Gaussian::sample()
{
    const int rows = mean.rows();
    
    assert(rows == covariance.rows());
    assert(rows == covariance.cols());
    assert(rows > 0);
    
    Eigen::VectorXf randNVector(rows);
    for (int i = 0; i < rows; i++)
    {
        randNVector(i) = randN();
    }
    
    Eigen::VectorXf x = transform * randNVector + mean;
    
    return asDataPoint(x);
}

float Gaussian::evaluate(const DataPoint* x)
{
    assert(x->getDimensionality() == mean.rows());
    assert(x->getDimensionality() == covariance.rows());
    
    // Invert the covariance matrix if not cached.
    if (!cachedInverse)
    {
        covarianceInverse = covariance.inverse();
        cachedInverse = true;
    }
    
    // @see http://eigen.tuxfamily.org/dox-devel/group__LU__Module.html
    if (!cachedDeterminant)
    {
        covarianceDeterminant = covariance.determinant();
        cachedDeterminant = true;
    }
    
    Eigen::VectorXf x_bar = asEigenVector(x);
    Eigen::VectorXf offset = x_bar - mean;
    
    float p = 1/std::sqrt(pow(2*M_PI, mean.rows())*covarianceDeterminant)
            * std::exp(- (1./2.) * offset.transpose()*covarianceInverse*offset);
    
    return p;
}

Eigen::VectorXf Gaussian::asEigenVector(const DataPoint* x)
{
    Eigen::VectorXf x_bar(x->getDimensionality());
    
    for (int i = 0; i < x->getDimensionality(); i++)
    {
        x_bar(i) = x->at(i);
    }
    
    return x_bar;
}

DataPoint* Gaussian::asDataPoint(const Eigen::VectorXf & x)
{
    DataPoint* x_bar = new DataPoint(x.rows());
    
    for (int i = 0; i < x.rows(); i++)
    {
        x_bar->at(i) = x(i);
    }
    
    return x_bar;
}

////////////////////////////////////////////////////////////////////////////////
/// DensityTree
////////////////////////////////////////////////////////////////////////////////

DensityTree::DensityTree() : Tree()
{    
    splitFeatures.reserve(LIBF_GRAPH_BUFFER_SIZE);
    thresholds.reserve(LIBF_GRAPH_BUFFER_SIZE);
    leftChild.reserve(LIBF_GRAPH_BUFFER_SIZE);
    depths.reserve(LIBF_GRAPH_BUFFER_SIZE);
    gaussians.reserve(LIBF_GRAPH_BUFFER_SIZE);
}

void DensityTree::addNodeDerived(int depth)
{
    gaussians.push_back(Gaussian());
}

float DensityTree::getPartitionFunction(int D)
{
    if (!cachedPartitionFunction)
    {
        const int N = 100 * pow(10, D);
        const int nodes = static_cast<int>(leftChild.size());
        
        for (int node = 0; node < nodes; node++)
        {
            if (leftChild[node] == 0)
            {
                int count = 0;
                for (int n = 0; n < N; n++)
                {
                    DataPoint* x = gaussians[node].sample();
                    
                    int leaf = findLeafNode(x);
                    if (leaf == node)
                    {
                        count++;
                    }
                    
                    delete x;
                }
                
                float volume = ((float) count)/N;
                partitionFunction += gaussians[node].getDataSupport()/volume;
            }
        }
        
        cachedPartitionFunction = true;
    }
    
    return partitionFunction;
}

float DensityTree::estimate(const DataPoint* x)
{
    assert(leftChild.size() > 0);
    
    int node = findLeafNode(x);
    
    // Get the normalizer for our distribution.
    const float Z = getPartitionFunction(x->getDimensionality());
    
    return this->gaussians[node].evaluate(x)/Z;
}

DataPoint* DensityTree::sample()
{
    assert(leftChild.size() > 0);
    
    // We begin by sampling a random path in the tree.
    int node = std::rand()%leftChild.size();
    while (leftChild[node] > 0)
    {
        node = std::rand()%leftChild.size();
    }
    
    assert(leftChild[node] == 0);
    
    // Now sample from the final Gaussian.
    return gaussians[node].sample();
}

////////////////////////////////////////////////////////////////////////////////
/// DensityForest
////////////////////////////////////////////////////////////////////////////////

float DensityForest::estimate(const DataPoint* x)
{
    const int T = static_cast<int>(trees.size());
    
    float p_x = 0;
    for (int t = 0; t < T; t++)
    {
        p_x += getTree(t)->estimate(x);
    }
    
    return p_x/T;
}

DataPoint* DensityForest::sample()
{
    int t = std::rand()%trees.size();
    DensityTree* tree = getTree(t);
    
    // We begin by sampling a random path in the tree.
    int node = std::rand()%tree->getNumNodes();
    while (tree->getLeftChild(node) > 0)
    {
        node = std::rand()%tree->getNumNodes();
    }
    
    assert(tree->getLeftChild(node) == 0);
    
    // Now sample from the final Gaussian.
    return tree->getGaussian(node).sample();
}