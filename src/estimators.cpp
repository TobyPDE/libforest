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

DensityTree::DensityTree()
{    
    splitFeatures.reserve(LIBF_GRAPH_BUFFER_SIZE);
    thresholds.reserve(LIBF_GRAPH_BUFFER_SIZE);
    leftChild.reserve(LIBF_GRAPH_BUFFER_SIZE);
    depths.reserve(LIBF_GRAPH_BUFFER_SIZE);
    gaussians.reserve(LIBF_GRAPH_BUFFER_SIZE);
    
    // Add at least the root node with index 0
    addNode(0);
}

void DensityTree::addNode(int depth)
{
    splitFeatures.push_back(0);
    thresholds.push_back(0);
    leftChild.push_back(0);
    depths.push_back(depth);
    gaussians.push_back(Gaussian());
}

int DensityTree::splitNode(int node)
{
    // Make sure this is a valid node ID
    assert(0 <= node && node < static_cast<int>(splitFeatures.size()));
    // Make sure this is a former leaf node
    assert(leftChild[node] == 0);
    
    // Determine the index of the new left child
    const int leftNode = static_cast<int>(splitFeatures.size());
    
    // Add the child nodes
    const int depth = depths[node] + 1;
    addNode(depth);
    addNode(depth);
    
    // Set the child relation
    leftChild[node] = leftNode;
    
    return leftNode;
}

int DensityTree::findLeafNode(const DataPoint* x) const
{
    // Select the root node as current node
    int node = 0;
    
    // Follow the tree until we hit a leaf node
    while (leftChild[node] != 0)
    {
        // Check the threshold
        if (x->at(splitFeatures[node]) < thresholds[node])
        {
            // Go to the left
            node = leftChild[node];
        }
        else
        {
            // Go to the right
            node = leftChild[node] + 1;
        }
    }
    
    return node;
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
    int node = findLeafNode(x);
    
    // Get the normalizer for our distribution.
    const float Z = getPartitionFunction(x->getDimensionality());
    
    return this->gaussians[node].evaluate(x);//Z;
}

DataPoint* DensityTree::sample()
{
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