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

using namespace libf;

////////////////////////////////////////////////////////////////////////////////
/// Gaussian
////////////////////////////////////////////////////////////////////////////////

Gaussian::Gaussian(Eigen::VectorXf _mean, Eigen::MatrixXf _covariance) :
        mean(_mean), 
        covariance(_covariance),
        cachedInverse(false),
        cachedDeterminant(false),
        randN(rng, norm)
{
    const int rows = _covariance.rows();
    const int cols = _covariance.cols();
    
    assert(rows == cols);
    assert(rows == _mean.rows());
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(covariance);
    transform = solver.eigenvectors()*solver.eigenvalues().cwiseSqrt().asDiagonal();
}

Gaussian::Gaussian(Eigen::VectorXf _mean, Eigen::MatrixXf _covariance, float _covarianceDeterminant) :
        mean(_mean), 
        covariance(_covariance),
        cachedInverse(false),
        cachedDeterminant(true),
        covarianceDeterminant(_covarianceDeterminant),
        randN(rng, norm)
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

float DensityTree::estimate(const DataPoint* x)
{
    int node = findLeafNode(x);
    return this->gaussians[node].evaluate(x);
}

DataPoint* DensityTree::sample()
{
    // We begin by sampling a random path in the tree.
    int node = 0;
    while (leftChild[node] > 0)
    {
        int r = std::rand()%2;
        node = node + r;
        
        assert(r < static_cast<int>(leftChild.size()));
    }
    
    assert(leftChild[node] == 0);
    
    // Now sample from the final Gaussian.
    return gaussians[node].sample();
}

void DensityTree::addNode(int depth)
{
    Tree::addNode(depth);
    
    gaussians.push_back(Gaussian());
}