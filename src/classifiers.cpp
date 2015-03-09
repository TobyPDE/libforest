#include "libforest/classifiers.h"
#include "libforest/data.h"
#include "libforest/io.h"
#include "libforest/util.h"
#include "fastlog.h"
#include <ios>
#include <iostream>
#include <string>
#include <cmath>
#include <Eigen/LU>

using namespace libf;

#define ENTROPY(p) (-(p)*fastlog2(p))

////////////////////////////////////////////////////////////////////////////////
/// Classifier
////////////////////////////////////////////////////////////////////////////////

void Classifier::classify(const DataStorage* storage, std::vector<int> & results) const
{
    // Clean the result set
    results.erase(results.begin(), results.end());
    results.resize(storage->getSize());
    
    // Classify each individual data point
    for (int i = 0; i < storage->getSize(); i++)
    {
        results[i] = classify(storage->getDataPoint(i));
    }
}

int Classifier::classify(const DataPoint* x) const
{
    // Get the class posterior
    std::vector<float> posterior;
    this->classLogPosterior(x, posterior);
    
    assert(posterior.size() > 0);
    
    int label = 0;
    float prob = posterior[0];
    const int size = static_cast<int>(posterior.size());
    
    for (int i = 1; i < size; i++)
    {
        if (posterior[i] > prob)
        {
            label = i;
            prob = posterior[i];
        }
    }
    
    return label;
}

////////////////////////////////////////////////////////////////////////////////
/// EfficientEntropyHistogram
////////////////////////////////////////////////////////////////////////////////

void EfficientEntropyHistogram::addOne(const int i)
{
    // assert(i >= 0 && i < bins);
    
    totalEntropy += ENTROPY(mass);
    mass += 1;
    totalEntropy -= ENTROPY(mass);
    histogram[i]++;
    totalEntropy -= entropies[i];
    entropies[i] = ENTROPY(histogram[i]); 
    totalEntropy += entropies[i];
}

void EfficientEntropyHistogram::subOne(const int i)
{
    // assert(i >= 0 && i < bins);
    
    totalEntropy += ENTROPY(mass);
    mass -= 1;
    totalEntropy -= ENTROPY(mass);

    histogram[i]--;
    totalEntropy -= entropies[i];
    if (histogram[i] < 1)
    {
        entropies[i] = 0;
    }
    else
    {
        entropies[i] = ENTROPY(histogram[i]); 
        totalEntropy += entropies[i];
    }
}

////////////////////////////////////////////////////////////////////////////////
/// DecisionTree
////////////////////////////////////////////////////////////////////////////////

DecisionTree::DecisionTree()
{
    // Indicates that we do not need to maintain statistics.
    statistics = false;
    
    splitFeatures.reserve(LIBF_GRAPH_BUFFER_SIZE);
    thresholds.reserve(LIBF_GRAPH_BUFFER_SIZE);
    leftChild.reserve(LIBF_GRAPH_BUFFER_SIZE);
    histograms.reserve(LIBF_GRAPH_BUFFER_SIZE);
    depths.reserve(LIBF_GRAPH_BUFFER_SIZE);
    
    // Add at least the root node with index 0
    addNode(0);
}

DecisionTree::DecisionTree(bool _statistics)
{
    statistics = _statistics;
    
    splitFeatures.reserve(LIBF_GRAPH_BUFFER_SIZE);
    thresholds.reserve(LIBF_GRAPH_BUFFER_SIZE);
    leftChild.reserve(LIBF_GRAPH_BUFFER_SIZE);
    histograms.reserve(LIBF_GRAPH_BUFFER_SIZE);
    depths.reserve(LIBF_GRAPH_BUFFER_SIZE);
    
    if (statistics)
    {
        nodeStatistics.reserve(LIBF_GRAPH_BUFFER_SIZE);
        leftChildStatistics.reserve(LIBF_GRAPH_BUFFER_SIZE);
        rightChildStatistics.reserve(LIBF_GRAPH_BUFFER_SIZE);
        nodeThresholds.reserve(LIBF_GRAPH_BUFFER_SIZE);
        nodeFeatures.reserve(LIBF_GRAPH_BUFFER_SIZE);
    }
    
    // Important to add the first node after enabling statistics!
    addNode(0);
}

void DecisionTree::addNode(int depth)
{
    splitFeatures.push_back(0);
    thresholds.push_back(0);
    leftChild.push_back(0);
    histograms.push_back(std::vector<float>());
    depths.push_back(depth);
    
    if (statistics)
    {
        nodeStatistics.push_back(EfficientEntropyHistogram());
        leftChildStatistics.push_back(std::vector<EfficientEntropyHistogram>());
        rightChildStatistics.push_back(std::vector<EfficientEntropyHistogram>());
        nodeThresholds.push_back(std::vector< std::vector<float> >());
        nodeFeatures.push_back(std::vector<int>());
    }
}

int DecisionTree::splitNode(int node)
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

int DecisionTree::findLeafNode(const DataPoint* x) const
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

void DecisionTree::classLogPosterior(const DataPoint* x, std::vector<float> & probabilities) const
{
    // Get the leaf node
    const int leafNode = findLeafNode(x);
    probabilities = getHistogram(leafNode);
}

void DecisionTree::read(std::istream& stream)
{
    // Write the attributes to the file
    readBinary(stream, splitFeatures);
    readBinary(stream, thresholds);
    readBinary(stream, leftChild);
    readBinary(stream, histograms);
    
    readBinary(stream, statistics);
    if (statistics) {
        readBinary(stream, nodeStatistics);
        readBinary(stream, leftChildStatistics);
        readBinary(stream, rightChildStatistics);
        readBinary(stream, nodeThresholds);
        readBinary(stream, nodeFeatures);
    }
}

void DecisionTree::write(std::ostream& stream) const
{
    // Write the attributes to the file
    writeBinary(stream, splitFeatures);
    writeBinary(stream, thresholds);
    writeBinary(stream, leftChild);
    writeBinary(stream, histograms);
    
    writeBinary(stream, statistics);
    if (statistics) {
        writeBinary(stream, nodeStatistics);
        writeBinary(stream, leftChildStatistics);
        writeBinary(stream, rightChildStatistics);
        writeBinary(stream, nodeThresholds);
        writeBinary(stream, nodeFeatures);
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
        randN(rng, norm)
{
    const int rows = _covariance.rows();
    const int cols = _covariance.cols();
    
    assert(rows == cols);
    assert(rows == _mean.rows());
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(covariance);
        
    rot = solver.eigenvectors();
    scl = solver.eigenvalues();
    for (int i =  0; i < rows; i++)
    {
        scl(i, 0) = std::sqrt(scl(i));
    }
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
        
    rot = solver.eigenvectors();
    scl = solver.eigenvalues();
    for (int i =  0; i < rows; i++)
    {
        scl(i, 0) = std::sqrt(scl(i));
    }
}

void Gaussian::setCovariance(Eigen::MatrixXf _covariance)
{
    const int rows = _covariance.rows();
    const int cols = _covariance.cols();
    
    assert(rows == cols);
    
    covariance = Eigen::MatrixXf(_covariance);
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(covariance);
        
    rot = solver.eigenvectors();
    scl = solver.eigenvalues();
    for (int i =  0; i < rows; i++)
    {
        scl(i, 0) = std::sqrt(scl(i));
    }
    
    cachedInverse = false;
    cachedDeterminant = false;
}

DataPoint* Gaussian::sample()
{
    const int rows = mean.rows();
    
    DataPoint* x = new DataPoint(rows);
    for (int i = 0 ; i < rows; i++)
    {
        x->at(i) = randN()*scl(i, 0);
    }
    
    return x;
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
    return 1/std::sqrt(pow(2*M_PI, mean.rows())*covarianceDeterminant)
            * std::exp(-1/2 * offset.transpose()*covarianceInverse*offset);
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

////////////////////////////////////////////////////////////////////////////////
/// EfficientCovarianceMatrix
////////////////////////////////////////////////////////////////////////////////

void EfficientCovarianceMatrix::addOne(const DataPoint* x)
{
    assert(x->getDimensionality() == mean.rows());
    assert(x->getDimensionality() == covariance.rows());
    
    for (int i = 0; i < x->getDimensionality(); i++)
    {
        // Update runnign estimate of mean.
        mean(i) += x->at(i);
        
        for (int j = 0; j < x->getDimensionality(); j++)
        {
            // Update runnign estimate of covariance.
            covariance(i, j) += x->at(i)*x->at(j);
        }
    }
    
    mass += 1;
}

void EfficientCovarianceMatrix::subOne(const DataPoint* x)
{
    assert(x->getDimensionality() == mean.rows());
    assert(x->getDimensionality() == covariance.rows());
    
    for (int i = 0; i < x->getDimensionality(); i++)
    {
        // Update runnign estimate of mean.
        mean(i) -= x->at(i);
        
        for (int j = 0; j < x->getDimensionality(); j++)
        {
            // Update runnign estimate of covariance.
            covariance(i, j) -= x->at(i)*x->at(j);
        }
    }
    
    mass += 1;
}

float EfficientCovarianceMatrix::getEntropy()
{
    return ENTROPY(mass)*ENTROPY(getDeterminant());
}

float EfficientCovarianceMatrix::getDeterminant()
{
    if (!cachedDeterminant)
    {
        covarianceDeterminant = getCovariance().determinant();
    }
    
    return covarianceDeterminant;
}

Eigen::MatrixXf & EfficientCovarianceMatrix::getCovariance()
{
    if (!cachedTrueCovariance)
    {
        trueCovariance = (covariance - mean*mean.transpose())/mass;
    }
    
    return trueCovariance;
}

////////////////////////////////////////////////////////////////////////////////
/// RandomForest
////////////////////////////////////////////////////////////////////////////////

RandomForest::~RandomForest()
{
    for (size_t i = 0; i < trees.size(); i++)
    {
        delete trees[i];
    }
}

void RandomForest::classLogPosterior(const DataPoint* x, std::vector<float> & probabilities) const
{
    assert(getSize() > 0);
    trees[0]->classLogPosterior(x, probabilities);
    
    // Let the crowd decide
    for (size_t i = 1; i < trees.size(); i++)
    {
        // Get the probabilities from the current tree
        std::vector<float> currentHist;
        trees[i]->classLogPosterior(x, currentHist);
        
        // Accumulate the votes
        for (int  c = 0; c < currentHist.size(); c++)
        {
            probabilities[c] += currentHist[c];
        }
    }
}

void RandomForest::write(std::ostream& stream) const
{
    // Write the number of trees in this ensemble
    writeBinary(stream, getSize());
    
    // Write the individual trees
    for (int i = 0; i < getSize(); i++)
    {
        getTree(i)->write(stream);
    }
}

void RandomForest::read(std::istream& stream)
{
    // Read the number of trees in this ensemble
    int size;
    readBinary(stream, size);
    
    // Read the trees
    for (int i = 0; i < size; i++)
    {
        DecisionTree* tree = new DecisionTree();
        tree->read(stream);
        addTree(tree);
    }
}

////////////////////////////////////////////////////////////////////////////////
/// BoostedRandomForest
////////////////////////////////////////////////////////////////////////////////

BoostedRandomForest::~BoostedRandomForest()
{
    for (size_t i = 0; i < trees.size(); i++)
    {
        delete trees[i];
    }
}


void BoostedRandomForest::write(std::ostream& stream) const
{
    // Write the number of trees in this ensemble
    writeBinary(stream, getSize());
    
    // Write the individual trees
    for (int i = 0; i < getSize(); i++)
    {
        // Write the weight
        writeBinary(stream, weights[i]);
        getTree(i)->write(stream);
    }
}

void BoostedRandomForest::read(std::istream& stream)
{
    // Read the number of trees in this ensemble
    int size;
    readBinary(stream, size);
    
    // Read the trees
    for (int i = 0; i < size; i++)
    {
        DecisionTree* tree = new DecisionTree();
        tree->read(stream);
        // Read the weight
        float weight;
        readBinary(stream, weight);
        addTree(tree, weight);
    }
}


void BoostedRandomForest::classLogPosterior(const DataPoint* x, std::vector<float> & probabilities) const
{
    assert(getSize() > 0);
    // Determine the number of classes by looking at a histogram
    trees[0]->classLogPosterior(x, probabilities);
    // Initialize the result vector
    const int C = static_cast<int>(probabilities.size());
    for (int c = 0; c < C; c++)
    {
        probabilities[c] = 0;
    }
    
    // Let the crowd decide
    for (size_t i = 0; i < trees.size(); i++)
    {
        // Get the resulting label
        const int label = trees[i]->classify(x);
        probabilities[label] += weights[i];
    }
}