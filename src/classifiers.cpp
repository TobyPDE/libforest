#include "libforest/classifiers.h"
#include "libforest/data.h"
#include "libforest/io.h"
#include "libforest/util.h"
#include "fastlog.h"
#include <ios>
#include <iostream>
#include <string>
#include <cmath>

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
    assert(i >= 0 && i < bins);
    
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
    assert(i >= 0 && i < bins);
    
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

void EfficientEntropyHistogram::initEntropies()
{
    if (getMass() > 1)
    {
        totalEntropy = -ENTROPY(getMass());
        for (int i = 0; i < bins; i++)
        {
            if (at(i) == 0)
            {
                continue;
            }
            
            entropies[i] = ENTROPY(histogram[i]);
            totalEntropy += entropies[i];
        }
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
}

void DecisionTree::write(std::ostream& stream) const
{
    // Write the attributes to the file
    writeBinary(stream, splitFeatures);
    writeBinary(stream, thresholds);
    writeBinary(stream, leftChild);
    writeBinary(stream, histograms);
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