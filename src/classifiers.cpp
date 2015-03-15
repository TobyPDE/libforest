#include "libforest/classifiers.h"
#include "libforest/data.h"
#include "libforest/io.h"
#include "libforest/util.h"
#include <ios>
#include <iostream>
#include <string>
#include <cmath>

using namespace libf;

////////////////////////////////////////////////////////////////////////////////
/// Classifier
////////////////////////////////////////////////////////////////////////////////

void Classifier::classify(AbstractDataStorage::ptr storage, std::vector<int> & results) const
{
    // Clean the result set
    results.resize(storage->getSize());
    
    // Classify each individual data point
    for (int i = 0; i < storage->getSize(); i++)
    {
        results[i] = classify(storage->getDataPoint(i));
    }
}

int Classifier::classify(const DataPoint & x) const
{
    // Get the class posterior
    std::vector<float> posterior;
    classLogPosterior(x, posterior);
    
    return Util::argMax(posterior);
}

////////////////////////////////////////////////////////////////////////////////
/// Tree
////////////////////////////////////////////////////////////////////////////////

Tree::Tree()
{
    splitFeatures.reserve(LIBF_GRAPH_BUFFER_SIZE);
    thresholds.reserve(LIBF_GRAPH_BUFFER_SIZE);
    leftChild.reserve(LIBF_GRAPH_BUFFER_SIZE);
    depths.reserve(LIBF_GRAPH_BUFFER_SIZE);
    
    // Add at least the root node with index 0
    // addNode(0);
}

void Tree::addNode(int depth)
{
    splitFeatures.push_back(0);
    thresholds.push_back(0);
    leftChild.push_back(0);
    depths.push_back(depth);
    
    addNodeDerived(depth);
}

int Tree::splitNode(int node)
{
    // Make sure this is a valid node ID
    BOOST_ASSERT_MSG(0 <= node && node < static_cast<int>(splitFeatures.size()), "Invalid node index.");
    // Make sure this is a former leaf node
    BOOST_ASSERT_MSG(leftChild[node] == 0, "Cannot split non-leaf node.");
    
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

int DecisionTree::findLeafNode(const DataPoint & x) const
{
    // Select the root node as current node
    int node = 0;
    
    // Follow the tree until we hit a leaf node
    while (leftChild[node] != 0)
    {
        BOOST_ASSERT(splitFeatures[node] >= 0 && splitFeatures[node] < x.rows());
        
        // Check the threshold
        if (x(splitFeatures[node]) < thresholds[node])
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

void Tree::read(std::istream& stream)
{
    // Write the attributes to the file
    readBinary(stream, splitFeatures);
    readBinary(stream, thresholds);
    readBinary(stream, leftChild);
}

void Tree::write(std::ostream& stream) const
{
    // Write the attributes to the file
    writeBinary(stream, splitFeatures);
    writeBinary(stream, thresholds);
    writeBinary(stream, leftChild);
}

////////////////////////////////////////////////////////////////////////////////
/// DecisionTree
////////////////////////////////////////////////////////////////////////////////

DecisionTree::DecisionTree() : Tree()
{
    // Indicates that we do not need to maintain statistics.
    statistics = false;
    
    // Note that it doe snot matter that Tree already adds a node
    // before reserving the space!
    histograms.reserve(LIBF_GRAPH_BUFFER_SIZE);
}

DecisionTree::DecisionTree(bool _statistics) : Tree()
{
    histograms.reserve(LIBF_GRAPH_BUFFER_SIZE);
    
    statistics = _statistics;
    if (statistics)
    {
        nodeStatistics.reserve(LIBF_GRAPH_BUFFER_SIZE);
        leftChildStatistics.reserve(LIBF_GRAPH_BUFFER_SIZE);
        rightChildStatistics.reserve(LIBF_GRAPH_BUFFER_SIZE);
        nodeThresholds.reserve(LIBF_GRAPH_BUFFER_SIZE);
        nodeFeatures.reserve(LIBF_GRAPH_BUFFER_SIZE);
    }
}

void DecisionTree::addNodeDerived(int depth)
{
    histograms.push_back(std::vector<float>());
    
    if (statistics)
    {
        nodeStatistics.push_back(EfficientEntropyHistogram());
        leftChildStatistics.push_back(std::vector<EfficientEntropyHistogram>());
        rightChildStatistics.push_back(std::vector<EfficientEntropyHistogram>());
        nodeThresholds.push_back(std::vector< std::vector<float> >());
        nodeFeatures.push_back(std::vector<int>());
    }
}

void DecisionTree::classLogPosterior(const DataPoint & x, std::vector<float> & probabilities) const
{
    // Get the leaf node
    const int leafNode = findLeafNode(x);
    probabilities = getHistogram(leafNode);
}

void DecisionTree::read(std::istream& stream)
{
    Tree::read(stream);
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
    Tree::write(stream);
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
/// RandomForest
////////////////////////////////////////////////////////////////////////////////

void RandomForest::classLogPosterior(const DataPoint & x, std::vector<float> & probabilities) const
{
    BOOST_ASSERT_MSG(getSize() > 0, "Cannot classify a point from an empty ensemble.");
    
    trees[0]->classLogPosterior(x, probabilities);
    
    // Let the crowd decide
    for (size_t i = 1; i < trees.size(); i++)
    {
        // Get the probabilities from the current tree
        std::vector<float> currentHist;
        trees[i]->classLogPosterior(x, currentHist);
        
        BOOST_ASSERT(currentHist.size() > 0);
        
        // Accumulate the votes
        for (size_t c = 0; c < currentHist.size(); c++)
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
        DecisionTree::ptr tree = std::make_shared<DecisionTree>();
        
        tree->read(stream);
        addTree(tree);
    }
}

////////////////////////////////////////////////////////////////////////////////
/// BoostedRandomForest
////////////////////////////////////////////////////////////////////////////////

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
        DecisionTree::ptr tree = std::make_shared<DecisionTree>();
        tree->read(stream);
        // Read the weight
        float weight;
        readBinary(stream, weight);
        addTree(tree, weight);
    }
}


void BoostedRandomForest::classLogPosterior(const DataPoint & x, std::vector<float> & probabilities) const
{
    BOOST_ASSERT_MSG(getSize() > 0, "Cannot classify a point from an empty ensemble.");
    // TODO: This can be done way more efficient
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