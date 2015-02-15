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

void Classifier::classify(DataStorage* storage, std::vector<int> & results) const
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

int Classifier::classify(DataPoint* x) const
{
    // Get the class posterior
    std::vector<float> posterior;
    float normalization;
    this->classPosterior(x, posterior, normalization);
    
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
/// DecisionTree
////////////////////////////////////////////////////////////////////////////////

DecisionTree::DecisionTree()
{
    splitFeatures.reserve(GRAPH_BUFFER_SIZE);
    thresholds.reserve(GRAPH_BUFFER_SIZE);
    leftChild.reserve(GRAPH_BUFFER_SIZE);
    histograms.reserve(GRAPH_BUFFER_SIZE);
    // Add at least the root node with index 0
    addNode();
}

void DecisionTree::addNode()
{
    splitFeatures.push_back(0);
    thresholds.push_back(0);
    leftChild.push_back(0);
    histograms.push_back(std::vector<int>());
}

int DecisionTree::splitNode(int node)
{
    // Make sure this is a valid node ID
    assert(0 <= node && node < static_cast<int>(splitFeatures.size()));
    // Make sure this is a former child node
    assert(leftChild[node] == 0);
    
    // Determine the index of the new left child
    const int leftNode = static_cast<int>(splitFeatures.size());
    
    // Add the child nodes
    addNode();
    addNode();
    
    // Set the child relation
    leftChild[node] = leftNode;
    
    return leftNode;
}

int DecisionTree::findLeafNode(DataPoint* x) const
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

void DecisionTree::classPosterior(DataPoint* x, std::vector<float>& probabilities, float & normalization) const
{
    // Get the leaf node
    int leafNode = findLeafNode(x);
    
    normalization = 0;
    probabilities.resize(getHistogram(leafNode).size());
    
    // Copy the histogram
    for (size_t i = 0; i < getHistogram(leafNode).size(); i++)
    {
        probabilities[i] = static_cast<int>(getHistogram(leafNode)[i]);
        normalization += probabilities[i];
    }
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

void RandomForest::classPosterior(DataPoint* x, std::vector<float> & probabilities, float & normalization) const
{
    // The normalization constant will we 1 for this type of
    // decision
    normalization = 1;
    
    std::vector<float> currentProbs;
    float currentNormalization;
    
    // Let the crowd decide
    for (size_t i = 0; i < trees.size(); i++)
    {
        // Get the probabilities from the current tree
        trees[i]->classPosterior(x, currentProbs, currentNormalization);
        
        // Do we need to initialize the result?
        if (i == 0)
        {
            // Does the result have the correct size?
            if (probabilities.size() != currentProbs.size())
            {
                // Nope, correct it
                probabilities.resize(currentProbs.size());
            }
            // Initialize all bins with 0
            for (size_t c = 0; c < probabilities.size(); c++)
            {
                probabilities[c] = 0;
            }
        }
        
        const int C = static_cast<int>(probabilities.size());
        
        // Accumulate the votes
        for (int  c = 0; c < C; c++)
        {
            probabilities[c] += std::log((currentProbs[c] + smoothing)/(currentNormalization + C*smoothing));
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